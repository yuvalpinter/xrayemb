"""
Main entry point for training a downstream task model on a base, second-base, or TDT model.
Supports NER, classification, ranking, and a little generation (see todos).
"""
import argparse
import json
import logging
import os
import random
from typing import List

import numpy as np
#import tensorflow as tf
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizer, BertTokenizer, GPT2Tokenizer, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW as tAdamW

from src.downstream.cls_utils import emoji_batch_from_preds, cls_evaluate
from src.downstream.conlleval import ner_evaluate
from src.downstream.consts import *
from src.downstream.dataset import get_dataset, is_cls, is_qa, is_ner, is_gen
from src.downstream.models import NerPredictor, EmojiPredictor, QaScorer, QaGenerator
from src.downstream.ner_utils import excess_tokens, NerHelper, validate_ner_seq
from src.downstream.qa_utils import qa_batch_from_preds, QA_REPORT_HEADER, QAGEN_HEADER, reorg_for_mrg
from src.tdt.consts import *
from src.tdt.infer import load_tdt_for_inference, load_lm_for_inference, call_bmod
from src.tdt.utils import set_seed, word_med, is_single_easy_suffix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.NOTSET)


def main():
    # parse arguments. Most of these are documented in tdt.run_tokdetok.
    parser = argparse.ArgumentParser()

    # model loading, all this is needed until streamlined TODO
    parser.add_argument('--seed', type=int, default=496351)
    parser.add_argument('--local-rank', type=int, default=0)  # need for seed
    parser.add_argument('--n-gpu', type=int, default=1)  # need for seed
    parser.add_argument('--evaluate-file', action='store_true')
    parser.add_argument('--model-type', choices=['bert', 'cbert', 'gpt2', 't5', 'xlmr', 'roberta'])
    parser.add_argument('--base-model-dir', help="location for base model")
    parser.add_argument('--second-base-model-dir', help="location for second base model, i.e. no TDT training")
    parser.add_argument('--out-dir')
    parser.add_argument('--tdt-model-dir', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--logdir', help="location for tensorboard log directory")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument("--lowercase-vocab", action='store_true', help="Vocab to be lowercased - depends on model.")
    parser.add_argument('--max-gen', type=int, default=MAX_WORDLEN,
                        help='longest generated word possible')
    parser.add_argument('--char-emb-size', type=int, default=CHAR_EMB_SIZE)

    parser.add_argument('--verbose', action='store_true', help="")

    # vectorizer
    parser.add_argument('--vectorizer-type', choices=[None, 'lstm', 'conv', 'transformer'], default=None)
    parser.add_argument('--infer-policy', choices=['all-multi', 'no-easy-suffs', 'all', 'stoch'],
                        default='all-multi', help="policy for selecting tokens to infer using TDT vectorizer")
    parser.add_argument("--stoch-rate", default=0.0, type=float,
                        help="if stochastic policy, how often are words vectorized")
    parser.add_argument('--pool-policy', choices=['max', 'avg', 'first'], default='max',
                        help="policy for pooling multi-tokens")

    # generator params, possibly not needed for most tasks
    parser.add_argument('--generate-all', action='store_true', help="generate all tokens to fine-tune Detok")
    parser.add_argument('--spaces-end', action="store_true", help="all space characters end generation")

    # fine-tuning
    parser.add_argument('--finetune', action='store_true',
                        help='keep base LM (and TDT) params training, unfrozen')
    parser.add_argument("--ft-lr", default=2e-5, type=float, help="finetune initial learning rate for Adam.")
    parser.add_argument("--ft-wd", default=0.0, type=float, help="finetune weight decay.")
    parser.add_argument("--ft-adam-eps", default=1e-8, type=float, help="finetune epsilon for Adam optimizer.")
    parser.add_argument('--alpha-tdt', default=0.0, type=float, help="factor for incorporating TDT loss.")
    parser.add_argument("--alpha-vec", default=0.1, type=float, help="finetune weight for vectorizer loss.")
    parser.add_argument("--alpha-gen", default=0.075, type=float, help="finetune weight for generator loss.")

    # actual params for task
    parser.add_argument('--lowercase', action='store_true', help="task text should be lowercased.")
    parser.add_argument('--hashtml', action='store_true', help="task text contains escaped HTML.")
    parser.add_argument("--task-lr", default=5e-4, type=float, help="maximum learning rate for Adam.")
    parser.add_argument("--task-warmup", default=0.1, type=float, help="proportion of steps for upwards tick in LR.")
    parser.add_argument("--task-num-lstm-layers", default=TASK_LSTM_LAYERS, type=int,
                        help="number of LSTM layers in task model")
    parser.add_argument("--task-lstm-hidden-size", default=TASK_LSTM_HIDDEN_SIZE, type=int,
                        help="dimension of each hidden LSTM layer in task model")
    parser.add_argument("--task-num-mlp-layers", default=TASK_MLP_LAYERS, type=int,
                        help="number of MLP layers in task model")
    parser.add_argument('--batch-size', type=int, default=16, help="batch size for task training")
    parser.add_argument('--margin-loss', action='store_true', help='train with a margin loss set to 1.0')
    parser.add_argument('--dataset') #choices=['emoji', 'conll', 'nemer', 'ner', 'cqa',
                                     #        'marcoqa', 'marcosamp', 'marcogen'])
    parser.add_argument('--data-sample', type=float, default=1.0, help="reduce for debugging")
    parser.add_argument('--stopping-patience', type=int, default=STOPPING_PATIENCE,
                        help='number of epochs before early stopping is invoked')
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs for task training")

    args = parser.parse_args()
    set_seed(args)

    is_ft = '' if args.finetune else 'no '
    if args.tdt_model_dir is not None:
        tdtmod, _, chars = load_tdt_for_inference(args)
        emb_size = tdtmod.out_size()
        btok = tdtmod.btok
        base_lm = None
        logger.info(f"loaded TDT model with underlying {args.model_type}, {type(btok)} tokenizer, "
                    f"{is_ft}fine-tuning, device = {tdtmod.device}")
        main_model = tdtmod
    else:
        assert args.vectorizer_type is None
        assert args.alpha_tdt == 0.0, f"alpha-tdt needs to be set to zero, not {args.alpha_tdt}"
        tdtmod, chars = None, None
        base_lm, btok, _ = load_lm_for_inference(args)
        emb_size = base_lm.get_output_embeddings().in_features
        logger.info(f"loaded only {args.model_type} base model, {is_ft}fine-tuning, device = {base_lm.device}")
        main_model = base_lm

    tb_writer = SummaryWriter(log_dir=args.logdir)

    # load datasets
    ds = get_dataset(args)
    dstr = ds["train"]
    dsdv = ds["dev"]
    dsts = ds["test"]
    lab_list = list(set(dstr.labels + dsdv.labels)) + [UNK_LABEL]
    labs = {l: i for i, l in enumerate(lab_list)}
    logger.info(f'dataset {args.dataset}:train has {len(labs)} label types.')
    bvocab = dstr.vocab()
    logger.info(f'dataset {args.dataset}:train has {len(bvocab)} word types '
                f'with {sum(bvocab.values())} tokens.')
    logger.info(f'dataset {args.dataset}:dev has {len(dsdv.vocab())} word types '
                f'with {sum(dsdv.vocab().values())} tokens.')
    logger.info(f'dataset {args.dataset}:test has {len(dsts.vocab())} word types '
                f'with {sum(dsts.vocab().values())} tokens.')

    # tensify datasets
    if type(btok) == GPT2Tokenizer:
        logger.warning("Manually adding BOS tokens for GPT2 processing but not EOS; consider implementing for"
                       " sequence-level tasks.")
    if is_gen(args) and args.tdt_model_dir is not None:
        char_table = {c: i for i, c in enumerate(chars)}
        logger.info(f'Using character table to initialize generator')
    else:
        char_table = None
    tr_set = tensify_batch(dstr, labs, btok, args, char_table)
    dv_set = tensify_batch(dsdv, labs, btok, args, char_table, get_all_toks=True)
    ts_set = tensify_batch(ds['test'], labs, btok, args, char_table, get_all_toks=True)
    logger.info(f'training dataset device is {tr_set[0][0].device}')

    # initialize model
    if is_ner(args):
        mod = NerPredictor(args, emb_size, len(labs))
    elif is_cls(args):
        mod = EmojiPredictor(args, emb_size, len(labs))
    elif is_qa(args):
        mod = QaScorer(args, emb_size)
    elif is_gen(args):
        mod = QaGenerator(args, emb_size, space_char_idx=char_table[' '], eow_index=char_table[BOUND_CHAR],
                          gen=tdtmod.tdtgen)
    else:
        raise NotImplementedError('unknown dataset parameter')

    if is_ner(args):
        ner_helper = NerHelper(btok, easy_suffs=args.infer_policy == 'no-easy-suffs',
                               stoch=args.infer_policy == 'stoch',
                               is_tdt=tdtmod is not None)

    # calculate steps for optimizer
    t_total = len(tr_set) * args.epochs
    logger.info(f'Total (maximum) number of steps: {t_total}')
    if args.task_warmup > 1.0:
        raise AttributeError('task-warmup parameter is proportion of total steps')
    t_warm = int(args.task_warmup * t_total)
    logger.info(f'Warmup steps: {t_warm}')

    # finetuning optimizer
    if args.finetune:
        logger.info('Finetuning base model')
        if tdtmod is not None and args.alpha_tdt > 0.0:
            logger.info('Using MSE loss to tune TDT tokenizer, XEnt for generator')
            distloss = torch.nn.MSELoss()
            genloss = torch.nn.CrossEntropyLoss()
        else:
            distloss, genloss = None, None

        # optimizer, scheduler
        all_named_params = main_model.named_parameters()

        # complicated decay-sensitive regime that isn't working too well
        # doesn't work when commented out, maybe there's a side effect?
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in all_named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": args.ft_wd,
            },
            {"params": [p for n, p in all_named_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        lm_opt = tAdamW(optimizer_grouped_parameters, lr=args.ft_lr, eps=args.ft_adam_eps)
        lm_sch = get_linear_schedule_with_warmup(lm_opt, num_warmup_steps=0, num_training_steps=t_total)
    else:
        distloss, genloss, lm_opt, lm_sch = None, None, None, None
        args.alpha_tdt = 0.0

    finetune_all_tdt = args.finetune and args.alpha_tdt > 0.0

    # task optimizer
    params_for_main_opt = mod.parameters()
    opt = AdamW(params_for_main_opt, lr=args.task_lr)
    sch = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=t_warm, num_training_steps=t_total
    )

    # init loss
    if is_qa(args):
        if args.margin_loss:
            mrg_loss = nn.MarginRankingLoss(margin=1.0)
            logger.info('Using Margin loss for training model')
            bce_loss = None
        else:
            bce_loss = nn.BCEWithLogitsLoss()
            logger.info('Using BCE logit loss for training model')
            mrg_loss = None
        xent_loss = None
    else:
        xent_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        logger.info('Using Cross-entropy loss for training model')
        bce_loss = None
        mrg_loss = None

    # For early stopping
    epoch_metrics = []
    best_model = None
    best_dv_report = None
    best_dv_prd = None
    max_met = -np.inf

    # for reporting
    excess_toks = excess_tokens(args.model_type)

    # training
    for ep in trange(args.epochs, mininterval=120, desc='Training epoch'):
        mod.train()
        if args.finetune:
            main_model.train()

        ep_loss = 0.0
        tdt_tot_loss = 0.0
        random.shuffle(tr_set)
        #logger.info(tr_set[0][0][0].cpu().numpy())
        #logger.info(tr_set[0][1][0].cpu().numpy())
        #logger.info(tr_set[0][2][0])
        for batch, targets, sents in tqdm(tr_set, mininterval=120, desc='Training batches'):
            mod.zero_grad()
            if args.finetune:
                main_model.zero_grad()
            if tdtmod is not None:
                m_out, vec_batch, gen_batch, inps, joins = tdtmod(batch, mask=False,
                                                                  get_inputs=True, get_joins=True)
            else:
                m_out, inps = call_bmod(base_lm, batch)
                vec_batch, gen_batch, joins = None, None, None
            if m_out is None:
                continue

            if finetune_all_tdt:
                lm_lss = m_out[0].to(args.device)  # is zero when there's no masks
                if vec_batch is not None:
                    if len(vec_batch) > 2:
                        vec_batch = vec_batch[:2]
                    vec_lss = distloss(*vec_batch).to(args.device)
                else:
                    vec_lss = torch.tensor(0.0, requires_grad=True).to(args.device)
                if gen_batch is not None:
                    gen_lss = genloss(*gen_batch).to(args.device)
                else:
                    gen_lss = torch.tensor(0.0, requires_grad=True).to(args.device)
                tdt_loss = lm_lss + \
                           args.alpha_vec * vec_lss + \
                           args.alpha_gen * gen_lss
                tdt_loss_item = tdt_loss.item()
                tdt_tot_loss += tdt_loss_item
            else:
                tdt_loss = None

            if m_out[-1] is None:
                snt_lines = "\n".join(sents)
                logger.info(f'Base model output no embeddings at:\n{snt_lines}')
                if finetune_all_tdt:
                    tdt_loss.backward()
                    lm_opt.step()
                    lm_sch.step()
                continue

            lm_out_class = m_out[-1][-1].detach()
            assert lm_out_class.shape[:2] == inps.shape[:2]

            if is_ner(args):
                targets = ner_helper.align_targets(targets, inps, joins)
                validate_ner_seq(btok, excess_toks, inps, targets, joins, sents)
                preds = mod(lm_out_class).permute(0, 2, 1)
                batch_loss = xent_loss(preds, targets)
            elif is_cls(args) or is_qa(args):
                mod_out = model_out_for_classification(lm_out_class, inps, btok.pad_token_id, args.model_type)
                preds = mod(mod_out)
                if is_qa(args):
                    if args.margin_loss:
                        preds_t, preds_f, mrg_targs = reorg_for_mrg(preds, targets, args.device)
                        batch_loss = mrg_loss(preds_t, preds_f, mrg_targs)
                    else:
                        preds = preds.view(1, -1)
                        batch_loss = bce_loss(preds, targets)
                else:
                    batch_loss = xent_loss(preds, targets)
            elif is_gen(args):
                mod_out = model_out_for_generation(lm_out_class, inps, btok.pad_token_id, args.model_type)
                batch_loss = torch.tensor([0.0], device=args.device)
                for q, binst in zip(sents, mod(mod_out, targets)):
                    for prds, trgs in binst:
                        inst_loss = xent_loss(prds, trgs)
                        batch_loss += inst_loss
            else:
                raise NotImplementedError('Dataset type not supported')

            ep_loss += (len(batch) * batch_loss.item())

            if finetune_all_tdt:
                batch_loss += (args.alpha_tdt * tdt_loss)

            # backprop
            batch_loss.backward()
            opt.step()
            sch.step()
            if args.finetune:
                lm_opt.step()
                lm_sch.step()

        logger.info(f'Training loss after epoch {ep + 1} = {ep_loss:.3f}')
        tb_writer.add_scalar("train loss", ep_loss, ep + 1)

        lr_val = sch.get_last_lr()[0]
        logger.info(f'learning rate = {lr_val}')
        tb_writer.add_scalar("learning rate", lr_val, ep + 1)

        if args.finetune:
            if args.alpha_tdt > 0.0:
                logger.info(f'TDT loss after epoch {ep + 1} (x100) = {100 * tdt_tot_loss:.3f}')
                tb_writer.add_scalar("TDT loss", tdt_tot_loss, ep + 1)

            lm_lr_val = lm_sch.get_last_lr()[0]
            logger.info(f'main model learning rate = {lm_lr_val}')
            tb_writer.add_scalar("main model learning rate", lm_lr_val, ep + 1)

        # dev
        with torch.no_grad():
            mod.eval()
            if args.finetune:
                main_model.eval()
            dv_loss = 0.0
            true_lst = []
            pred_lst = []
            if is_qa(args):
                rank_lst = []
                counts_lst = []
                dv_prd_strs = [QA_REPORT_HEADER]
            elif is_gen(args):
                dv_prd_strs = [QAGEN_HEADER]
            elif is_cls(args):
                dv_prd_strs = []
            for batch, targets, sents in tqdm(dv_set, mininterval=120, desc='Dev eval'):
                if tdtmod is not None:
                    m_out, _, _, inps, joins = tdtmod(batch, mask=False, get_inputs=True, get_joins=True)
                else:
                    m_out, inps = call_bmod(base_lm, batch)
                    joins = None

                if m_out is None or m_out[-1] is None:
                    if args.verbose:
                        if is_qa(args):
                            snt_lines = "\n".join(sents[0])
                        else:
                            snt_lines = "\n".join(sents)
                        logger.info(f'Base model output no embeddings at:\n{snt_lines}')

                    if is_qa(args):
                        preds = torch.randn_like(targets, device=args.device)
                        t_l, p_l, ranks, counts, dv_prd = qa_batch_from_preds(sents, preds, targets)
                    else:
                        # generating results is less important because test set exists
                        continue

                else:
                    lm_out_class = m_out[-1][-1]
                    assert lm_out_class.shape[:2] == inps.shape[:2]
                    if is_gen(args):
                        mod_out = model_out_for_generation(lm_out_class, inps, btok.pad_token_id, args.model_type)
                        batch_loss = torch.tensor([0.0], device=args.device)
                        t_l = []
                        p_l = []
                        for q, binst in zip(sents, mod(mod_out, targets, forcing=False)):
                            qtrgs = []
                            qprds = []
                            for gend, trgs in binst:
                                trg_str = ''.join([chars[i] for i in trgs[0]])
                                prd_str = ''.join([chars[i] for i in gend.tolist()[0]])
                                # TODO BLEU or something similar
                                qtrgs.append(trg_str)
                                qprds.append(prd_str)
                            # TODO change all this when full answer is given with no word alignment
                            strg = ' '.join(qtrgs)
                            sprd = ' '.join(qprds)
                            t_l.append(strg)
                            p_l.append(sprd)
                            dv_prd_strs.append('\t'.join([q, strg, sprd]))
                    elif is_ner(args):
                        targets = ner_helper.align_targets(targets, inps, joins)
                        validate_ner_seq(btok, excess_toks, inps, targets, joins, sents)
                        preds = mod(lm_out_class)
                        batch_loss = xent_loss(preds.permute(0, 2, 1), targets)
                        t_l, p_l, _ = ner_helper.ner_batch_from_preds(sents, preds, targets, lab_list, joins)
                    else:
                        mod_out = model_out_for_classification(lm_out_class, inps, btok.pad_token_id, args.model_type)
                        preds = mod(mod_out)
                        if is_cls(args):
                            batch_loss = xent_loss(preds, targets)
                            t_l, p_l, p_strs = emoji_batch_from_preds(sents, preds, targets)
                        elif is_qa(args):
                            if args.margin_loss:
                                preds_t, preds_f, mrg_targs = reorg_for_mrg(preds, targets, args.device)
                                batch_loss = mrg_loss(preds_t, preds_f, mrg_targs)
                            else:
                                preds = preds.view(1, -1)
                                batch_loss = bce_loss(preds, targets)
                            t_l, p_l, ranks, counts, dv_prd = qa_batch_from_preds(sents, preds, targets)
                            dv_prd_strs.extend(dv_prd)
                    dv_loss += (len(batch) * batch_loss.item())
                true_lst.extend(t_l)
                pred_lst.extend(p_l)
                if is_cls(args):
                    dv_prd_strs.extend(p_strs)
                if is_qa(args):
                    rank_lst.extend(ranks)
                    counts_lst.extend(counts)

            logger.info(f'Dev loss: {dv_loss:.3f}')
            tb_writer.add_scalar("dev loss", dv_loss, ep + 1)

            if is_ner(args):
                (acc, no_o_acc, _, _, mic_f1, mac_f1), dv_report = ner_evaluate(true_lst, pred_lst, verbose=False)
                logger.info(f'dev accuracy: {acc:.4f}, acc-no-O: {no_o_acc:.4f}, '
                            f'micro F1: {mic_f1:.4f}, macro F1: {mac_f1:.4f}')
                epoch_metrics.append(mic_f1)

                tb_writer.add_scalar("dev micro f1", mic_f1, ep + 1)
                tb_writer.add_scalar("dev macro f1", mac_f1, ep + 1)

            elif is_cls(args):
                acc, mic_f1, mac_f1 = cls_evaluate(true_lst, pred_lst)
                logger.info(f'dev accuracy: {acc:.4f}, dev micro f1: {mic_f1:.4f}, dev macro f1: {mac_f1:.4f}')
                epoch_metrics.append(mac_f1)

                tb_writer.add_scalar("dev micro f1", mic_f1, ep + 1)
                tb_writer.add_scalar("dev macro f1", mac_f1, ep + 1)

            elif is_qa(args):
                acc = len([p for p, g in zip(true_lst, pred_lst) if p == g]) / len(true_lst)
                logger.info(f'dev accuracy: {acc:.4f}')
                mr = 1. + (sum(rank_lst) / len(rank_lst))
                max_mr = sum(counts_lst) / len(counts_lst)
                rrs = [1. / (1. + r) for r in rank_lst]
                mrr = np.average(rrs)
                logger.info(f'mean rank: {mr:.4f} of possible {max_mr:.4f}; mean reciprocal rank: {mrr:.4f}')
                tb_writer.add_scalar("dev mean rank percentile", mr / max_mr, ep + 1)
                tb_writer.add_scalar("dev mean reciprocal rank", mrr, ep + 1)
                epoch_metrics.append(mrr)

            elif is_gen(args):
                # TODO BLEU etc.
                corrects = [p for p, g in zip(true_lst, pred_lst) if p == g]
                acc = len(corrects) / len(true_lst)
                logger.info(f'dev accuracy: {acc:.4f}')
                for p in corrects:
                    logger.info(f'Correctly predicted:\t{p}')
                epoch_metrics.append(acc)

            tb_writer.add_scalar("dev acc", acc, ep + 1)

            max_met = max(epoch_metrics)
            if epoch_metrics[-1] == max_met:
                prev_best = best_model
                best_model = os.path.join(args.out_dir, f'down_model_{ep + 1:04}.pt')
                if is_ner(args):
                    best_dv_report = dv_report
                if is_cls(args):
                    inst_delim = ''
                    best_dv_prd = inst_delim.join(dv_prd_strs)
                elif is_qa(args):
                    inst_delim = '\n'
                    best_dv_prd = inst_delim.join(dv_prd_strs)
                elif is_gen(args):
                    inst_delim = '\n'
                    best_dv_prd = inst_delim.join(dv_prd_strs)
                logger.info(f'saving best model so far to {best_model}, removing {prev_best}')
                torch.save(mod.state_dict(), best_model)
                if prev_best is not None:
                    os.remove(prev_best)

            elif max(epoch_metrics[-args.stopping_patience:]) < max_met:
                break

    # reloading best
    logger.info(f'epoch metrics: {epoch_metrics}, max = {max_met}')
    logger.info(f'loading best model from {best_model}')
    # TODO also load (and save!) main LM and TDT, if there's finetuning
    sd = torch.load(best_model)  # map_location param?
    mod.load_state_dict(sd)

    # saving config and best model
    cfg = json.dumps(mod.config_params(), indent=2, sort_keys=True) + "\n"
    cfg_path = os.path.join(args.out_dir, 'config.json')
    with open(cfg_path, "w", encoding="utf-8") as writer:
        writer.write(cfg)
    #tf.io.gfile.copy(cfg_path, os.path.join(args.out_dir, 'config.json'))
    #tf.io.gfile.copy(best_model, os.path.join(args.out_dir, 'model.pt'))

    if is_qa(args) or is_gen(args) or is_cls(args):
        #out_tmpfile = os.path.join(tmp_dir, 'dev_preds.tsv')
        out_filename = os.path.join(args.out_dir, 'dev_preds.tsv')
        logger.info(f'Writing best dev predictions to {out_filename}.')
        with open(out_filename, 'w') as outf:
            outf.write(best_dv_prd)
        #tf.io.gfile.copy(out_tmpfile, out_filename)

    # test
    if not ts_set:  # MARCO-QA is not annotated
        tb_writer.close()
        return

    with torch.no_grad():
        mod.eval()
        if args.finetune:
            main_model.eval()
        ts_loss = 0.0
        pred_strs = [QA_REPORT_HEADER] if is_qa(args) else []
        true_lst = []
        pred_lst = []
        rank_lst = []
        for batch, targets, sents in tqdm(ts_set, mininterval=20, desc='Test eval'):
            if tdtmod is not None:
                m_out, _, _, inps, joins = tdtmod(batch, mask=False, get_inputs=True, get_joins=True)
            else:
                m_out, inps = call_bmod(base_lm, batch)

            if m_out is None or m_out[-1] is None:
                snt_lines = "\n".join(sents)
                logger.info(f'Base model output no embeddings at:\n{snt_lines}')
                continue

            lm_out_class = m_out[-1][-1]
            assert lm_out_class.shape[:2] == inps.shape[:2]
            if is_ner(args):
                targets = ner_helper.align_targets(targets, inps, joins)
                validate_ner_seq(btok, excess_toks, inps, targets, joins, sents)
                preds = mod(lm_out_class)
                batch_loss = xent_loss(preds.permute(0, 2, 1), targets)

                t_l, p_l, p_strs = ner_helper.ner_batch_from_preds(sents, preds, targets, lab_list, joins)
            else:
                mod_out = model_out_for_classification(lm_out_class, inps, btok.pad_token_id, args.model_type)
                preds = mod(mod_out)
                if is_cls(args):
                    batch_loss = xent_loss(preds, targets)
                    t_l, p_l, p_strs = emoji_batch_from_preds(sents, preds, targets, inps, btok)
                elif is_qa(args):
                    if args.margin_loss:
                        preds_t, preds_f, mrg_targs = reorg_for_mrg(preds, targets, args.device)
                        batch_loss = mrg_loss(preds_t, preds_f, mrg_targs)
                    else:
                        preds = preds.view(1, -1)
                        batch_loss = bce_loss(preds, targets)
                    t_l, p_l, ranks, counts, p_strs = qa_batch_from_preds(sents, preds, targets)
            ts_loss += (len(batch) * batch_loss.item())
            true_lst.extend(t_l)
            pred_lst.extend(p_l)
            if is_qa(args):
                rank_lst.extend(ranks)
            pred_strs.extend(p_strs)

        logger.info(f'Test loss: {ts_loss:.3f}')

        if is_ner(args):
            _, report = ner_evaluate(true_lst, pred_lst)  # includes logging individual class metrics
        elif is_cls(args):
            acc, mic_f1, mac_f1 = cls_evaluate(true_lst, pred_lst)
            logger.info(f'test accuracy: {acc:.4f}, micro f1: {mic_f1:.4f}, macro f1: {mac_f1:.4f}')
        elif is_qa(args):
            acc = len([p for p, g in zip(true_lst, pred_lst) if p == g]) / len(true_lst)
            logger.info(f'test accuracy: {acc:.4f}')
            mr = 1. + (sum(rank_lst) / len(rank_lst))
            logger.info(f'mean rank: {mr:.4f}')
            rrs = [1. / (1. + r) for r in rank_lst]
            mrr = np.average(rrs)
            logger.info(f'mean reciprocal rank: {mrr:.4f}')

        #out_tmpfile = os.path.join(tmp_dir, 'test_preds.tsv')
        out_filename = os.path.join(args.out_dir, 'test_preds.tsv')
        logger.info(f'Writing predictions to {out_filename}.')
        with open(out_filename, 'w') as outf:
            if is_qa(args):
                inst_delim = '\n'
            elif is_ner(args):
                inst_delim = '\n\n'
            else:
                inst_delim = ''
            outf.write(inst_delim.join(pred_strs))
        #tf.io.gfile.copy(out_tmpfile, out_filename)

        if is_ner(args):
            #rep_tmpfile = os.path.join(tmp_dir, 'report.txt')
            rep_filename = os.path.join(args.out_dir, 'report.txt')
            logger.info(f'Writing metrics report to {rep_filename}.')
            with open(rep_filename, 'w') as outf:
                if best_dv_report is not None:
                    outf.write("Dev:\n\n")
                    outf.write("\n".join(best_dv_report))
                    outf.write("\n\n")
                outf.write("Test:\n\n")
                outf.write("\n".join(report))
            #tf.io.gfile.copy(rep_tmpfile, rep_filename)

    tb_writer.close()


def model_out_for_generation(out_vecs, inputs, pad_tok_id, model_type):
    return model_out_for_classification(out_vecs, inputs, pad_tok_id, model_type)


def model_out_for_classification(out_vecs, inputs: torch.Tensor,
                                 pad_tok_id: int, model_type: str) -> torch.Tensor:
    """
    Mush output from language model for the task model
    :param out_vecs: LM output
    :param inputs: original inputs (for tracking)
    :param pad_tok_id: padding token ID, passed from tokenizer
    :param model_type: "enum" value for model type
    :return: torch tensor, but edible to downstream task
    """
    if model_type in ['bert', 'cbert', 'roberta']:
        # take [CLS] output
        return out_vecs[:, 0]
    elif model_type == 'gpt2':
        # take last unpadded output
        last_idcs = [max([i for i, e in enumerate(emb_ids) if e != pad_tok_id]) for emb_ids in inputs]
        return out_vecs[list(range(len(out_vecs))), last_idcs]
    raise NotImplementedError('Define classification input for model type.')


def tensify_batch(dataset, lab_dict, tokzr: PreTrainedTokenizer, args, char_table=None, get_all_toks=False):
    """
    Take instances, return batches
    :param dataset: dataset object containing instances
    :param lab_dict: label dictionary for the task
    :param tokzr: tokenizer
    :param args: user-supplied arguments
    :param char_table: mapping from characters to indices
    :param get_all_toks: if true, all tokens are returned in `sents`, not just main-problem tokens
    :return: sequence of batches, each containing tuples of the form:
        idc_tns - tensor of indices for torch module operation
        targets - correct target labels
        sents - texts for debugging and generation
    """
    batches = []
    batch_size = args.batch_size
    annotate_all_toks = args.vectorizer_type is None or args.infer_policy == 'stoch'
    if is_qa(args):
        for d in tqdm(dataset, mininterval=120, desc='Dataset tensification'):
            if not d.true_passage:
                continue

            # TODO maybe also condition on query type
            passages = zip([1] + [0] * len(d.false_passages),
                           [(d.query, d.true_passage)] + [(d.query, f) for f in d.false_passages])
            len_sorted_passages = list(sorted(passages, key=lambda x: -len(tokzr.tokenize(x[1][1]))))
            assert len(len_sorted_passages) <= batch_size, f'{len(len_sorted_passages)} > {batch_size}'
            idc_tns = encode_seqs(tokzr, len_sorted_passages, 1, False, args.device)
            batches.append((idc_tns, torch.tensor([x[0] for x in len_sorted_passages],
                                                  dtype=torch.float32, device=args.device).view(1, -1),
                            ([' [SEP] '.join(x[1]) for x in len_sorted_passages], d)))
        return batches
    if is_gen(args):
        queries = []
        answers = []
        for d in tqdm(dataset, mininterval=120, desc='Dataset tensification'):
            if not d.wf_answer:
                continue
            sep = f' {tokzr.sep_token} ' if isinstance(tokzr, BertTokenizer) or isinstance(tokzr, RobertaTokenizer) else '. '
            queries.append(d.true_passage + sep + d.query)
            answers.append(d.wf_answer)
        len_sorted_insts = sorted(list(zip(queries, answers)), key=lambda x: -len(x[0]))
    elif is_ner(args):
        if isinstance(tokzr, GPT2Tokenizer):  # TODO maybe this should also be done for BERT
            # join and retokenize because GPT2 marks spaces on tokens
            # NOTE THIS INCLUDES ROBERTA WHICH IS A SUBCLASS OF GPT2TOKENIZER
            n_dataset = []
            for s, l in dataset:
                #cobbled_s = tokzr.tokenize(' '.join(s))
                cobbled_s = tokzr.encode(' '.join(s))
                n_dataset.append((cobbled_s, l))
            dataset = n_dataset
        len_sorted_insts = sorted(dataset, key=lambda x: -len(x[0]))
    elif is_cls(args):
        len_sorted_insts = sorted(dataset, key=lambda x: -len(tokzr.tokenize(x[0])))
    else:
        raise NotImplementedError('Dataset type not supported')
    for i in range(0, len(len_sorted_insts) - 1, batch_size):
        # TODO handle cases of length 1&2 which get read as two-sequence inputs by batch_encode_plus()
        # this is inexact for emoji, since it computes string length.
        to_batch = [x for x in len_sorted_insts[i:i + batch_size] if len(x[0]) > 2]
        if not to_batch:
            continue

        idc_tns = encode_seqs(tokzr, to_batch, 0, is_ner(args), args.device)
        idc_tns = idc_tns.to(args.device)

        if is_ner(args):
            # align tags and subword tokens
            targets = torch.full_like(idc_tns, IGNORE_INDEX, dtype=torch.int64, device=args.device)

            for j, s in enumerate(to_batch):
                tj = targets[j]
                idcj = idc_tns[j].cpu().numpy().tolist()
                k = 1  # lookup in indices and targets, ignoring the [CLS]/<BOS>
                tags = s[1]
                t = 0  # lookup in tags
                u = 1  # index to update, ignoring the [CLS]/<BOS>
                curr_tag = None
                while t < len(tags):
                    while word_med(tokzr, idcj[k], idx_in_seq=k - 1):
                        #logger.info(f"word-med in {k}={idcj[k]}")
                        if not annotate_all_toks:
                            #logger.info(f"not annotate_all_toks!")
                            # not pretty, may be replaced with passing the filter in as an argument
                            if (args.infer_policy != 'no-easy-suffs'
                                    or not is_single_easy_suffix(tokzr, idcj, k)):
                                # this has been [VECTORIZED] and targets followed suit
                                k += 1
                                continue
                        tj[u] = curr_tag
                        u += 1
                        k += 1  # crucially, t isn't advanced
                    if tags[t] not in lab_dict:
                        logger.info(f'Found unknown tag in: {s}')
                    curr_tag = lab_dict.get(tags[t], lab_dict[UNK_LABEL])
                    tj[u] = curr_tag
                    t += 1
                    u += 1
                    k += 1
                if ((annotate_all_toks or args.infer_policy == 'no-easy-suffs')
                        and isinstance(tokzr, GPT2Tokenizer)):  # includes RoBERTa
                    while k < len(idcj) and word_med(tokzr, idcj[k], idx_in_seq=k - 1):
                        #logger.info(f"word-med in {k}={idcj[k]}")
                        if not annotate_all_toks and not is_single_easy_suffix(tokzr, idcj, k):
                            break
                        tj[u] = curr_tag
                        u += 1
                        k += 1
                #logger.info(f"u={u}")
                #logger.info(tj.cpu().numpy())
            #logger.info(targets[0].cpu().numpy())
            #exit()
        elif is_cls(args):
            targets = torch.tensor([x[1] for x in to_batch], dtype=torch.int64, device=args.device)
        elif is_gen(args):
            if args.tdt_model_dir is not None:
                targets = [tensify_chars(x[1], args.device, char_table=char_table) for x in to_batch]
            else:
                targets = [tensify_chars(x[1], args.device, tokzr=tokzr) for x in to_batch]
        else:
            raise NotImplementedError('Dataset type not supported')
        if is_gen(args):
            sents = [x[0] for x in to_batch]
        elif get_all_toks:
            sents = to_batch
        else:
            if is_ner(args):
                #sents = [' '.join(w) for w, t in to_batch]
                sents = [' '.join(tokzr.convert_ids_to_tokens(w)) for w, t in to_batch]
            elif is_cls(args) or is_gen(args):
                sents = [x[0] for x in to_batch]
            else:
                raise NotImplementedError('Dataset type not supported')
        batches.append((idc_tns, targets, sents))
    return batches


def tensify_chars(seq, device: torch.device, char_table: dict = None, tokzr: PreTrainedTokenizer = None):
    """
    :param seq: first sentence of this provided sequence will be tensified
    :param device: torch device
    :param char_table: mapping from character vocabulary to indices
    :param tokzr: tokenizer
    :return: tensor of character indices ready for torch module operation
    """
    seq = seq.split('\n')[0]
    if char_table is not None:
        return torch.tensor([char_table.get(c, char_table[UNK_CHAR]) for c in seq], dtype=torch.int64, device=device)
    elif tokzr is not None:
        return tokzr.encode_plus(seq, return_tensors='pt')['input_ids']
    raise AttributeError('Must supply either char_table or tok')


def encode_seqs(tokzr, passages: List[str], txt_idx, is_presplit, device: torch.device):
    """
    :param passages: tuples containing strings to be encoded as one of their entries (or token id ints for NER)
    :param tokzr: tokenizer
    :param txt_idx: entry index for text
    :param device: torch device
    :return: token index tensor ready for torch module operation
    """
    raw_batch = [s[txt_idx] for s in passages]
    #logger.info(raw_batch)
    #if is_presplit:
    #    raw_batch = [' '.join(x) for x in raw_batch]
    #logger.info(raw_batch)
    if is_presplit:
        idc_tns = tokzr.pad([{'input_ids': b} for b in raw_batch], padding='max_length', return_tensors='pt')['input_ids']
    else:
        idc_tns = tokzr.batch_encode_plus(raw_batch,
                                        padding='max_length', #pad_to_max_length=True, is_split_into_words=is_presplit,
                                        max_length=tokzr.model_max_length,
                                        return_tensors='pt')['input_ids']
    if type(tokzr) not in [BertTokenizer, RobertaTokenizer]:
        if isinstance(tokzr, GPT2Tokenizer):
            bos_s = torch.full((idc_tns.shape[0], 1), tokzr.bos_token_id, dtype=torch.int64)
            idc_tns = torch.cat([bos_s, idc_tns], dim=1)
        else:
            raise NotImplementedError("Proper text encoding with BOS and EOS for non-[Ro]BERT[a]/GPT2 models"
                                      " is still not implemented")
    if idc_tns.shape[1] > tokzr.model_max_length:   # tokzr.max_len
        logger.info(idc_tns.shape)
        idc_tns = idc_tns[:, :tokzr.model_max_length]   # tokzr.max_len
    return idc_tns.to(device)


if __name__ == '__main__':
    main()
