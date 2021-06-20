"""
Run a model on the NYTWIT dataset.
"""
import argparse
import logging
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import GPT2Tokenizer, PreTrainedTokenizer, BertTokenizer, RobertaTokenizer

from src.downstream.dataset import get_nytwit_dataset
from src.tdt.consts import MAX_WORDLEN
from src.tdt.infer import load_lm_for_inference, load_tdt_for_inference
from src.tdt.utils import set_seed, id_to_token, word_med

N_FOLDS = 10

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.NOTSET)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=496351)
    parser.add_argument('--local-rank', type=int, default=0)  # need for seed
    parser.add_argument('--n-gpu', type=int, default=1)  # need for seed
    parser.add_argument('--model-type', choices=['bert', 'cbert', 'gpt2', 't5', 'xlmr', 'roberta'])
    parser.add_argument('--base-model-dir')
    parser.add_argument('--second-base-model-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--tdt-model-dir', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--logdir')
    parser.add_argument('--dataset', default=None, help='This is to turn off generation by passing None')
    parser.add_argument('--data-loc')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-gen', type=int, default=MAX_WORDLEN,
                        help='longest generated word possible')

    # vectorizer
    parser.add_argument('--vectorizer-type', choices=[None, 'lstm', 'conv', 'transformer'], default=None)
    parser.add_argument('--infer-policy', choices=['all-multi', 'no-easy-suffs', 'all', 'stoch'],
                        default='all-multi', help="Policy for selecting tokens to infer using TDT vectorizer")
    parser.add_argument("--stoch-rate", default=0.0, type=float)
    parser.add_argument('--pool-policy', choices=['max', 'avg', 'first'], default='max',
                        help="Policy for pooling multi-tokens")

    # generator params, possibly not needed for most tasks
    parser.add_argument('--generate-all', action='store_true')
    parser.add_argument('--spaces-end', action="store_true", help="all space characters end generation")

    # task params
    parser.add_argument('--hashtml', action='store_true')
    parser.add_argument('--lowercase', action='store_true')

    args = parser.parse_args()
    set_seed(args)

    if args.tdt_model_dir is not None:
        tdtmod, tmp_dir, chars = load_tdt_for_inference(args)
        btok = tdtmod.btok
        logger.info(f"loaded TDT model with underlying {args.model_type}, "
                    f"device = {tdtmod.device}")
        base_lm = None
    else:
        assert args.vectorizer_type is None
        base_lm, btok, tmp_dir = load_lm_for_inference(args)
        logger.info(f"loaded only {args.model_type} base model, device = {base_lm.device}")
        tdtmod = None

    sents_df = get_nytwit_dataset(args.data_loc)

    vecs = []
    with torch.no_grad():
        for i, x in tqdm(sents_df.reset_index().iterrows(), mininterval=120, desc='NYTWIT inference'):
            if tdtmod is not None:
                inst_vecs = tdt_enc(btok, tdtmod, x)
            else:
                inst_vecs = base_enc(btok, base_lm, x)
            vecs.extend(inst_vecs)

    all_sent_vecs = np.array([x for _, x in vecs])
    logger.info(f"Collected all target vectors, shape: {all_sent_vecs.shape}")

    neolog_df = pd.DataFrame([w for w, _ in vecs], columns=['Word'])
    neolog_df = neolog_df.merge(sents_df[['Word', 'Category', 'fold']].drop_duplicates()) \
        .reset_index(drop=True)

    logger.info(f'Merged tables, total size {len(neolog_df)}')

    models = []
    test_insts = 0
    for fold in range(N_FOLDS):
        train_df = neolog_df.iloc[np.where(neolog_df['fold'] != fold)]
        test_df = neolog_df.iloc[np.where(neolog_df['fold'] == fold)]
        train_Xs = all_sent_vecs[train_df.index]
        test_Xs = all_sent_vecs[test_df.index]
        if len(test_Xs) == 0:
            logger.warning(f'No test instances found for fold {fold}')
            continue
        lm = LogisticRegression(class_weight='balanced', n_jobs=1,
                                multi_class='multinomial', penalty='l2',
                                solver='lbfgs', max_iter=1000)  # elasticnet
        lm.fit(X=train_Xs, y=train_df['Category'])
        preds = list(zip(lm.predict(test_Xs), test_df['Category']))
        models.append((preds, test_df['Word']))
        test_insts += len(test_Xs)

    logger.info(f'Total test instances: {test_insts}')

    cm = defaultdict(Counter)
    mod_matches = []
    all_mod_preds = []
    for prds, words in models:
        total_matches = 0
        for w, (p, g) in zip(words, prds):
            all_mod_preds.append((w, p, g))
            if p == g:
                total_matches += 1
            cm[g][p] += 1
        mod_matches.append(total_matches)

    ypred = [x[1] for x in all_mod_preds]
    ytrue = [x[2] for x in all_mod_preds]
    micf1 = f1_score(ytrue, ypred, average='micro')
    macf1 = f1_score(ytrue, ypred, average='macro')
    logger.info(f'Matches: {sum(mod_matches)}, confusion matrix:\n{cm}')
    logger.info(f'Micro F1: {micf1:.4f}')
    logger.info(f'Macro F1: {macf1:.4f}')


def tdt_enc(btok: PreTrainedTokenizer, tdt_model, inst):
    target_w = inst['Word']
    orig_sent = inst['sentence']
    batch = btok.encode(orig_sent, return_tensors='pt').to(tdt_model.bmodel.device)
    m_out, _, _, inps, joins = tdt_model(batch, mask=False, get_inputs=True, get_joins=True)
    states = m_out[-1][-1][0]  # shape (seqlen, vecdim)

    inst_vecs = []
    sent_align = orig_sent.lower()
    has_cls = type(btok) in [BertTokenizer, RobertaTokenizer]
    k = 1 if has_cls else 0
    bsent = batch[0].cpu().numpy()

    skips = 0
    just_inps = inps[0].cpu().numpy()
    for j, tok in enumerate(just_inps):
        if skips > 0:
            skips -= 1
            continue
        if has_cls and j == 0:
            continue
        assert skips == 0
        if tok == btok.pad_token_id or not sent_align:
            break
        join_id = joins[0, k]
        if join_id >= 0:
            len_to_trunc = 0
            orig_tok_len = 0
            while k < len(bsent) and joins[0, k] == join_id:
                stok = bsent[k]
                len_to_trunc += len(id_to_token(btok, stok, clean=True))
                orig_tok_len += 1
                k += 1
        else:
            orig_tok_len = 1
            len_to_trunc = len(id_to_token(btok, tok, clean=True))
            k += 1
            while (len(just_inps) > j + orig_tok_len) \
                    and word_med(btok, just_inps[j + orig_tok_len], idx_in_seq=j + orig_tok_len):
                len_to_trunc += len(id_to_token(btok, just_inps[j + orig_tok_len], clean=True))
                k += 1
                skips += 1
                orig_tok_len += 1

        if sent_align.startswith(target_w):
            base_vec = states[j:j + orig_tok_len].mean(axis=0)
            inst_vecs.append((target_w, base_vec.reshape(-1).cpu().detach().numpy()))
        sent_align = sent_align[len_to_trunc:].lstrip()

    return inst_vecs


def base_enc(btok, base_model, inst):
    target_w = inst['Word']
    orig_sent = inst['sentence']
    if target_w not in orig_sent and target_w.capitalize() in orig_sent:
        target_w = target_w.capitalize()
    if target_w not in orig_sent and target_w.upper() in orig_sent:
        target_w = target_w.upper()
    if isinstance(btok, GPT2Tokenizer) and f' {target_w}' in orig_sent:
        target_w = f' {target_w}'
    if target_w not in orig_sent:
        logger.warning(f'The word {target_w} is not included in:\n{orig_sent}.')
    targ_ids = btok.encode(target_w, add_special_tokens=False)
    sent_ids = btok.encode(orig_sent)
    len_trg = len(targ_ids)
    states = base_model(torch.tensor([sent_ids], device=base_model.device))[-1]
    inst_vecs = []
    for j, tok in enumerate(sent_ids):
        if sent_ids[j:j + len_trg] == targ_ids:
            base_vec = states[0][:, j:j + len_trg].mean(axis=1).reshape(-1)
            inst_vecs.append((inst['Word'], base_vec.cpu().detach().numpy()))
    return inst_vecs


if __name__ == '__main__':
    main()
