"""
Analyze multi-token vs. single-token performance (fairly old)
"""

import argparse
import logging
import random

import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

from src.tdt.io_utils import load_model_files, DOC_FILE
from src.tdt.nlp_utils import NlpTokenizer
from src.tdt.utils import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.NOTSET)


def sample_sentences(fname, pretokenizer, portion=0.01, column=-1):
    with tf.io.gfile.GFile(fname, mode="r") as f:
        for line in tqdm(f, mininterval=120):
            if random.random() > portion:
                continue
            lds = line.strip()
            if len(lds) >= 0:
                if column >= 0 and '\t' in lds:
                    lds = lds.split('\t')[column]
                yield pretokenizer(lds)


def get_non_init_toks(btok):
    return [t for w, t in btok.vocab.items() if w.startswith('##')]


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=496351)
    parser.add_argument('--local-rank', type=int, default=0)  # need for seed
    parser.add_argument('--n-gpu', type=int, default=1)       # need for seed
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-size', type=float, default=0.01)

    parser.add_argument('--doc-file', default=DOC_FILE)
    parser.add_argument('--column', type=int, default=-1, help="column selection for tab-delimited input files")
    parser.add_argument('--model-dir')
    parser.add_argument('--out-dir')                          # not used for now

    args = parser.parse_args()
    set_seed(args)

    # sample sentences
    hashtml = args.column >= 0
    pretok = NlpTokenizer(hashtml=hashtml)
    sents = list(sample_sentences(args.doc_file, pretok, portion=args.sample_size, column=args.column))
    logger.info(f'loaded {len(sents)} sentences.')

    # load bert for masked lm
    btok, bmod, _ = load_model_files(args.model_type, args.model_dir)
    bmod.to(args.device)
    mask_id = btok.mask_token_id
    logger.info(f'loaded models.')

    # prepare tokenizer
    non_init_toks = get_non_init_toks(btok)
    logger.info(f'found {len(non_init_toks)} non-init tokens out of a vocabulary of {len(btok)}.')

    t_preds = []
    f_preds = []

    # iterate over sample sentences
    for s in sents:
        logger.info('======')
        logger.info(s)
        logger.info(btok.tokenize(s))
        toks = btok.encode_plus(s, return_tensors='pt')['input_ids']
        m_toks = torch.tensor(toks, device=args.device)
        tok_ids = toks[0].cpu().numpy()
        s_t_preds = []
        s_f_preds = []
        for i, (t, tn) in enumerate(zip(tok_ids, tok_ids[1:])):
            if t in btok.all_special_ids:
                continue
            rt = btok.ids_to_tokens[t]
            rtn = btok.ids_to_tokens[tn]

            m_toks[0][i] = mask_id
            res_sm = bmod(m_toks)[0][0][i].softmax(0)
            raw_prob = res_sm[t].item()
            non_init_prob_sum = sum(res_sm[non_init_toks]).item()
            logger.info(f'raw prob for token {i}: {rt} ({rtn}) = {raw_prob:.5f}; '
                        f'total for non-initial = {non_init_prob_sum:.5f}.')

            # construct after-token prediction, collect class label
            in_pref = m_toks[0][:i+1].cpu().numpy() + [mask_id]
            if tn in non_init_toks:  # add to "true" probs
                cls = 1
                next_init_tok = min([j for j, tt in enumerate(tok_ids) if j > i and tt not in non_init_toks])
                in_suf = m_toks[0][next_init_tok:].cpu().numpy()
            else:  # add to "false" probs
                cls = -1
                in_suf = m_toks[0][i+1:].cpu().numpy()
            pred_inp = torch.tensor(np.concatenate([in_pref, in_suf]), device=args.device).view(1, -1)
            prd_res = bmod(pred_inp)[0][0][i+1].softmax(0)
            non_init_prd_sum = sum(prd_res[non_init_toks]).item()
            logger.info(f'total sum for next-up prediction of class {cls}: {non_init_prd_sum:.5f}')

            if cls == 1:
                s_t_preds.append(non_init_prd_sum)
            else:
                s_f_preds.append(non_init_prd_sum)

        if any(s_t_preds):
            logger.info(f'Prediction average/std for true instances: {np.mean(s_t_preds)}, {np.std(s_t_preds)}')
        logger.info(f'Prediction average/std for false instances: {np.mean(s_f_preds)}, {np.std(s_f_preds)}')

        t_preds.extend(s_t_preds)
        f_preds.extend(s_f_preds)

    logger.info('=============================')
    logger.info(f'Overall average/std for true instances: {np.mean(t_preds)}, {np.std(t_preds)}')
    logger.info(f'Overall average/std for false instances: {np.mean(f_preds)}, {np.std(f_preds)}')

    # TODO more info: how many multitok words, what's going on with each

    return


if __name__ == '__main__':
    main()
