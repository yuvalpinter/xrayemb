"""
Analyze dataset by count statistics - TTR, multi-token words, etc. - vs. a comparably-sized sample from Wikipedia.
"""
import argparse
import gzip
import logging
import random
from collections import Counter

import tensorflow as tf
from tqdm import tqdm
from transformers import GPT2Tokenizer, PreTrainedTokenizer

from src.downstream.dataset import get_dataset, get_nytwit_dataset, nytwit_vocab
from src.tdt.infer import load_lm_for_inference
from src.tdt.nlp_utils import NlpTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def process_multitoks(tok: PreTrainedTokenizer, vocab: Counter):
    word_pref = ' ' if isinstance(tok, GPT2Tokenizer) else ''
    total_toks, total_types, total_pcs, multi_toks, multi_types = 0, 0, 0, 0, 0
    for w, c in vocab.items():
        wps = len(tok.tokenize(word_pref + w))
        total_types += 1
        total_toks += c
        total_pcs += (wps * c)
        if wps > 1:
            multi_types += 1
            multi_toks += c
    mtp_pc = multi_types / total_types
    mtk_pc = multi_toks / total_toks
    mass_inc = (total_pcs / total_toks) - 1.
    logger.info(f'Found {multi_types} multi-types out of {total_types} types ({mtp_pc:.2%}).')
    logger.info(f'Found {multi_toks} multi-tokens out of {total_toks} tokens ({mtk_pc:.2%}).')
    logger.info(f'Total mass increase: {mass_inc:.2%}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['emoji', 'conll', 'nemer', 'ner',
                                              'marcoqa', 'marcosamp', 'nytwit'])
    parser.add_argument('--dataloc', choices=['group', 'local'], default='local')
    parser.add_argument('--nyt-data')
    parser.add_argument('--hashtml', type=bool, default=True)
    parser.add_argument('--wiki-corpus')
    parser.add_argument('--base-model-dir')
    parser.add_argument('--second-base-model-dir', default=None)
    parser.add_argument('--lowercase', action='store_true')
    parser.add_argument('--wiki-sample', type=float, default=0.01)
    parser.add_argument('--data-sample', type=float, default=1.0)
    parser.add_argument('--model-type', choices=['bert', 'cbert', 'roberta', 'gpt2'], default='gpt2')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if args.model_type == 'gpt2':
        _, tok, _ = load_lm_for_inference(args)
    else:
        raise ValueError(f'model type {args.model_type} noot supported yet.')

    if args.dataset == 'nytwit':
        ds = get_nytwit_dataset(args.nyt_data)
        ds_voc = nytwit_vocab(ds)
    else:
        ds = get_dataset(args)
        dstr = ds["train"]
        ds_voc = dstr.vocab()
    ds_total = sum(ds_voc.values())
    logger.info(f'Loaded {ds_total} space-delimited words from task dataset.')
    process_multitoks(tok, ds_voc)

    wiki_vocab = Counter()
    wiki_total = 0
    pretok = NlpTokenizer(hashtml=True)
    with tf.io.gfile.GFile(args.wiki_corpus, mode="rb") as f, gzip.GzipFile(fileobj=f) as zf:
        for line in tqdm(zf, mininterval=120):
            if wiki_total >= ds_total:
                break
            if random.random() > args.wiki_sample:
                continue
            lds = line.decode().strip()
            if lds:
                lds = pretok(lds)
                if lds:
                    words = lds.split()
                    wiki_total += len(words)
                    wiki_vocab.update(words)
    logger.info(f'Loaded {wiki_total} space-delimited words from Wikipedia dump.')
    process_multitoks(tok, wiki_vocab)


if __name__ == "__main__":
    main()
