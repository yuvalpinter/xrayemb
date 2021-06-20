"""
Analyze results from two models, one TDT and one not, based on word vocab-ness.
Run locally.
"""
import argparse
import logging
from collections import Counter

from transformers import RobertaTokenizer

from src.downstream.conlleval import ner_evaluate
from src.tdt.consts import GPT_SPACE, EASY_SUFFIXES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CmpStats:
    def __init__(self, golds, base_preds, tdt_preds):
        self.total_toks = 0
        self.b_corr = 0
        self.t_corr = 0
        self.eq_corrs = 0
        self.eq_wrongs = 0
        logger.info('base model performance:')
        ner_evaluate(golds, base_preds, verbose=True)
        logger.info('tdt model performance:')
        ner_evaluate(golds, tdt_preds, verbose=True)
        for g, b_p, t_p in zip(golds, base_preds, tdt_preds):
            self.total_toks += 1
            if g == b_p: self.b_corr += 1
            if g == t_p: self.t_corr += 1
            if g == b_p and b_p == t_p: self.eq_corrs += 1
            if g != b_p and b_p == t_p: self.eq_wrongs += 1
        logger.info(f'{self.total_toks} tokens, '
                    f'{self.b_corr} correct in base, {self.t_corr} in tdt, '
                    f'{self.eq_corrs} equal correct, {self.eq_wrongs} equal wrong.')


def get_ner_results(f):
    seqs = []
    buff = []
    for l in f:
        l = l.strip()
        if not l:
            seqs.append(buff)
            buff = []
            continue
        word, gold, pred = l.split('\t')
        buff.append((word, gold, pred))
    seqs.append(buff)
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-loc', help="path to directory containing tokenizer files")
    parser.add_argument('--base-results')
    parser.add_argument('--tdt-results')
    parser.add_argument('--tokenizer', choices=['bert', 'cbert', 'roberta', 'gpt2'], default='roberta')
    args = parser.parse_args()

    with open(args.base_results) as br_f:
        b_res = get_ner_results(br_f)

    with open(args.tdt_results) as tdt_f:
        t_res = get_ner_results(tdt_f)

    if args.tokenizer == 'roberta':
        tok = RobertaTokenizer.from_pretrained(args.model_loc)
    else:
        raise ValueError(f'model {args.tokenizer} not supported yet.')

    total_seqs = 0
    all_g = []
    all_b = []
    all_t = []
    sing_g = []
    sing_b = []
    sing_t = []
    mult_g = []
    mult_b = []
    mult_t = []
    alltwo_g = []
    alltwo_b = []
    alltwo_t = []
    suff_g = []
    suff_b = []
    suff_t = []
    missed_suffs = Counter()
    for b_seq, t_seq in zip(b_res, t_res):
        total_seqs += 1
        for i, ((b_w, b_g, b_p), (t_w, t_g, t_p)) in enumerate(zip(b_seq, t_seq)):
            assert b_w == t_w, f"Word mismatch: {b_w} vs. {t_w}"
            w = b_w.replace(GPT_SPACE, ' ')
            wtoks = tok.tokenize(w)
            assert b_g == t_g, f"Annotation mismatch: {b_g} vs. {t_g}"
            g = b_g
            all_g.append(g)
            all_b.append(b_p)
            all_t.append(t_p)
            if len(wtoks) == 1:
                sing_g.append(g)
                sing_b.append(b_p)
                sing_t.append(t_p)
            elif len(wtoks) == 2:
                alltwo_g.append(g)
                alltwo_b.append(b_p)
                alltwo_t.append(t_p)
                sf = wtoks[1]
                if sf in EASY_SUFFIXES:
                    suff_g.append(g)
                    suff_b.append(b_p)
                    suff_t.append(t_p)
                else:
                    missed_suffs[sf] += 1
            else:
                mult_g.append(g)
                mult_b.append(b_p)
                mult_t.append(t_p)
    logger.info(f'{total_seqs} sequences.')
    logger.info(f'\nAll tokens:')
    all_cmp = CmpStats(all_g, all_b, all_t)
    logger.info(f'\nSingle-piece tokens:')
    sing_cmp = CmpStats(sing_g, sing_b, sing_t)
    logger.info(f'\nMulti-piece tokens:')
    mult_cmp = CmpStats(mult_g, mult_b, mult_t)
    logger.info(f'\nTwo-piece tokens:')
    alltwo_cmp = CmpStats(alltwo_g, alltwo_b, alltwo_t)
    logger.info(f'\nSingle-easy-suffix tokens:')
    suff_cmp = CmpStats(suff_g, suff_b, suff_t)

    # print(missed_suffs.most_common(20))


if __name__ == "__main__":
    main()
