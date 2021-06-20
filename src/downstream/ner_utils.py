"""
Auxiliary functions for NER tasks.
"""
import logging
from typing import List, Union

import torch
from transformers import BertTokenizer, PreTrainedTokenizer, GPT2Tokenizer

from src.downstream.conlleval import split_tag
from src.downstream.consts import UNK_LABEL
from src.tdt.consts import IGNORE_INDEX
from src.tdt.utils import word_med, is_single_easy_suffix

logger = logging.getLogger(__name__)

OTHER_ENT_TYPE = 'other'


def norm_chunk(chunk_types: Union[List[str], None]) -> List[str]:
    """
    Handle prediction chunks with conflicting types
    """
    if not chunk_types:
        return []
    if len(set(chunk_types)) == 1 or chunk_types[0] != OTHER_ENT_TYPE:
        wintype = chunk_types[0]
    else:
        wintype = [t for t in chunk_types if t != OTHER_ENT_TYPE][0]
    return [f'B-{wintype}'] + [f'I-{wintype}' for _ in chunk_types[1:]]


def excess_tokens(model_type):
    """
    Hard-coding some more tranformers package quirks
    :param model_type:
    :return: number of tokens in excess for NER tasks
    """
    if model_type in ['bert', 'cbert', 'roberta']:
        return 2
    elif model_type == 'gpt2':
        return 1
    raise NotImplementedError('Define # of excess tokens for model type')


def validate_ner_seq(btok, excess_toks, inps, targets, joins, sents):
    for i, (semb, targ, sent) in enumerate(zip(inps, targets, sents)):
        nonpads = len([e for e in semb if e != btok.pad_token_id])
        targs = len([t for t in targ if t != IGNORE_INDEX])
        assert targs == nonpads - excess_toks, f'\n{i}:\n{semb};\n{targ};\n{joins};\n{sent}'


def fix_tags(tag_list: List[str]) -> List[str]:
    """
    Heuristically fix illegal tag sequences.
    Other possible fixes not implemented here:
        - maybe I-B-I should be B-I-I
        - maybe use probabilities, viterbi, etc.
    :param tag_list: whatever BIO tags came out of the neural classifier
    :return: tag list in cromulent BIO form
    """
    ret_tags = []
    chunk_types = []
    for t in tag_list:
        if t in ['O', UNK_LABEL]:
            ret_tags.extend(norm_chunk(chunk_types))
            chunk_types = []
            ret_tags.append('O')
            continue
        try:
            pref, etype = split_tag(t)
        except ValueError:
            logger.warning('\t'.join(tag_list))
            ret_tags.append('O')
            continue
        if pref == 'B':
            ret_tags.extend(norm_chunk(chunk_types))
            chunk_types = []
        chunk_types.append(etype)
    ret_tags.extend(norm_chunk(chunk_types))
    return ret_tags


class NerHelper:
    """
    Various helper functions that depend on model setup.
    """
    def __init__(self, tokzr: PreTrainedTokenizer,
                 easy_suffs: bool, stoch: bool, is_tdt: bool = False):
        self.easy_suffs = easy_suffs
        self.stoch = stoch
        self.is_tdt = is_tdt
        self.tokzr = tokzr

    def has_cls(self):
        return not isinstance(self.tokzr, GPT2Tokenizer)

    def rejoin_words(self, words: List[str], joins=None):
        """
        :param words: words in original sentence
        :param joins: list-of-lists dictionary of token join indices
        :return: tuple of rejoined words, and indices to be ignored for sequence tagging
        """
        if isinstance(self.tokzr, BertTokenizer):
            return words, None
        elif isinstance(self.tokzr, GPT2Tokenizer):
            joined_ws = []
            ig_idcs = []
            k = 0
            if joins is None:
                joins = [-1] * len(words)
            for i, (w, j) in enumerate(zip(words, joins)):
                if word_med(self.tokzr, w, i):
                    joined_ws[-1] += w
                    if not self.easy_suffs and j == -1:
                        ig_idcs.append(k)
                    else:
                        if self.easy_suffs and is_single_easy_suffix(self.tokzr,
                                                                     words, i, has_cls=False):
                            ig_idcs.append(k)
                        else:
                            k -= 1
                else:
                    joined_ws.append(w)
                k += 1
            if self.is_tdt and not self.easy_suffs and not self.stoch:
                ig_idcs = None
            return joined_ws, ig_idcs
        else:
            logger.warning('Word rejoining not implemented for this tokenizer.')
            return words, None

    def ner_batch_from_preds(self, sents, preds: torch.Tensor, targets: torch.Tensor,
                             lab_list: List[str], joins=None):
        """
        :param sents: list of lists of string tokens in batch
        :param preds: predicted label scores for batch
        :param targets: true label indices for batch
        :param lab_list: list of labels for stringifying indices
        :param joins: token joins performed by tokenizer and vectorization policy
        :return: tuple of true label list, predicted label list, output strings for presentation
        """
        true_l = []
        pred_l = []
        out_strs = []
        if joins is None:
            joins = [None] * len(sents)

        for snt, prds, trgs, jn in zip(sents, preds.argmax(dim=2), targets, joins):
            prds = prds.cpu().tolist()
            trgs = trgs.cpu().tolist()
            words, tags = snt
            words, ignore_idcs = self.rejoin_words(words,
                                                   joins=None if jn is None else jn[1:])
            if self.has_cls():
                prds.pop(0)
                trgs.pop(0)
            if ignore_idcs is not None:
                for m, i in enumerate(ignore_idcs):
                    # m is number of pops already made, so needs to be subtracted each time.
                    prds.pop(i - m)
                    trgs.pop(i - m)

            prds = prds[:len(tags)]
            trg_tags = [lab_list[t] for t in trgs if t >= 0]

            prds = fix_tags([lab_list[p] for p in prds if p >= 0])
            assert tags == trg_tags, f'{len(tags)} tags, {len(trgs)} trgs\n' \
                                     f'tokens: {snt[0]}\nwords:  {words}\n' \
                                     f'joins:  {jn}\nignore: {ignore_idcs}\n' \
                                     f'tags:   {snt[1]}\ntargs:  {trgs}\npreds:  {prds}'
            true_l.extend(tags)
            pred_l.extend(prds)

            out_str = '\n'.join([f'{w}\t{t}\t{p}' for w, t, p in zip(words, tags, prds)])
            out_strs.append(out_str)

        return true_l, pred_l, out_strs

    def align_targets(self, targets, inps, joins):
        max_seq_len = inps.shape[1]
        if not self.stoch:
            return targets[:, :max_seq_len]
        new_targs = []
        for seq, js in zip(targets.tolist(), joins.tolist()):
            new_s = []
            curr_idx = -1
            for t, j in zip(seq, js):
                if j != -1 and j == curr_idx:
                    continue
                curr_idx = j
                new_s.append(t)
            new_targs.append(new_s + ([IGNORE_INDEX] * (max_seq_len - len(new_s))))
        return torch.tensor(new_targs, dtype=targets.dtype, device=targets.device)