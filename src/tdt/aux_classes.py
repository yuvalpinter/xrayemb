"""
Auxiliary classes for core TDT elements: Filters, Poolers, Maskers.
"""
import logging
import random
from argparse import Namespace
from typing import Callable

import torch
from transformers import PreTrainedTokenizer

from src.tdt.consts import EASY_SUFFIXES, IGNORE_INDEX
from src.tdt.utils import id_to_token, word_med

logger = logging.getLogger(__name__)


class FilterRules(object):
    """
    Decides which embedded tokens are to be replaced with a vectorized output.
    May be used for 2pt phase (learn vectors for single-token words) and fine-tuning / inference flavor
    """

    def __init__(self):
        super().__init__()

    def __call__(self, in_ids, *args, **kwargs) -> torch.Tensor:
        """
        :param in_ids: a list of token ids from a Tokenizer
        :return: a torch tensor corresponding to inputs, marking unreplaceables as -1
        and replaceables with successive nonnegative indices, each index to be replaced by a single vector.
        """
        filt = torch.full(in_ids.shape, -1, dtype=torch.int)
        # there's gotta be a better way that this, maybe torch.where() somehow
        # TODO consider parallelizing or batching
        next_vec_id = -1
        for i, s in enumerate(in_ids.tolist()):
            curr_t = -1
            for j, t in self.apply_filter(s):  # main entry point for implementing classes
                if t == curr_t:
                    filt[i, j] = next_vec_id
                else:
                    next_vec_id += 1
                    filt[i, j] = next_vec_id
                    curr_t = t
        return filt

    def apply_filter(self, s):
        return []


class StochasticFilter(FilterRules):
    """
    Filters based on a probability. Maintains whole-words (selects iff their first token is selected)
    """

    def __init__(self, base_tok: PreTrainedTokenizer, prob):
        """
        :param base_tok: tokenizer for an LLM
        :param prob: proportion of words to be replaced
        """
        super().__init__()
        self.btok = base_tok
        self.prob = prob

    def apply_filter(self, s):
        idx = -1
        active_filter = False
        for j, t in enumerate(s):
            tok = id_to_token(self.btok, t)
            if tok in self.btok.special_tokens_map.values() \
                    or tok in self.btok.additional_special_tokens:
                active_filter = False
                continue
            if active_filter:
                if word_med(self.btok, tok):
                    yield j, idx
                    continue
                else:
                    active_filter = False
            u = random.uniform(0, 1)
            if u < self.prob:
                if word_med(self.btok, tok, j - 1):  # this word's beginning wasn't selected
                    assert not active_filter
                    continue
                idx += 1
                active_filter = True
                yield j, idx


class NoEasySuffsFilter(FilterRules):
    """
    Selects all tokens that are part of multipiece words (and groups them together),
    excluding words of the form [piece_1 suffix], where suffix is in a closed list.
    """

    def __init__(self, base_tok):
        super().__init__()
        self.btok = base_tok

    def apply_filter(self, s):
        idx = -1
        jump = False
        toks = [id_to_token(self.btok, t) for t in s]
        for j, (tok, next_tok, next_next_tok) in enumerate(zip(toks,
                                                               toks[1:] + [self.btok.pad_token],
                                                               toks[2:] + [self.btok.pad_token] * 2)):
            if jump:
                jump = False
                continue
            if word_med(self.btok, tok, j - 1):
                yield j, idx
            elif (word_med(self.btok, next_tok)
                  and tok not in self.btok.special_tokens_map.values()
                  and tok not in self.btok.additional_special_tokens):
                if (word_med(self.btok, next_next_tok)
                        or next_tok not in EASY_SUFFIXES):
                    idx += 1
                    yield j, idx
                else:
                    jump = True


class AllMultiToksFilter(FilterRules):
    """
    Selects all tokens that are part of multipiece words (and groups them together).
    """

    def __init__(self, base_tok):
        super().__init__()
        self.btok = base_tok

    def apply_filter(self, s):
        idx = -1
        for j, (t, tn) in enumerate(zip(s, s[1:])):
            tok = id_to_token(self.btok, t)
            next_tok = id_to_token(self.btok, tn)
            if word_med(self.btok, tok, j - 1):
                yield j, idx
            elif (word_med(self.btok, next_tok)
                  and tok not in self.btok.special_tokens_map.values()
                  and tok not in self.btok.additional_special_tokens):
                idx += 1
                yield j, idx

        # GPT2 uses no <EOS> / SEP token
        j = len(s) - 1
        if s[j] not in self.btok.all_special_ids:
            tok = id_to_token(self.btok, s[j])
            if word_med(self.btok, tok, j - 1):
                yield j, idx


class AllSingleToksFilter(FilterRules):
    """
    Selects all tokens that are part of single-piece words.
    """

    def __init__(self, base_tok):
        super().__init__()
        self.btok = base_tok

    def apply_filter(self, s):
        idx = 0
        for j, (t, tn) in enumerate(zip(s, s[1:])):
            tok = id_to_token(self.btok, t)
            next_tok = id_to_token(self.btok, tn)
            if not word_med(self.btok, tok, j - 1) \
                    and not word_med(self.btok, next_tok) \
                    and tok not in self.btok.special_tokens_map.values() \
                    and tok not in self.btok.additional_special_tokens:
                yield j, idx
                idx += 1

        # GPT2 uses no <EOS> / SEP token
        j = len(s) - 1
        if s[j] not in self.btok.all_special_ids:
            tok = id_to_token(self.btok, s[j])
            if not word_med(self.btok, tok, j - 1):
                yield j, idx


class Pooler(object):
    """
    A parent class for pooling representations of multiple wordpieces into a single vector,
    typically to act as a target for a vectorized form or to produce a basis for multitoken-word classification
    """

    def __init__(self):
        super().__init__()

    def __call__(self, reps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AvgPooler(Pooler):
    def __init__(self):
        super().__init__()

    def __call__(self, reps: torch.Tensor):
        return reps.mean(dim=0)


class MaxPooler(Pooler):
    def __init__(self):
        super().__init__()

    def __call__(self, reps: torch.Tensor):
        return torch.max(reps, dim=0).values


class FirstTokPooler(Pooler):
    def __init__(self):
        super().__init__()

    def __call__(self, reps: torch.Tensor):
        return reps[0]


class LMMasker(Callable):
    """
    Parent class for masking tokens in an language modeling objective, when vectorized inputs are in the mix.
    """

    def __init__(self, tok: PreTrainedTokenizer, args: Namespace):
        super(LMMasker, self).__init__()

        self.device = args.device
        self.tok = tok

    def __call__(self, inputs: torch.Tensor):
        """
        :param inputs: tensor of token IDs, may or may not contain the special vectorized form marker
        :return: a tuple consisting of the following:
            `inputs`: for use as "input_ids" to LM
            `labels`: for use as "labels" (prediction targets) input to LM
            `pad_mask`: for use as "padding_mask" input to LM
        """
        raise NotImplementedError


class AutoRegressiveMasker(LMMasker):
    def __init__(self, tok: PreTrainedTokenizer, args: Namespace):
        super(AutoRegressiveMasker, self).__init__(tok, args)

        self.device = args.device
        self.tok = tok

    def __call__(self, inputs: torch.Tensor):
        labels = inputs.clone().to(self.device)
        if self.tok.pad_token is not None:
            # Ignore vectorized words (no way to predict) and pads (obviously)
            # NOTE: we're leaving EOS in (because we want to predict end of sentence)
            # and BOS too (because it's never actually predicted)
            padding_bool_mask = labels.eq(self.tok.pad_token_id).to(self.device)
            vectorized_bool_mask = labels.eq(self.tok.additional_special_tokens_ids[0]).to(self.device)

            pad_mask = torch.ones_like(padding_bool_mask, dtype=torch.int64).to(self.device)
            pad_mask.masked_fill_(padding_bool_mask, value=0)

            labels[padding_bool_mask] = IGNORE_INDEX  # No loss for pads
            labels[vectorized_bool_mask] = IGNORE_INDEX  # No LM loss for [VECTORIZED]
        else:
            pad_mask = None

        return inputs, labels, pad_mask


class BertTokenMasker(LMMasker):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    def __init__(self, tok: PreTrainedTokenizer, args: Namespace):
        super(BertTokenMasker, self).__init__(tok, args)

        self.mlm_prob = args.mlm_probability
        self.device = args.device
        self.tok = tok

    def __call__(self, inputs: torch.Tensor):

        if self.tok.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling."
                "Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone().to(self.device)
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.15 in BERT/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_prob).to(self.device)
        special_tokens_mask = [
            self.tok.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(self.device), value=0.0)
        if self.tok.pad_token is not None:
            padding_bool_mask = labels.eq(self.tok.pad_token_id).to(self.device)
            probability_matrix.masked_fill_(padding_bool_mask, value=0.0)
            pad_mask = torch.ones_like(padding_bool_mask, dtype=torch.int64).to(self.device)
            pad_mask.masked_fill_(padding_bool_mask, value=0)
        else:
            pad_mask = None
        masked_indices = torch.bernoulli(probability_matrix).bool().to(self.device)
        labels[~masked_indices] = IGNORE_INDEX  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(self.device) & masked_indices
        inputs[indices_replaced] = self.tok.convert_tokens_to_ids(self.tok.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(self.device) \
                         & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tok), labels.shape, dtype=torch.long).to(self.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, pad_mask
