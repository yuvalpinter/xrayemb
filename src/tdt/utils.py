"""
Mostly model-specific hacks to deal with `transformers`'s idiosyncratic style
as well as some functions abstracting TDT-specific lookups
"""
import glob
import html
import logging
import os
import random
import re
import shutil
from typing import List, Union

import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, PreTrainedTokenizer, GPT2Tokenizer

from src.downstream.consts import PAD_TOKEN
from src.tdt.consts import VEC_TOKEN, GPT_SPACE, BERT_WORDMID, EASY_SUFFIXES

logger = logging.getLogger(__name__)


class PreTokenizer:
    """
    Lazy default; implement your own.
    """

    def __init__(self, hashtml=False):
        """
        :param hashtml: whether text is expected or not to contain escaped HTML.
        """
        self.hashtml = hashtml

    def __call__(self, s):
        if self.hashtml:
            s = html.unescape(s)
        return s


def rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def set_seed(args):
    if args.seed == -1:
        return
    seed = np.abs(args.seed * (args.local_rank + 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def id_to_token(tokzer: PreTrainedTokenizer, idx: int, clean: bool = False) -> str:
    """
    Finds string token for given ID, normalized across tokenizers.
    :param tokzer: tokenizer
    :param idx: numeric token id
    :param clean: remove characters signifying relation of token with word boundary
    :return: token surface form
    """
    if isinstance(tokzer, BertTokenizer):
        # TODO why not just use convert_ids_to_tokens() like in GPT2?
        t = tokzer.ids_to_tokens.get(idx, VEC_TOKEN)
        if clean:
            return t.replace(BERT_WORDMID, '')
        return t
    elif isinstance(tokzer, GPT2Tokenizer):  # includes RoBERTa
        t = tokzer.convert_ids_to_tokens([idx])[0]
        if clean:
            return t.replace(GPT_SPACE, '')
        return t
    else:
        raise NotImplementedError('Implemented only for [Ro]BERT[a] and GPT2')


def word_med(tokzer: PreTrainedTokenizer, tok: Union[str, int, np.number], idx_in_seq=-1) -> bool:
    if idx_in_seq == 0:  # GPT2 doesn't prepend the Ġ at the beginning of a sequence
        return False
    if isinstance(tok, np.number):
        tok = tok.item()
    if isinstance(tok, int):
        tok = id_to_token(tokzer, tok, clean=False)
    if tok in tokzer.all_special_tokens:
        return False  # policy
    if isinstance(tokzer, BertTokenizer):
        return tok.startswith(BERT_WORDMID)
    elif isinstance(tokzer, GPT2Tokenizer):  # includes RoBERTa
        return not tok.startswith(GPT_SPACE)
    # TODO check for T5
    raise NotImplementedError('Implemented only for [Ro]BERT[a] and GPT2')


def add_pad_token(btok: PreTrainedTokenizer, ptok=PAD_TOKEN):
    """
    :param btok: tokenizer
    :param ptok: padding token symbol
    """
    btok.add_special_tokens({'pad_token': ptok})
    logger.info(f'Tokenizer updated with {btok.pad_token}:{btok.pad_token_id}.')


def add_vector_token(btok: PreTrainedTokenizer, vtok=VEC_TOKEN):
    """
    :param btok: tokenizer
    :param vtok: special token symbol
    """
    btok.add_special_tokens({'additional_special_tokens': [vtok]})
    assert len(btok.additional_special_tokens_ids) == 1
    spec_ids = btok.encode(vtok, add_special_tokens=False)
    assert len(spec_ids) == 1
    logger.info(f'Tokenizer updated with {vtok}:{spec_ids[0]}.')


def make_tformer_config(args, vocab_size: int, is_decoder=False):
    cfg = BertConfig()
    cfg.is_decoder = is_decoder
    cfg.hidden_size = args.tform_hidden_size
    cfg.max_position_embeddings = args.max_position_embeddings
    cfg.num_attention_heads = args.tform_heads
    cfg.num_hidden_layers = args.num_tform_layers
    cfg.pad_token_id = vocab_size - 1
    cfg.vocab_size = vocab_size
    cfg.type_vocab_size = 1  # no [SEP]s
    return cfg


def is_single_easy_suffix(tokzr: PreTrainedTokenizer, idcj: Union[List[int], List[str]], k: int,
                          has_cls=True):
    """
    Is this a word with only two tokens, the second of which is a simple inflectional suffix?
    :param tokzr: tokenizer
    :param idcj: list of tokens
    :param k: location of first token in originating sentence
    :param has_cls: does this model have a CLS token
    :return: answer to the question above
    """
    iis_in = -1 if has_cls else 0
    t = idcj[k]
    assert word_med(tokzr, t, idx_in_seq=k + iis_in), f'item {k} with iis index {k + iis_in} in:\n{idcj}'
    if type(t) == int:
        t = id_to_token(tokzr, t)
    return (t in EASY_SUFFIXES
            and (not word_med(tokzr, idcj[k - 1], idx_in_seq=k - 1 + iis_in))
            and (len(idcj) <= k + 1
                 or not word_med(tokzr, idcj[k + 1], idx_in_seq=k + 1 + iis_in)))


def is_masked_model(mtype):
    return mtype in ['bert', 'cbert', 'roberta']
