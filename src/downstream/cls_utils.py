"""
Auxiliary functions for classification tasks
"""
from typing import List

from sklearn.metrics import f1_score
from torch import Tensor
from transformers import PreTrainedTokenizer

from src.tdt.utils import id_to_token


def orig_str(tok: PreTrainedTokenizer, inp: List[int]) -> str:
    """
    :param tok: tokenizer
    :param inp: list of token indices
    :return: original string producing this tokenization
    """
    return ' '.join([id_to_token(tok, i, clean=False) for i in inp if i != tok.pad_token_id])


def emoji_batch_from_preds(sents: List[str],
                           preds: Tensor,
                           targets: Tensor,
                           inputs: Tensor = None,
                           tok: PreTrainedTokenizer = None):
    """
    Create reportable batch from emoji prediction
    :param sents: sentences in batch
    :param preds: tensor containing predicted label scores
    :param targets: tensor containing true labels
    :param inputs: input token IDs for batch
    :param tok: tokenizer
    :return: tuple containing:
        true_l - a list of targets (true labels)
        pred_l - a list of predicted labels
        out_strs - a list of strings from the original instances
    """
    true_l = targets.tolist()
    pred_l = preds.argmax(dim=1).tolist()
    assert len(true_l) == len(pred_l), f"Batch has {len(true_l)} targets but {len(pred_l)} predictions."
    if inputs is None:
        out_strs = [f'{snt[0]}\t{trg}\t{prd}\n' for snt, prd, trg in zip(sents, pred_l, true_l)]
    else:
        out_strs = [f'{snt[0]}\t{orig_str(tok, inp)}\t{trg}\t{prd}\n'
                    for snt, inp, prd, trg
                    in zip(sents, inputs.cpu().numpy().tolist(), pred_l, true_l)]
    return true_l, pred_l, out_strs


def cls_evaluate(trues, preds):
    """
    Wrapper for classification evaluation
    :param trues: true labels for batch
    :param preds: predicted labels for batch
    :return: tuple containing scores: accuracy, micro-F1, macro-F1
    """
    acc = len([p for p, g in zip(trues, preds) if p == g]) / len(trues)
    mic_f1 = f1_score(trues, preds, average='micro')
    mac_f1 = f1_score(trues, preds, average='macro')
    return acc, mic_f1, mac_f1
