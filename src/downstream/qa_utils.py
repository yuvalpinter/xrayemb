"""
Helper consts and methods for QA datasets
"""
import torch
from torch import Tensor

QA_REPORT_HEADER = 'total\ttrue i\tpredicted i\trank of true\ttrue passage\tpredicted passage'
QAGEN_HEADER = 'context+query\ttarget\tprediction'


def qa_batch_from_preds(sents: str, preds: Tensor, targets: Tensor):
    """
    A QA batch is a single instance, with all passages.
    :param sents: raw sentences, which are
    :param preds:
    :param targets:
    :return: tuple containing a singleton list for each item, to be processed by aggregator:
        gold true passage index, predicted index, rank of true, passage count (list length), reporting string
    """
    passages, doc = sents
    count = len(passages)
    preds = preds.view(-1)
    assert count == len(preds)
    if len(targets.shape) > 1:
        # assumes one instance (one query, many passages) per batch
        # if multiple instances in batch, use .argmax(dim=1) instead
        targets = targets[0]
    true_i = targets.argmax().item()
    true_s = passages[true_i]
    pred_sort = preds.argsort(descending=True).tolist()
    pred_i = pred_sort[0]
    pred_top_s = passages[pred_i]
    rank = pred_sort.index(true_i)
    out_str = f'{count}\t{true_i}\t{pred_i}\t{rank}\t{true_s}\t{pred_top_s}'
    return [true_i], [pred_i], [rank], [count], [out_str]


def reorg_for_mrg(preds: Tensor, targets: Tensor, device: torch.device):
    """
    Margin-loss adapter for QA datasets.
    :param preds: prediction scores for passages
    :param targets: correct passage indices
    :param device: torch device
    :return: tuple containing correct instance, all false instances, margin-loss targets
    """
    tpi = targets.argmax()
    num_insts = len(preds) - 1
    true_ex = preds[tpi]
    return (true_ex.repeat(num_insts).view(1, -1),
            torch.cat((preds[:tpi], preds[tpi + 1:])).view(1, -1),
            torch.ones(1, num_insts, dtype=torch.float32, device=device))
