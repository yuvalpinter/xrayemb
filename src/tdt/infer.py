"""
Loading utilities for performing predictions using a base LM or a TDT module.
"""

from src.tdt.tokdetok import load_wrapper
from .aux_classes import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.NOTSET)


def load_tdt_for_inference(args):
    tdtmod, tmp_dir, all_chars = load_wrapper(args.model_type, args.base_model_dir, args.tdt_model_dir,
                                              args.vectorizer_type, args.checkpoint, device=args.device,
                                              for_lm=args.dataset is not None, hashtml=args.hashtml,
                                              infer_policy=args.infer_policy, pool_policy=args.pool_policy,
                                              stoch_rate=args.stoch_rate)
    tdtmod.eval()
    return tdtmod, tmp_dir, all_chars


def load_lm_for_inference(args):
    second_dir = args.second_base_model_dir if args.second_base_model_dir is not None else None
    basemod, btok, tmp_dir = load_wrapper(args.model_type, args.base_model_dir, second_dir,
                                          None, None, device=args.device,
                                          for_lm=False, hashtml=args.hashtml)
    basemod.eval()
    return basemod, btok, tmp_dir


def call_bmod(bmod, in_idcs):
    """
    :param bmod: a base model (no TDT)
    :param in_idcs: input indices
    :return: a tuple containing the base model's outputs and the input indices passed
    """
    if in_idcs.shape[1] > bmod.config.max_position_embeddings - 2:
        logger.info(f'Aborting batch with {len(in_idcs)} sequences of max length {in_idcs.shape[1]}.')
        return None, in_idcs
    mlm_out = bmod(input_ids=in_idcs)
    return mlm_out, in_idcs

