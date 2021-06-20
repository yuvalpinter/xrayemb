"""
Generate sequences from a Detok model by various means.
"""
from argparse import Namespace
from typing import List, Iterable

import torch

from src.tdt.tokdetok import TdtWrapper


def gen_rand(tdtmod: TdtWrapper,
             all_chars: str,
             args: Namespace, samples=10) -> Iterable[str]:
    """
    Generate randomly from vector space
    :param tdtmod: main TDT object, wrapping a generator inter alia
    :param all_chars: mapping of all characters available to generator
    :param args: user-specified argument dictionary (Namespace)
    :param samples: number of samples to return
    :return: iterable sequence of greedily generated strings
    """
    hid_size = tdtmod.bmodel.get_output_embeddings().in_features
    gen = tdtmod.tdtgen
    with torch.no_grad():
        for i in range(samples):
            vec = torch.randn(1, hid_size).to(args.device)
            yield ''.join([all_chars[i] for i in gen.generate(vec)[0].tolist()[0]])


def gen_from_masks(tdtmod: TdtWrapper,
                   all_chars: str,
                   sent: str,
                   mask_ids: List[int]) -> Iterable[str]:
    """
    Predict strings from contextualized vectors
    :param tdtmod: main TDT object, wrapping a generator inter alia
    :param all_chars: mapping of all characters available to generator
    :param sent: sentence to be used as contexts for generation. Use [MASK] in desired mask location
    :param mask_ids: which token IDs in the sentence are masked
    :return: sequence of greedily generated strings for all masked locations
    """
    inp_ids = tdtmod.btok.encode_plus(sent)['input_ids']
    inp_tns = torch.tensor(inp_ids)
    with torch.no_grad():
        outp_vecs = tdtmod.tdtemb(inp_tns)[1][0]
        for mask_id in mask_ids:
            mask_vec = outp_vecs[mask_id].view(1, -1)
            gend = tdtmod.tdtgen.generate(mask_vec)
            yield ''.join([all_chars[i] for i in gend[0].tolist()[0]])


def cycle_check(tdtmod: TdtWrapper, all_chars: str, seq):
    """
    Generate outputs of a T->D cycle.
    :param tdtmod: main TDT object, wrapping a generator inter alia
    :param all_chars: mapping of all characters available to generator
    :param seq: input sequence, ideally also the target
    :return: tuple containing:
        cyclic - most likely character for each location knowing what the true sequence so far should have been
        manual - most likely sequence from the vector embedded by Tok at the end of `seq`
    """
    td_out = tdtmod(None, other_inp=[seq], action='td_cycle')[0].argmax(1)
    cyclic = ''.join([all_chars[i] for i in td_out.tolist()])

    tout = tdtmod.tdtemb.vectorizer(tdtmod.char_vec(seq))
    dout = tdtmod.tdtgen.generate(tout)
    manual = ''.join([all_chars[i] for i in dout[0][0].tolist()])

    return cyclic, manual
