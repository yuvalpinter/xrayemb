from typing import List, Tuple

import numpy as np
import torch
from tqdm import trange

from src.tdt.tokdetok import TdtWrapper
from src.tdt.aux_classes import logger


class CycleTrainer:
    """
    Generalizes the cycle dependency loops
    """

    def __init__(self, args, **kwargs):
        self.device = args.device
        self.action = ''
        self.alpha = 1.0

    def __call__(self, args, tdt_wrapper: TdtWrapper, loss_fn, optimizer=None, scheduler=None):
        logger.info(f"{self.__class__} starting cycle dependency loop.")
        batch_size = args.per_gpu_train_batch_size
        tdt_wrapper.train()
        tdt_wrapper.zero_grad()
        losses = []
        for _ in trange(args.cycle_batch_iters,
                        desc="Iteration",
                        disable=args.local_rank not in [-1, 0],
                        mininterval=60):
            batch = self.sample_batch(batch_size)
            pred, gold = tdt_wrapper(in_ids=None, other_inp=batch, action=self.action)
            loss = loss_fn(pred, gold).to(self.device) * self.alpha
            loss.backward()
            optimizer.step()
            scheduler.step()

            tdt_wrapper.zero_grad()
            losses.append(float(loss))
        return sum(losses)

    def sample_batch(self, batch_size):
        raise NotImplementedError


class TdCycleTrainer(CycleTrainer):
    """
    Vectorize a word, detokenize the vector to try and reach the same word
    """

    def __init__(self, args, vocab: List[Tuple[str, int]]):
        super(TdCycleTrainer, self).__init__(args)

        self.action = 'td_cycle'
        self.alpha = args.alpha_cyc_td

        self.strategy = args.td_strategy
        self.vocab_words, freqs = list(zip(*vocab))
        self.freqs = self.adjust_freqs(freqs)

    def adjust_freqs(self, freqs):
        if self.strategy == 'uniform':
            freqs = np.ones_like(freqs)
        elif self.strategy == 'sqrt':
            freqs = np.sqrt(freqs)
        else:
            assert self.strategy == 'freq', f'Unknown strategy param {self.strategy}'
            freqs = np.array(freqs)
        freqs = freqs / freqs.sum()
        return freqs

    def sample_batch(self, batch_size: int):
        sampled = np.random.choice(self.vocab_words, batch_size, p=self.freqs)
        return sorted(sampled, key=lambda x: -len(x))


class DtCycleTrainer(CycleTrainer):
    """
    Detokenize a vector, vectorize resulting sequence to try and reach the same vector
    """

    def __init__(self, args):
        super(DtCycleTrainer, self).__init__(args)

        self.action = 'dt_cycle'
        self.alpha = args.alpha_cyc_dt

        self.vec_dim = args.word_emb_dim

    def sample_batch(self, batch_size):
        return (torch.randn(batch_size, self.vec_dim) / np.sqrt(self.vec_dim)).to(self.device)