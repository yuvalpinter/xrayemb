"""
Generation module for TDT ("Detok")
"""
from argparse import Namespace
from typing import List, Tuple

import torch
from torch import nn

from src.tdt.consts import BOUND_CHAR, PAD_CHAR, UNK_CHAR


class TdtGenerator(nn.Module):
    def __init__(self, char_vocab: List, emb_size, vec_size, args: Namespace):
        super().__init__()

        self.lstm_hidden_size = args.gen_lstm_hidden_size
        self.num_lstm_layers = args.gen_num_lstm_layers
        self.device = args.device

        self.char_embs = nn.Embedding(len(char_vocab), emb_size).to(self.device)
        self.bound_idx = char_vocab.index(BOUND_CHAR)
        self.non_starts = [i for i, c in enumerate(char_vocab) if c == BOUND_CHAR or c.isspace()]
        if not args.__contains__("spaces_end"):
            args.spaces_end = False
        self.gen_ends = self.non_starts if args.spaces_end else [self.bound_idx]
        self.pad_idx = char_vocab.index(PAD_CHAR)
        self.unk_idx = char_vocab.index(UNK_CHAR)
        self.proj_emb = len(char_vocab) - 1
        assert self.pad_idx >= self.proj_emb  # if not, self.proj output handling needs to change
        self.max_gen = args.max_gen

        self.bound_tens = torch.tensor([self.bound_idx], requires_grad=False).to(self.device)
        self.h_0 = nn.Linear(vec_size, self.lstm_hidden_size * self.num_lstm_layers).to(self.device)
        self.lstm = nn.LSTM(emb_size, self.lstm_hidden_size, self.num_lstm_layers,
                            batch_first=True, bidirectional=False) \
            .to(self.device)
        self.proj1 = nn.Linear(self.lstm_hidden_size, self.proj_emb).to(self.device)
        self.proj2 = nn.Linear(self.proj_emb, self.proj_emb).to(self.device)

    def forward(self, in_vecs: torch.Tensor, gold_ids: torch.Tensor, **kwargs):
        """
        :param in_vecs: batch of (possibly contextualized) embeddings for supervised generation
        :param gold_ids: desired output character IDs
        :return: predicted character probabilities
        """
        if len(in_vecs.shape) == 1:
            in_vecs = in_vecs.view(1, -1)
        batch_size = in_vecs.shape[0]

        h, c = self.init_states(batch_size, in_vecs)
        embs = self.char_embs(gold_ids)
        lstmed, _ = self.lstm(embs, (h, c))
        preds = self.proj2(torch.tanh(self.proj1(lstmed)))
        return preds

    def generate(self, in_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        auto-regressively apply a seq2seq module
        :param in_vec: a (possibly contextualized) embedding for free-form generation
        :return: tuple of (generated characters, probabilities for all characters in each step)
        """
        if len(in_vec.shape) == 1:
            in_vec = in_vec.view(1, -1)
        batch_size = in_vec.shape[0]

        in_char = self.bound_tens.expand(batch_size, -1)
        out_pred = torch.tensor([-1] * batch_size).to(self.device)
        finished_gens = torch.tensor([False] * batch_size).to(self.device)
        probs = []
        gend = []
        lens = [self.max_gen] * batch_size

        h, c = self.init_states(batch_size, in_vec)
        step = 0
        while any([p not in self.gen_ends for p in out_pred]) and len(gend) < self.max_gen:
            in_embs = self.char_embs(in_char)
            o, (h, c) = self.lstm(in_embs, (h, c))

            sm = self.proj2(torch.tanh(self.proj1(o.view(batch_size, -1)))  # get rid of seq_len, always 1
                            .to(self.device))  # TODO parameterize the nonlinearity
            if step == 0:
                # don't predict empty strings, don't start with space
                sm[:, self.non_starts] = float('-inf')
            probs.append(sm.softmax(1))
            out_pred = sm.argmax(1).to(self.device)
            out_pred[finished_gens] = self.pad_idx
            for j in self.gen_ends:
                finished_gens |= (out_pred == j)
            for i in range(batch_size):
                if out_pred[i] in self.gen_ends:
                    assert lens[i] == self.max_gen
                    lens[i] = step

            gend.append(out_pred.clone().detach())
            in_char = out_pred.view(-1, 1)  # seq len again
            step += 1
        return torch.stack(gend, dim=1).to(self.device), torch.stack(probs, dim=1).to(self.device), lens

    def init_states(self, batch_size, in_vec):
        h = self.h_0(in_vec).view(self.num_lstm_layers, batch_size, -1).to(self.device)
        c = torch.zeros(h.shape, requires_grad=False).to(self.device)
        return h, c
