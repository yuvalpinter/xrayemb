"""
Specialized models for downstream task types.
"""
from typing import List, Tuple, Union

import torch
from torch import nn, Tensor

from src.tdt.generation import TdtGenerator


class QaExtractor(nn.Module):
    """
    FUTURE take the selected passage and create an answer based on words from it
    """
    pass


class QaGenerator(nn.Module):
    """
    Generates an answer for a MarcoQA instance.
    NOTE this implementation is partial and slow. Vectorize somehow after initial implementation.
    """

    def __init__(self, args, vec_dim, space_char_idx, eow_index, gen: TdtGenerator):
        super().__init__()

        self.device = args.device
        self.vec_dim = vec_dim
        self.space_char_idx = space_char_idx
        self.gen = gen
        self.eow_vec = torch.tensor([eow_index], dtype=torch.int64, device=self.device)

    def forward(self, in_vecs, targets: Tensor, forcing: bool = True) ->\
            List[List[Tuple[Union[str, Tensor], Tensor]]]:
        batch_insts = []
        for v, t in zip(in_vecs, targets):
            curr_v = v
            trg_bounds = (t == self.space_char_idx).nonzero(as_tuple=True)[0].tolist()
            q_insts = []
            for b, bn in zip([0] + trg_bounds, trg_bounds + [len(t)]):
                w_s = b+1 if b > 0 else 0
                chars_tens = t[w_s:bn]
                trg_word = torch.cat([chars_tens, self.eow_vec]).view(1, -1)
                if forcing:
                    in_word = torch.cat([self.eow_vec, chars_tens]).view(1, -1)
                    preds = self.gen(curr_v, in_word)
                    q_insts.append((preds.permute(0, 2, 1), trg_word))
                else:
                    gen_word = self.gen.generate(curr_v)[0]  # returns indices, probabilities, lengths
                    q_insts.append((gen_word, trg_word))
                # TODO tokenize the output word (or gold if teacher forcing),
                #  add to preceding context,
                #  update `curr_v` accordingly
            batch_insts.append(q_insts)
        return batch_insts

    def config_params(self):
        params = {'space_char_idx': self.space_char_idx}
        return params


class QaScorer(nn.Module):
    """
    Scores a QA sub-instance: a passage as an answer's origin to a query
    """

    def __init__(self, args, vec_dim):
        super().__init__()

        self.device = args.device
        self.hid_size = vec_dim

        self.proj1 = nn.Linear(vec_dim, self.hid_size).to(self.device)
        self.proj2 = nn.Linear(self.hid_size, self.hid_size).to(self.device)
        self.score = nn.Linear(self.hid_size, 1).to(self.device)

    def forward(self, in_vecs):
        return self.score(nn.functional.relu(self.proj2(nn.functional.relu(self.proj1(in_vecs)))))

    def config_params(self):
        params = {'hidden_size': self.hid_size,
                  'num_mlp_layers': 2}
        return params


class EmojiPredictor(nn.Module):
    """
    Predict a class of emoji for a tweet.
    """

    def __init__(self, args, vec_dim, num_classes):
        super().__init__()

        self.device = args.device
        self.hid_size = num_classes
        self.num_mlp_layers = args.task_num_mlp_layers

        if self.num_mlp_layers < 2:
            self.projs = nn.ModuleList([nn.Linear(vec_dim, num_classes).to(self.device)])
            return

        prjs = [nn.Linear(vec_dim, self.hid_size).to(self.device)]
        for _ in range(self.num_mlp_layers - 2):
            prjs.append(nn.Linear(self.hid_size, self.hid_size).to(self.device))
        prjs.append(nn.Linear(self.hid_size, num_classes).to(self.device))
        self.projs = nn.ModuleList(prjs)

    def forward(self, in_vecs):
        for p in self.projs[:-1]:
            in_vecs = nn.functional.relu(p(in_vecs))
        return self.projs[-1](in_vecs)

    def config_params(self):
        params = {'hidden_size': self.hid_size,
                  'num_mlp_layers': self.num_mlp_layers}
        return params


class NerPredictor(nn.Module):
    """
    Tag individual words in an NER dataset.
    """

    def __init__(self, args, vec_dim, num_classes):
        super().__init__()

        self.device = args.device
        self.lstm_hid_size = args.task_lstm_hidden_size
        self.num_lstm_layers = args.task_num_lstm_layers
        self.num_mlp_layers = args.task_num_mlp_layers

        if self.num_lstm_layers > 0:
            self.lstm = nn.LSTM(vec_dim, self.lstm_hid_size, self.num_lstm_layers, bidirectional=True, dropout=0.5)\
                .to(self.device)
        else:
            self.lstm = None
            self.lstm_hid_size = vec_dim // 2

        prjs = [nn.Linear(self.lstm_hid_size * 2, num_classes).to(self.device)]
        for _ in range(self.num_mlp_layers - 1):
            prjs.append(nn.Linear(num_classes, num_classes).to(self.device))
        self.projs = nn.ModuleList(prjs)

    def forward(self, in_vecs):
        if self.lstm:
            p_out, _ = self.lstm(in_vecs)
        else:
            p_out = in_vecs
        for p in self.projs[:-1]:
            p_out = nn.functional.relu(p(p_out))
        return self.projs[-1](p_out)

    def config_params(self):
        params = {'lstm_hidden_size': self.lstm_hid_size,
                  'num_lstm_layers': self.num_lstm_layers,
                  'num_mlp_layers': self.num_mlp_layers}
        return params


