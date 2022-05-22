# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Classes in this file are adapted from
https://github.com/huggingface/transformers/blob/v2.8.0/examples/run_language_modeling.py
"""
import gzip
import logging
import os
import random
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BertTokenizer, RobertaTokenizer

from src.tdt.consts import SPECIAL_CHAR_LIST

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Dataset for text treated contiguously
    """

    def __init__(self, pretokenizer, tokenizer: PreTrainedTokenizer, file_path: str,
                 block_size=512, portion=1.0, column=-1):
        """
        :param pretokenizer: PTB or NLP tokenizer, if necessary
        :param tokenizer: LM tokenizer
        :param file_path: Remote or local path of text file
        :param block_size: Size of pretraining sequence, special tokens CLS, SEP included

        No shuffling!
        """
        if column >= 0:
            raise ValueError(f'Cannot accept column param {column}.')

        self.device = None
        self.tok = tokenizer
        # assert not tf.io.gfile.isdir(file_path)

        # [ro]bert[a] tokenizers have two special tokens for single sentence input
        block_size = block_size - 2

        directory, filename = os.path.split(file_path)
        logger.info("Creating features from free text dataset file at %s", directory)
        self.examples = []
        if file_path.endswith('.gz'):
            with open(file_path, mode="rb") as f, gzip.GzipFile(fileobj=f) as zf:
                text = zf.read().decode()
        else:
            with open(file_path, mode="r") as f:
                text = f.read()

        self.chars = sorted(list(set(text))) + SPECIAL_CHAR_LIST

        tokenized_text = self.tok.convert_tokens_to_ids(self.tok.tokenize(pretokenizer(text)))
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            if random.random() > portion:
                continue
            self.examples.append(self.tok.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
        # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
        # If your dataset is small, first you should look for a bigger one :-) and second you
        # can change this behavior by adding (model specific) padding.

    def set_device(self, device):
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if type(self.tok) not in [BertTokenizer, RobertaTokenizer]:
            return torch.tensor([self.tok.bos_token_id] + self.examples[item] + [self.tok.eos_token_id],
                                dtype=torch.long, device=self.device)
        return torch.tensor(self.examples[item], dtype=torch.long, device=self.device)


# noinspection PyTypeChecker
class LineByLineTextDataset(Dataset):
    """
    Dataset for text files, in which sequences are separated as lines.
    """

    def __init__(self, pretokenizer, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512, portion=1.0,
                 shuffle=True, column=-1):
        """
        :param pretokenizer: PTB tokenizer, if necessary
        :param tokenizer: LM Tokenizer
        :param file_path: Remote or local path of text file
        :param block_size: Max size of pretraining sequences, special tokens CLS, SEP included. Longer
            sequences are truncated, shorter ones padded.
        """
        self.device = None
        self.tok = tokenizer
        # assert not tf.io.gfile.isdir(file_path)
        logger.info("Creating features from line-by-line dataset file at %s", file_path)
        logger.info(f"Pretokenizer type is {type(self.tok)}")

        lines = []
        if file_path.endswith('.gz'):
            with open(file_path, mode="rb") as f, gzip.GzipFile(fileobj=f) as zf:
                for line in tqdm(zf, mininterval=120):
                    if random.random() > portion:
                        continue
                    lds = line.decode().strip()
                    if lds:
                        if column >= 0 and '\t' in lds:
                            lds = lds.split('\t')[column]
                        lds = pretokenizer(lds)
                        if lds:
                            lines.append(lds)
        else:
            with open(file_path, mode="r") as f:
                for line in tqdm(f, mininterval=120):
                    if random.random() > portion:
                        continue
                    lds = line.strip()
                    if lds:
                        if column >= 0 and '\t' in lds:
                            lds = lds.split('\t')[column]
                        lds = pretokenizer(lds)
                        if lds:
                            lines.append(lds)

        if shuffle:
            random.shuffle(lines)

        if type(self.tok) not in [BertTokenizer, RobertaTokenizer]:
            block_size = block_size - 2

        self.chars = sorted(list(set(''.join(lines)))) + SPECIAL_CHAR_LIST
        self.examples = self.tok.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def set_device(self, device):
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if type(self.tok) not in [BertTokenizer, RobertaTokenizer]:
            return torch.tensor([self.tok.bos_token_id] + self.examples[i] + [self.tok.eos_token_id],
                                dtype=torch.long, device=self.device)
        return torch.tensor(self.examples[i], dtype=torch.long, device=self.device)


def load_dataset(file_path, pretokenizer, tokenizer: PreTrainedTokenizer, line_by_line=False, block_size=512,
                 portion=1.0, shuffle=True, column=-1):
    if line_by_line:
        return LineByLineTextDataset(pretokenizer, tokenizer, file_path=file_path, block_size=block_size,
                                     portion=portion, shuffle=shuffle, column=column)
    else:
        if shuffle:
            logger.warning("Data will not be shuffled outside line-by-line mode.")
        return TextDataset(pretokenizer, tokenizer, file_path=file_path, block_size=block_size,
                           portion=portion, column=column)


def load_vocab(file_path: str, vocab_size: int = -1, lowercase: bool = False) -> List[Tuple[str, int]]:
    if file_path is None:
        logger.warning(f'  No vocab file supplied - no cycle dependency will be trained.')
        return []
    voc = Counter()
    reads = 0
    tot = 0
    logger.info(f'  Reading vocab from file {file_path} with lowercasing set to {lowercase}.')
    assert not open(file_path)
    if file_path.endswith('.gz'):
        with open(file_path, mode="rb") as f, gzip.GzipFile(fileobj=f) as zf:
            rawlines = [ln.decode() for ln in zf.readlines()]
    else:
        with open(file_path, mode="r") as f:
            rawlines = f.readlines()
    for line in rawlines:
        try:
            w, c = line.strip().split('\t')
        except ValueError:
            logger.info(f'Not 2 parts in line {line}')
            continue
        if lowercase:
            w = w.lower()
        count = int(c)
        voc[w] += count
        tot += count
        reads += 1
    if vocab_size > 0:
        voc = voc.most_common(vocab_size)
    else:
        voc = voc.most_common()
    logger.info(f'  Loaded {reads} words trimmed to {len(voc)} with total of {tot} counts.')
    return voc
