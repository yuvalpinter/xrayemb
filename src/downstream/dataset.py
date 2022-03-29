"""
Helper functions and classes for datasets in downstream tasks
"""
import codecs
import gzip
import json
import logging
import os
import random
import typing
import uuid
from collections import Counter
from typing import NamedTuple

import pandas as pd
#import tensorflow as tf
from tqdm import tqdm

from src.downstream.consts import *
#from src.downstream.io_interface import get_nemer_dataloc, get_ner_dataloc, \
#    get_marcosamp_loc, get_conll_loc, get_marco_loc, get_emoji_loc

logger = logging.getLogger(__name__)


# individual task type instances

class MarcoInstanceText(NamedTuple):
    key: str
    query: str
    query_type: str  # options: LOCATION, NUMERIC, PERSON, DESCRIPTION, ENTITY
    wf_answer: str
    answer: str
    true_passage: str
    false_passages: list

    def false_passages_str(self):
        return ' '.join(self.false_passages)


class NerDataset:
    def __init__(self, sents):
        self.sents = sents
        self.tok_count = sum([len(s) for s in self.sents])
        self.labels = self.label_dist()

    def __iter__(self):
        return iter(self.sents)

    def __len__(self):
        return len(self.sents)

    def label_dist(self):
        labs = Counter()
        for s in self.sents:
            labs.update(s[1])
        return labs

    def vocab(self, lowercase=False, only_alpha=False):
        voc = Counter()
        for s in self.sents:
            toks = s[0]
            if only_alpha:
                toks = [t for t in toks if t.isalpha()]
            if lowercase:
                toks = [t.lower() for t in toks]
            voc.update(toks)
        return voc


class EmojiDataset:
    def __init__(self, lab_tweets):
        tweets, labs = lab_tweets
        self.tweets = tweets
        self.labs = [int(lbl) for lbl in labs]
        self.labels = self.label_dist()

    def __iter__(self):
        return zip(self.tweets, self.labs)

    def __len__(self):
        return len(self.tweets)

    def label_dist(self):
        return Counter(self.labs)

    def vocab(self, lowercase=False, only_alpha=False):
        voc = Counter()
        for tw in self.tweets:
            # TODO tokenizer
            toks = tw.split()
            if only_alpha:
                toks = [t for t in toks if t.isalpha()]
            if lowercase:
                toks = [t.lower() for t in toks]
            voc.update(toks)
        return voc


class QaDataset:
    def __init__(self, insts):
        self.instances = insts
        self.labels = self.qtype_dist()

    def __iter__(self):
        return iter(self.instances)

    def __len__(self):
        return len(self.instances)

    def qtype_dist(self) -> typing.Counter[str]:
        return Counter([i.query_type for i in self.instances])

    def vocab(self, only_qa=False):
        qa_voc = Counter()
        all_voc = Counter()
        for i in self.instances:
            toks = i.query.split()
            toks.extend(i.wf_answer.split())
            toks.extend(i.answer.split())
            qa_voc.update(toks)

            if not only_qa:
                toks.extend(i.true_passage.split())
                toks.extend(i.false_passages_str().split())
                all_voc.update(toks)
        if only_qa:
            return qa_voc
        return all_voc


# individual dataset creators

def norm_marco_text(txt, lowercase: bool = False):
    if txt in NULL_ANSWERS:
        return ''
    if lowercase:
        return txt.lower()
    return txt


def load_marco(filename, sample=1.0, is_test: bool = False, lowercase: bool = False) -> QaDataset:
    reader = codecs.getreader("utf-8")
    with gzip.open(filename, 'rb') as in_f:
        data = json.load(reader(in_f))
    insts = []
    no_true_ps = 0
    multi_true_ps = 0
    no_wfa = 0
    no_ans = 0
    for k, q in tqdm(data['query'].items(), mininterval=20):
        if random.random() > sample:
            continue
        if not is_test:
            wfa = data['wellFormedAnswers'][k]
            if type(wfa) == str:
                assert wfa == MARCO_NO_WF_ANSWER, f'Suspected non-null answer in {k}: {wfa}'
                wfa = []
        else:
            wfa = []
        inst = (k, norm_marco_text(q, lowercase),
                data['query_type'][k],
                norm_marco_text('\n'.join(wfa), lowercase),
                norm_marco_text('\n'.join(data['answers'][k]), lowercase) if not is_test else '')
        passages = data['passages'][k]
        if not inst[3]:
            no_wfa += 1
        if not inst[4]:
            no_ans += 1
        true_ps = [p for p in passages if not is_test and p['is_selected']]
        if len(true_ps) == 0:
            no_true_ps += 1
        if len(true_ps) > 1:
            multi_true_ps += 1
        inst += (norm_marco_text(' '.join([p['passage_text'] for p in true_ps]), lowercase),)
        inst += ([norm_marco_text(p['passage_text'], lowercase) for p in passages
                  if not is_test and not p['is_selected']],)
        insts.append(MarcoInstanceText(*inst))
    logger.info(f'of {len(insts)} instances, '
                f'{no_true_ps} have zero true passages and {multi_true_ps} have more than one. '
                f'{no_ans} have no answers, {no_wfa} have no well-formed answers.')
    return QaDataset(insts)


def load_ner(filename, lowercase: bool = False, delim='\t', labcol=1) -> NerDataset:
    with open(filename) as in_f:
        sents = []
        curr_sent_w = []
        curr_sent_t = []
        for line in in_f.readlines():
            line = line.strip()
            if len(line) == 0 or delim not in line:
                sents.append((curr_sent_w, curr_sent_t))
                curr_sent_w = []
                curr_sent_t = []
            else:
                spl = line.split(delim)
                w = spl[0]
                t = spl[labcol]
                curr_sent_w.append(w.lower() if lowercase else w)
                curr_sent_t.append(t)
        return NerDataset(sents)


def emoji_proc(line, lowercase: bool):
    if lowercase:
        line = line.lower()
    return line.strip().split('\t')


def load_emoji(filename, sample=1.0, lowercase: bool = False) -> EmojiDataset:
    with open(filename) as in_f:
        sents = [emoji_proc(line, lowercase) for line in in_f]
    if sample < 1.0:
        random.shuffle(sents)
        sents = sents[:int(len(sents) * sample)]
    return EmojiDataset(list(map(list, zip(*sents))))


# main entry point

def get_dataset(args):
    ds = {}
    #temp_dir = "/tmp/{}".format(uuid.uuid4())
    #os.makedirs(temp_dir)
    for prt in ['train', 'dev', 'test']:
        if 'wnut16' in args.dataset:
            #f = get_ner_dataloc(args.dataloc, prt)
            #tmp_f = os.path.join(temp_dir, prt)
            #tf.io.gfile.copy(f, tmp_f)
            ds[prt] = load_ner(f'{args.dataset}/{prt}', lowercase=args.lowercase)
        elif 'wnut' in args.dataset:
            #f = get_nemer_dataloc(args.dataloc, prt)
            #tmp_f = os.path.join(temp_dir, prt)
            #tf.io.gfile.copy(f, tmp_f)
            ds[prt] = load_ner(f'{args.dataset}/{prt}', lowercase=args.lowercase)
        elif args.dataset == 'conll':
            #f = get_conll_loc(args.dataloc, prt)
            #tmp_f = os.path.join(temp_dir, prt)
            #tf.io.gfile.copy(f, tmp_f)
            ds[prt] = load_ner(f'{args.dataset}/{prt}', lowercase=args.lowercase, delim=' ', labcol=3)
        elif 'marcoqa' in args.dataset:
            #f = get_marco_loc(args.dataloc, prt)
            #tmp_f = os.path.join(temp_dir, prt)
            #tf.io.gfile.copy(f, tmp_f)
            data_sample = args.data_sample if prt == 'train' else args.data_sample * 10
            ds[prt] = load_marco(f'{args.dataset}/{prt}', sample=data_sample,
                                 is_test=prt == 'test',
                                 lowercase=args.lowercase)
        # TODO fix this case's loading
        elif args.dataset in ['marcosamp', 'marcogen']:
            #f = get_marcosamp_loc(args.dataloc, prt)
            #tmp_f = os.path.join(temp_dir, prt)
            #tf.io.gfile.copy(f, tmp_f)
            data_sample = args.data_sample if prt == 'train' else args.data_sample * 10
            ds[prt] = load_marco(f'{args.dataset}/{prt}', sample=data_sample,
                                 is_test=prt == 'test',
                                 lowercase=args.lowercase)
        elif 'emoji' in args.dataset:
            #f = get_emoji_loc(args.dataloc, prt)
            #tmp_f = os.path.join(temp_dir, prt)
            #tf.io.gfile.copy(f, tmp_f)
            data_sample = args.data_sample if prt == 'train' else args.data_sample * 10
            ds[prt] = load_emoji(f'{args.dataset}/{prt}', sample=data_sample, lowercase=args.lowercase)
        else:
            raise ValueError(f'dataset {args.dataset} not supported')
    return ds


# identification methods for branching in train/test

def is_cls(args):
    return args.dataset == 'emoji' or 'emoji' in args.dataset


def is_qa(args):
    return args.dataset in ['marcoqa', 'marcosamp'] or ('marco' in args.dataset and 'gen' not in args.dataset)


def is_gen(args):
    return args.dataset == 'marcogen' or 'marcogen' in args.dataset


def is_ner(args):
    return args.dataset in ['ner', 'nemer', 'conll'] or 'wnut' in args.dataset or 'conll' in args.dataset


# nytwit is a special case

def get_nytwit_dataset(data_loc):
    #temp_dir = "/tmp/{}".format(uuid.uuid4())
    #os.makedirs(temp_dir)
    #tmp_f = os.path.join(temp_dir, 'nytwit')
    #tf.io.gfile.copy(data_loc, tmp_f)

    with open(data_loc) as in_f:
        sents_df = pd.read_csv(in_f, sep='\t', quoting=3)

    return sents_df


def nytwit_vocab(df, only_alpha=False, lowercase=False):
    sents = df['sentence']
    voc = Counter()
    for s in sents:
        toks = s.split()
        if only_alpha:
            toks = [t for t in toks if t.isalpha()]
        if lowercase:
            toks = [t.lower() for t in toks]
        voc.update(toks)
    return voc
