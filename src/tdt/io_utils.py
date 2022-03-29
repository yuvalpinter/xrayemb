import json
import os
import pickle
import uuid
from argparse import Namespace

#import tensorflow as tf
from transformers import BertTokenizer, BertForMaskedLM, \
    GPT2Tokenizer, GPT2LMHeadModel, \
    T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer, RobertaForMaskedLM

from src.tdt.consts import SPECIAL_CHAR_LIST


def load_model_files(model_type, model_dir, temp_dir=None):
    """
    Hacky code for loading all HF-compatible files of a base model through a temporary directory into memory
    :param model_type: string representation of model's name
    :param model_dir: path to directory where model files are located
    :param temp_dir: path for temporary directory if one already exists
    :return: tuple consisting of:
        tokenizer - PreTokenizer used by loaded model
        model - base large language model object
        temp_dir - location of temporary directory for future use (same as input[2] if provided)
    """
    # get files onto temp dir if needed
    #if temp_dir is None:
    #    temp_dir = "/tmp/{}".format(uuid.uuid4())
    #if not os.path.exists(temp_dir):
    #    os.makedirs(temp_dir)

    # aux files
    filenames_list = ["config.json"]
    if model_type == 'bert':
        filenames_list.append('vocab.txt')
    elif model_type == 'cbert':
        filenames_list.extend(['vocab.txt', 'special_tokens_map.json', 'tokenizer_config.json'])
    else:
        filenames_list.extend(['vocab.json', 'merges.txt'])
        if model_type == 'roberta':
            filenames_list.extend(['special_tokens_map.json', 'tokenizer_config.json'])
    #    for fn in filenames_list:
    #        orig = os.path.join(model_dir, fn)
    #        targ = os.path.join(temp_dir, fn)
    #        tf.io.gfile.copy(orig, targ)

    # main model file
    model_orig = model_dir + ".bin" if model_type == 'bert'\
        else os.path.join(model_dir, "pytorch_model.bin")
    model_targ = os.path.join(model_dir, "pytorch_model.bin")
    #    tf.io.gfile.copy(model_orig, model_targ)

    # load
    if model_type in ['bert', 'cbert']:
        tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=False)
        model = BertForMaskedLM.from_pretrained(
            model_dir,
            cache_dir=None
        )
    elif model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(model_dir, use_fast=False)
        model = RobertaForMaskedLM.from_pretrained(
            model_dir,
            cache_dir=None
        )
    elif model_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir, use_fast=False)
        model = GPT2LMHeadModel.from_pretrained(
            model_dir,
            cache_dir=None
        )
    elif model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained(model_dir, use_fast=False)
        model = T5ForConditionalGeneration.from_pretrained(
            model_dir,
            cache_dir=None
        )
    else:
        raise NameError('Only supported model options are "[ro]bert[a]", "gpt2" and "t5".')

    assert model.config.output_hidden_states
    return tokenizer, model, model_dir


def load_tdt_periphery(model_dir, checkpoint=None, temp_dir=None):
    """
    Load peripheral files necessary for TDT operation, not saved with the base model
    :param model_dir: path to location of TDT peripheral files
    :param checkpoint: id for checkpoint model if relevant
    :param temp_dir: path to temporary directory
    :return: tuple consisting of:
        all_chars - character vocabulary for vectorizer calls
        args - argument specifications to be loaded into TDT
        temp_dir - location of temporary directory (same as input[2] if provided)
    """
    #if temp_dir is None:
    #    temp_dir = "/tmp/{}".format(uuid.uuid4())
    #    os.makedirs(temp_dir)
    mod_f = f'{model_dir}/model.pt' if checkpoint is None else f'{model_dir}/checkpoint-{checkpoint}/model.checkpoint'
    #tf.io.gfile.copy(mod_f, f'{temp_dir}/model.pt')
    cvoc_path = os.path.join(model_dir, 'chars.txt')
    #cvoc_tmp_path = os.path.join(temp_dir, 'chars.txt')
    #tf.io.gfile.copy(cvoc_path, cvoc_tmp_path)
    all_chars = load_chars(cvoc_path)
    all_chars.extend(SPECIAL_CHAR_LIST)

    # load params from config file
    args = Namespace()
    #if not tf.io.gfile.exists(f'{model_dir}/config_tdt.json'):
    #    print('No TDT config file found; please populate args file in calling code.')
    #else:
    #    tf.io.gfile.copy(f'{model_dir}/config_tdt.json', f'{temp_dir}/config_tdt.json')
    with open(os.path.join(model_dir, "config_tdt.json")) as tdt_prms:
        cnf_prms = json.loads(tdt_prms.read())
        args.__dict__.update(cnf_prms)
    return all_chars, args, model_dir


def load_chars(file_name):
    with open(file_name, mode="r") as f:
        return list(f.read().rstrip())


def load_cached_dataset(file_path):
    with open(file_path, mode="rb") as f:
        return pickle.load(f)


def write_dataset(ds, file_path):
    with open(file_path, mode="wb") as f:
        pickle.dump(ds, f)
