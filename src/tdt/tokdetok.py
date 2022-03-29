"""
Main module wrapping a TDT model
"""
import json
import logging
import os
from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers import PreTrainedTokenizer, PreTrainedModel, \
    BertTokenizer, BertForMaskedLM, \
    GPT2Tokenizer, GPT2LMHeadModel, \
    T5Tokenizer, T5ForConditionalGeneration, \
    RobertaTokenizer, RobertaForMaskedLM

from src.tdt.aux_classes import FilterRules, BertTokenMasker, StochasticFilter, AllMultiToksFilter, \
    MaxPooler, AvgPooler, AutoRegressiveMasker, NoEasySuffsFilter, FirstTokPooler
from src.tdt.consts import PAD_CHAR, MASK_TOKEN_KEY
from src.tdt.embedding import TdtEmbedder
from src.tdt.generation import TdtGenerator
from src.tdt.io_utils import load_model_files, load_tdt_periphery
#from src.tdt.nlp_utils import NlpTokenizer
from src.tdt.utils import add_vector_token, add_pad_token

logger = logging.getLogger(__name__)


def load_wrapper(model_type, base_dir, tdt_dir, vec_type, checkpoint=None, device='cpu', for_lm=True, hashtml=False,
                 infer_policy='all-multi', pool_policy='max', stoch_rate: float = -1.0):
    """
    Must have config files present in directory
    :param stoch_rate: rate for inferring vectorized tokens stochastically if infer_policy=='stoch'
    :param pool_policy: policy for pooling multiple token embeddings into one target vector
    :param infer_policy: policy for selecting vectorized tokens for downstream LM
    :param for_lm: does the task involve language modeling
    :param device: torch device
    :param checkpoint: load a certain (named) checkpoint
    :param tdt_dir: path for directory containing TDT files
    :param base_dir: path for directory containing base LM files
    :param model_type: flavor of LM (gpt2, bert, etc.)
    :param hashtml: will the text encountered have unescaped HTML
    :param vec_type: type of vectorizer (lstm, conv, transformer)
    """
    #pret = NlpTokenizer(hashtml=hashtml)
    pret = None
    btok, bmod, _ = load_model_files(model_type, base_dir)

    # add tokens (can be done directly when json files are updated in directory or loading from post-pre-trained)
    if model_type == 'gpt2':
        add_pad_token(btok)
    if tdt_dir is not None:
        # this should ideally be "if vec_type is not None" but upstream implementation is complex
        add_vector_token(btok)
    bmod.resize_token_embeddings(len(btok))
    bmod.to(device)

    if tdt_dir is None:
        return bmod, btok, None

    all_chars, args, _ = load_tdt_periphery(tdt_dir, checkpoint, None)
    args.device = device
    args.vectorizer_type = vec_type

    if vec_type is None:  # second-base model
        sd = torch.load(f'{base_dir}/model.pt', map_location=torch.device(device))
        if checkpoint is not None:
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        sd = {k[len('bmodel.'):]: v for k, v in sd.items() if k.startswith('bmodel.')}
        bmod.load_state_dict(sd, strict=False)
        return bmod, btok, None

    if infer_policy == 'all-multi':
        infer_filter = AllMultiToksFilter(btok)
    elif infer_policy == 'no-easy-suffs':
        infer_filter = NoEasySuffsFilter(btok)
    elif infer_policy == 'all':
        infer_filter = StochasticFilter(btok, 1.0)
    elif infer_policy == 'stoch':
        assert 0.0 <= stoch_rate <= 1.0, f'stochastic inference policy incompatible with rate {stoch_rate}'
        infer_filter = StochasticFilter(btok, stoch_rate)
    else:
        raise ValueError(f'TDT inference policy not supported: {infer_policy}')
    if pool_policy == 'max':
        pooler = MaxPooler()
    elif pool_policy == 'avg':
        pooler = AvgPooler()
    elif pool_policy == 'first':
        pooler = FirstTokPooler()
    else:
        raise ValueError(f'Multi-token pooling policy not supported: {pool_policy}')
    strict_dict_loading = False
    if for_lm:
        logger.info("Loading embedder and generator 'for LM'")
        tdte = TdtEmbedder(btok, bmod.get_input_embeddings(), all_chars,
                           args.char_emb_size, StochasticFilter(btok, 0.2), infer_filter,
                           pooler, args).to(device)
        tdtg = TdtGenerator(all_chars, args.char_emb_size,
                            bmod.get_output_embeddings().in_features,
                            args).to(device)
        strict_dict_loading = True
    else:  # this is downstream, so inference only and no generation
        logger.info("Loading embedder without generation")
        tdte = TdtEmbedder(btok, bmod.get_input_embeddings(), all_chars,
                           args.char_emb_size, FilterRules(), infer_filter,
                           None, args).to(device)
        tdtg = None
    tdtmod = TdtWrapper(bmod, pret, btok, tdte, tdtg, args).to(device)

    sd = torch.load(f'{tdt_dir}/model.pt', map_location=torch.device(device))
    if checkpoint is not None:
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    tdtmod.load_state_dict(sd, strict=strict_dict_loading)

    return tdtmod, None, all_chars


class TdtWrapper(nn.Module):
    """
    Main class to save, load, train, run, everything.
    No trainable modules should live outside it.
    """

    def __init__(self,
                 base_model: PreTrainedModel,
                 pretokenizer,
                 base_tokenizer: PreTrainedTokenizer,
                 tdt_embedder: TdtEmbedder,
                 tdt_generator: TdtGenerator,
                 args: Namespace):
        super(TdtWrapper, self).__init__()

        self.mtype = args.model_type

        self.bmodel = base_model
        #self.pretok = pretokenizer
        self.btok = base_tokenizer
        self.tdtemb = tdt_embedder
        self.tdtgen = tdt_generator
        self.device = args.device

        self.masker = None  # filled in by subclass

    def vocab_size(self):
        tok_vs = len(self.btok)
        in_vs = self.bmodel.get_input_embeddings().num_embeddings
        assert in_vs == tok_vs
        out_vs = self.bmodel.get_output_embeddings().out_features
        assert out_vs == in_vs
        return out_vs

    def out_size(self):
        return self.bmodel.get_output_embeddings().in_features

    def char_vec(self, w) -> torch.Tensor:
        return self.tdtemb.char_vec(w)

    def tok_form(self, idx: int) -> str:
        return self.tdtemb.token_for_vectorizer_input(idx)

    def char_to_ids(self, c: str) -> int:
        return self.tdtemb.char_to_ids[c]

    def encode(self, text):
        return self.btok.batch_encode_plus(text)

    def is_mask(self, idx):
        if MASK_TOKEN_KEY not in self.btok.special_tokens_map:
            return False
        return idx == self.btok.mask_token_id

    def forward(self, in_ids: torch.Tensor, mask: bool = False, generate_all: bool = False,
                other_inp=None, action='mlm', get_inputs=False, get_embs=False, get_joins=False, **kwargs):
        """
        Accepts an encoded batch, returns an MLM output as well as loss-ready vectorizer and generator batches.
        :param get_joins: also return mapping from original index location to joined tokens indicators
        :param get_embs: also return embeddings input into the base LM (use for assertions)
        :param get_inputs: also return input indices, e.g. for alignment check
        :param in_ids: tokenizer-encoded batch (input_ids member output from batch_encode_plus())
        :param mask: should masked objective (MLM) be pursued (default: False)
        :param generate_all: should all words be generated from detokenizer (default: False, useful for pre-training)
        :param other_inp: ugly way of passing cycle dependency inputs without breaking multi-gpu trainability
        :param action: ugly way of passing whether cycle loop is detok->tok ("dt_cycle") or tok->detok ("td_cycle")
        :return: lm_out: final-layer vectors corresponding to all tokens in retokenized input
                 vec_lrn_batch: batch for tokenizer training (tuple of the form (predictions, target vectors))
                 generated: batch for detokenizer loss (tuple of the form (predictions, target indices))
        """
        # yes this is ugly but only __call__() works properly from distributed
        if action == 'td_cycle':
            char_ins = [self.char_vec(w) for w in other_inp]
            lens = [t.shape[0] for t in char_ins]
            in_batch = pad_sequence(char_ins, batch_first=True, padding_value=self.char_to_ids(PAD_CHAR))
            vecs = self.tdtemb.vectorizer(in_batch, lengths=lens)
            return self.generate_batch_for_loss(vecs, char_ins, lens)
        elif action == 'dt_cycle':
            wtns, _, lens = self.tdtgen.generate(other_inp)
            vecs = self.tdtemb.vectorizer(wtns, lengths=lens)
            return vecs, other_inp

        in_seqlen = in_ids.shape[1]
        if self.tdtemb is not None:
            inp_ids, inp_embs, mask_labels, attn_mask, vec_lrn_batch, orig_words, joins =\
                self.tdtemb(in_ids, mask=mask, get_infer_map=True)
            out_seqlen = inp_embs.shape[1]

            if out_seqlen > self.bmodel.config.max_position_embeddings - 2:  # possible result of re-tokenizing
                logger.warning(f'Sequence expanded from {in_seqlen} to {out_seqlen}; aborting.')
                ret = (None, None, None)
                if get_embs:
                    ret += (inp_embs,)
                if get_inputs:
                    ret += (inp_ids,)
                if get_joins:
                    ret += (joins,)
                return ret

            inp_embs = inp_embs.to(self.device)
            mask_labels = mask_labels.to(self.device)

            lm_out = self.run_base(input_ids=None,
                                   attention_mask=attn_mask,
                                   token_type_ids=None,  # these are the "sentence embeddings",
                                   # created on-the-fly in BertEmbeddings.forward()
                                   position_ids=None,  # created on-the-fly in BertEmbeddings.forward()
                                   head_mask=None,
                                   inputs_embeds=inp_embs,
                                   masked_lm_labels=mask_labels,
                                   past_hidden_values=None,
                                   encoder_attention_mask=None,
                                   lm_labels=None)
        else:
            vec_lrn_batch, mask_labels, inp_ids, orig_words, joins = None, None, None, None, None
            masked_in_ids, mask_labs, attn_mask = self.masker(in_ids)
            lm_out = self.run_base(input_ids=masked_in_ids,
                                   attention_mask=attn_mask,
                                   masked_lm_labels=mask_labs)

        # lm_out == (masked_lm_loss, prediction_scores, hidden_states)
        if self.mtype in ['bert', 'cbert', 'roberta']:
            assert len(lm_out) == 3, f"Output length from LM is {len(lm_out)}"
        elif self.mtype == 'gpt2':
            assert len(lm_out) == 4, f"Output length from LM is {len(lm_out)}"
            lm_out = lm_out[:2] + lm_out[-1:]  # get rid of "past"
        else:
            raise NotImplementedError(f'Please implement output assertion for {self.mtype}')

        # generation
        if self.tdtgen is not None:
            assert self.tdtemb is not None  # asserts mask_labels, inp_ids, orig_words exist

            out_vecs = lm_out[-1][-1]  # shape: (batch_size, sequence_length, hidden_size)
            target_vecs = []
            gold_ids = []
            lens = []

            for i, (vec_seq, mask_seq, inp_seq) in enumerate(zip(out_vecs, mask_labels, inp_ids)):
                for j, (v, m, idx) in enumerate(zip(vec_seq, mask_seq, inp_seq)):
                    if self.is_mask(idx) or (generate_all and idx not in self.btok.all_special_ids):
                        wid = idx if not self.is_mask(idx) else m
                        if wid == self.tdtemb.rep_tok_for_masking:
                            if (i, j) not in orig_words:
                                logger.info(f'Did not find index ({i},{j}) in original words mapping:\n'
                                            + str(orig_words) + f'\nword id = {wid}, index = {idx}')
                                continue
                            word_form = orig_words[(i, j)]
                        else:
                            word_form = self.tok_form(wid.item())
                        cvec = self.char_vec(word_form)
                        gold_ids.append(cvec)
                        target_vecs.append(v)
                        lens.append(cvec.shape[0])

            # sort from longest
            len_argsort = list(reversed(np.argsort(lens)))
            lens = np.array(lens)[len_argsort].tolist()
            gold_ids = np.array(gold_ids)[len_argsort].tolist()
            in_vecs_lst = np.array(target_vecs)[len_argsort].tolist()
            if not in_vecs_lst:
                generated = None
            else:
                in_vecs = torch.stack(in_vecs_lst).to(self.device)
                generated = self.generate_batch_for_loss(in_vecs, gold_ids, lens)
        else:
            generated = None

        ret = (lm_out, vec_lrn_batch, generated)
        if get_embs:
            ret += (inp_embs, )
        if get_inputs:
            ret += (inp_ids, )
        if get_joins:
            ret += (joins, )
        return ret

    def run_base(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                 inputs_embeds=None, masked_lm_labels=None, past_hidden_values=None,
                 encoder_attention_mask=None, lm_labels=None):
        if self.mtype in ['bert', 'cbert']:
            return self.bmodel(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                               labels=masked_lm_labels, encoder_hidden_states=past_hidden_values,
                               encoder_attention_mask=encoder_attention_mask, lm_labels=lm_labels)
        elif self.mtype == 'roberta':
            return self.bmodel(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                               labels=masked_lm_labels)
        elif self.mtype == 'gpt2':
            return self.bmodel(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                               labels=masked_lm_labels, past_key_values=past_hidden_values)
        else:
            raise NotImplementedError('Only [Ro]BERT[a] and GPT2 supported for now.')

    def generate_batch_for_loss(self, in_vecs, gold_ids, lens):
        gold_ids_tns = pad_sequence(gold_ids, batch_first=True, padding_value=self.char_to_ids(PAD_CHAR)) \
            .to(self.device)
        preds = self.tdtgen(in_vecs, gold_ids_tns)
        gold_outs = gold_ids_tns[:, 1:]
        out_lens = [ln - 1 for ln in lens]
        packed_preds = pack_padded_sequence(preds, out_lens, batch_first=True).data.to(self.device)
        packed_gold = pack_padded_sequence(gold_outs, out_lens, batch_first=True).data.to(self.device)
        assert packed_gold.min() >= 0
        assert packed_gold.max() < packed_preds.shape[1], \
            f"Found gold character {packed_gold.max()} on prediction tensor of shape {list(packed_preds.shape)}"
        return packed_preds, packed_gold

    def save(self, save_dir):
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
        self.btok.save_pretrained(save_dir)
        self.bmodel.config.to_json_file(os.path.join(save_dir, "config_base.json"))

        if self.tdtemb is not None:
            tdt_cfg = json.dumps(self.config_params(), indent=2, sort_keys=True) + "\n"
            with open(os.path.join(save_dir, "config_tdt.json"), "w", encoding="utf-8") as writer:
                writer.write(tdt_cfg)


# Implementing classes, mainly for distinguishing the params and for downstream flavor recognition

class RobertaTdtWrapper(TdtWrapper):
    def __init__(self,
                 base_model: RobertaForMaskedLM,
                 pretokenizer,
                 base_tokenizer: RobertaTokenizer,
                 tdt_embedder: TdtEmbedder,
                 tdt_generator: TdtGenerator,
                 args: Namespace):
        super(RobertaTdtWrapper, self).__init__(base_model, pretokenizer, base_tokenizer, tdt_embedder,
                                                tdt_generator, args)

        if self.tdtemb is None:
            assert isinstance(self.btok, RobertaTokenizer)
            logger.info('Masking using BERT policy.')
            self.masker = BertTokenMasker(self.btok, args)

    def config_params(self):
        params = {'model_type': 'roberta',
                  'char_emb_size': self.tdtemb.char_emb_size,
                  'mlm_probability': self.tdtemb.masker.mlm_prob,
                  'max_gen': self.tdtgen.max_gen,
                  'gen_lstm_hidden_size': self.tdtgen.lstm_hidden_size,
                  'gen_num_lstm_layers': self.tdtgen.num_lstm_layers}
        params.update(self.tdtemb.config_params())
        return params


class BertTdtWrapper(TdtWrapper):
    def __init__(self,
                 base_model: BertForMaskedLM,
                 pretokenizer,
                 base_tokenizer: BertTokenizer,
                 tdt_embedder: TdtEmbedder,
                 tdt_generator: TdtGenerator,
                 args: Namespace):
        super(BertTdtWrapper, self).__init__(base_model, pretokenizer, base_tokenizer, tdt_embedder,
                                             tdt_generator, args)

        if self.tdtemb is None:
            assert isinstance(self.btok, BertTokenizer)
            self.masker = BertTokenMasker(self.btok, args)

    def config_params(self):
        params = {'model_type': 'bert',
                  'char_emb_size': self.tdtemb.char_emb_size,
                  'mlm_probability': self.tdtemb.masker.mlm_prob,
                  'max_gen': self.tdtgen.max_gen,
                  'gen_lstm_hidden_size': self.tdtgen.lstm_hidden_size,
                  'gen_num_lstm_layers': self.tdtgen.num_lstm_layers}
        params.update(self.tdtemb.config_params())
        return params


class GptTdtWrapper(TdtWrapper):
    def __init__(self,
                 base_model: GPT2LMHeadModel,
                 pretokenizer,
                 base_tokenizer: GPT2Tokenizer,
                 tdt_embedder: TdtEmbedder,
                 tdt_generator: TdtGenerator,
                 args: Namespace):
        super(GptTdtWrapper, self).__init__(base_model, pretokenizer, base_tokenizer, tdt_embedder,
                                            tdt_generator, args)

        self.masker = AutoRegressiveMasker(self.btok, args)

    def config_params(self):
        params = {'model_type': 'gpt2',
                  'char_emb_size': self.tdtemb.char_emb_size,
                  'max_gen': self.tdtgen.max_gen,
                  'gen_lstm_hidden_size': self.tdtgen.lstm_hidden_size,
                  'gen_num_lstm_layers': self.tdtgen.num_lstm_layers}
        params.update(self.tdtemb.config_params())
        return params


class T5TdtWrapper(TdtWrapper):
    def __init__(self,
                 base_model: T5ForConditionalGeneration,
                 pretokenizer,
                 base_tokenizer: T5Tokenizer,
                 tdt_embedder: TdtEmbedder,
                 tdt_generator: TdtGenerator,
                 args: Namespace):
        super(T5TdtWrapper, self).__init__(base_model, pretokenizer, base_tokenizer, tdt_embedder,
                                           tdt_generator, args)

        self.masker = AutoRegressiveMasker(self.btok, args)

    def config_params(self):
        params = {'model_type': 't5',
                  'char_emb_size': self.tdtemb.char_emb_size,
                  'max_gen': self.tdtgen.max_gen,
                  'gen_lstm_hidden_size': self.tdtgen.lstm_hidden_size,
                  'gen_num_lstm_layers': self.tdtgen.num_lstm_layers}
        params.update(self.tdtemb.config_params())
        return params
