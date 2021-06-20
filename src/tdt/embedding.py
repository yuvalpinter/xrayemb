import logging
from argparse import Namespace
from typing import List, Union

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, PreTrainedTokenizer

from src.tdt.aux_classes import FilterRules, Pooler, BertTokenMasker, AutoRegressiveMasker
from src.tdt.consts import UNK_CHAR, PAD_CHAR, BOUND_CHAR, VEC_TOKEN, MAX_WORDLEN, IGNORE_INDEX
from src.tdt.utils import make_tformer_config, id_to_token, is_masked_model

logger = logging.getLogger(__name__)


class TdtVectorizer(nn.Module):
    """
    Parent class for Vectorizer API:
    accepts character sequence, returns vector in LLM space.
    """

    def __init__(self, char_vocab, emb_size, trg_size, args: Namespace):
        super().__init__()

        self.in_size = emb_size
        self.out_size = trg_size
        self.device = args.device

        self.char_embs = nn.Embedding(len(char_vocab), self.in_size).to(self.device)


class TdtTransformerVectorizer(TdtVectorizer):
    """
    Vectorizer implemented by (preferrably small) transformer
    """

    def __init__(self, char_vocab, emb_size, trg_size, args):
        super().__init__(char_vocab, emb_size, trg_size, args)

        self.tform_hidden_size = args.tform_hidden_size
        self.char_embs = None

        config = make_tformer_config(args, len(char_vocab), is_decoder=False)
        self.encoder = BertModel(config).to(self.device)

        self.proj = nn.Linear(self.tform_hidden_size, self.out_size).to(self.device)

    def config_params(self):
        return {'tform_hidden_size': self.tform_hidden_size}

    def forward(self, in_tns, **kwargs):
        if len(in_tns.shape) == 1:
            in_tns = in_tns.view(1, -1)

        in_tns = in_tns.to(self.device)
        encoded = self.encoder(in_tns)
        projd = self.proj(torch.nn.functional.relu(encoded[0].mean(axis=1)))

        return projd


class TdtConvVectorizer(TdtVectorizer):
    """
    Vectorizer implemented by convolutional net over character [2-, 3-, 4-]grams
    TODO maybe add a highway layer one day
    """

    def __init__(self, char_vocab, emb_size, trg_size, args):
        super().__init__(char_vocab, emb_size, trg_size, args)

        self.conv_hidden_size = args.conv_hidden_size

        self.bi_conv = nn.Conv1d(in_channels=self.in_size, out_channels=self.conv_hidden_size,
                                 kernel_size=2, padding=1).to(self.device)
        self.tri_conv = nn.Conv1d(in_channels=self.in_size, out_channels=self.conv_hidden_size,
                                  kernel_size=3, padding=2).to(self.device)
        self.quad_conv = nn.Conv1d(in_channels=self.in_size, out_channels=self.conv_hidden_size,
                                   kernel_size=4, padding=3).to(self.device)

        self.bi_pool = nn.AdaptiveMaxPool1d(1).to(self.device)
        self.tri_pool = nn.AdaptiveMaxPool1d(1).to(self.device)
        self.quad_pool = nn.AdaptiveMaxPool1d(1).to(self.device)

        self.proj = nn.Linear(self.conv_hidden_size * 3, self.out_size).to(self.device)
        if (('two_conv_proj' in args and args.two_conv_proj)
                or ('proj_layers' in args and args.proj_layers == 2)):
            logger.info('Initializing second projection layer for conv tok.')
            self.proj2 = nn.Linear(self.out_size, self.out_size).to(self.device)
        else:
            self.proj2 = None

    def config_params(self):
        return {'conv_hidden_size': self.conv_hidden_size,
                'proj_layers': 1 if self.proj2 is None else 2}

    def forward(self, in_tns, **kwargs):
        if len(in_tns.shape) == 1:
            in_tns = in_tns.view(1, -1)

        in_tns = in_tns.to(self.device)
        embs = self.char_embs(in_tns).permute(0, 2, 1)

        bi_chans = self.bi_conv(embs)
        bi_vec = self.bi_pool(bi_chans)
        tri_chans = self.tri_conv(embs)
        tri_vec = self.tri_pool(tri_chans)
        quad_vec = self.quad_pool(self.quad_conv(embs))

        catted = torch.cat([bi_vec, tri_vec, quad_vec], dim=1).sum(dim=-1)
        projd = self.proj(torch.nn.functional.relu(catted))
        if self.proj2 is not None:
            projd = self.proj2(torch.nn.functional.relu(projd))
        return projd


class TdtLstmVectorizer(TdtVectorizer):
    """
    Vectorizer implemented by character-LSTM
    """

    def __init__(self, char_vocab, emb_size, trg_size, args):
        super().__init__(char_vocab, emb_size, trg_size, args)

        self.lstm_hidden_size = args.lstm_hidden_size

        self.h_0 = torch.empty(args.num_lstm_layers * 2, 1, self.lstm_hidden_size) \
            .to(self.device)
        torch.nn.init.xavier_uniform_(self.h_0)
        self.c_0 = torch.zeros(self.h_0.shape, requires_grad=False, device=self.device)
        self.lstm = nn.LSTM(self.in_size, self.lstm_hidden_size, args.num_lstm_layers,
                            batch_first=True, bidirectional=True) \
            .to(self.device)

        self.mlp = nn.ModuleList([nn.Linear(self.lstm_hidden_size * 2, self.lstm_hidden_size * 2).to(self.device)
                                  for _ in range(args.num_mlp_layers - 1)]
                                 + [nn.Linear(self.lstm_hidden_size * 2, self.out_size).to(self.device)])

    def config_params(self):
        return {'lstm_hidden_size': self.lstm_hidden_size,
                'num_lstm_layers': self.lstm.num_layers,
                'num_mlp_layers': len(self.mlp)}

    def forward(self, in_tns, lengths: list = None, **kwargs):
        if len(in_tns.shape) == 1:
            in_tns = in_tns.view(1, -1)
            lengths = [in_tns.shape[1]]
        batch_size, seq_len = in_tns.shape

        in_tns = in_tns.to(self.device)
        embs = self.char_embs(in_tns)
        hidden_inits = self._init_lstm_states(batch_size)
        hid = self.lstm(embs, hidden_inits)[0]  # [1] is (hidden, cell)
        fend = torch.stack([hid[i, l - 1, :self.lstm_hidden_size] for i, l in enumerate(lengths)]).to(self.device)
        bend = hid[:, 0, self.lstm_hidden_size:]
        final_hids = torch.cat([fend, bend], dim=1).to(self.device)
        proj = final_hids
        for lin in self.mlp:
            proj = lin(torch.nn.functional.relu(proj))
        return proj

    def _init_lstm_states(self, batch_size):
        return self.h_0.expand(-1, batch_size, -1).contiguous(), self.c_0.expand(-1, batch_size, -1).contiguous()


class TdtEmbedder(nn.Module):
    """
    learns to predict vectors from word spelling
    """

    def __init__(self, base_tok: PreTrainedTokenizer, base_embs: nn.Embedding, char_vocab, char_emb_size: int,
                 learn_filter_rules: FilterRules, infer_filter_rules: FilterRules,
                 multitoken_pooler: Union[Pooler, None], args: Namespace = None):
        super().__init__()

        self.device = args.device
        self.char_emb_size = char_emb_size

        self.btok = base_tok
        self.bembs = base_embs
        self.learn_filter_rules = learn_filter_rules
        self.infer_filter_rules = infer_filter_rules
        self.pool = multitoken_pooler
        if is_masked_model(args.model_type):
            self.masker = BertTokenMasker(self.btok, args)
        else:
            self.masker = AutoRegressiveMasker(self.btok, args)
        if args.vectorizer_type == 'lstm':
            self.vectorizer = TdtLstmVectorizer(char_vocab=char_vocab,
                                                emb_size=self.char_emb_size,
                                                trg_size=self.word_emb_dim(),
                                                args=args)
        elif args.vectorizer_type == 'conv':
            self.vectorizer = TdtConvVectorizer(char_vocab=char_vocab,
                                                emb_size=self.char_emb_size,
                                                trg_size=self.word_emb_dim(),
                                                args=args)
        elif args.vectorizer_type == 'transformer':
            self.vectorizer = TdtTransformerVectorizer(char_vocab=char_vocab,
                                                       emb_size=self.char_emb_size,
                                                       trg_size=self.word_emb_dim(),
                                                       args=args)
        else:
            raise NotImplementedError(f'{args.vectorizer_type} not supported as vectorizer.')
        assert UNK_CHAR in char_vocab
        assert PAD_CHAR in char_vocab
        assert BOUND_CHAR in char_vocab

        self.char_to_ids = {c: i for i, c in enumerate(char_vocab)}
        self.special_token_id = self.btok.encode(VEC_TOKEN, add_special_tokens=False)[0]
        logger.info(f'Special token ID is {self.special_token_id}.')

        self.rep_tok_for_masking = self.special_token_id
        logger.info(f'Marking vectorized words using {self.rep_tok_for_masking}.')

    def word_emb_dim(self):
        return self.bembs.embedding_dim

    def token_for_vectorizer_input(self, tok_id: int) -> str:
        return id_to_token(self.btok, tok_id, clean=True)

    def char_vec(self, word: str):
        return torch.tensor([self.char_to_ids.get(c, self.char_to_ids[UNK_CHAR])
                             for c in [BOUND_CHAR] + list(word)[:MAX_WORDLEN] + [BOUND_CHAR]],
                            dtype=torch.int64, device=self.device)

    def config_params(self):
        params = {'char_emb_size': self.char_emb_size}
        params.update(self.vectorizer.config_params())
        return params

    def forward(self, in_ids: torch.Tensor, mask=True, get_infer_map=False):
        """
        :param in_ids: input token IDs produced by a PreTrainedTokenizer
        :param mask: is this a masking objective
        :param get_infer_map: should an inference mapping of replaced tokens be returned
        :return: a tuple containing:
            masked_in_ids - input IDs for LM
            encode_embs - input embeddings for LM (overrides the above)
            mask_labs - mask labels for LM (which token IDs to predict)
            attn_mask - attention mask for LM (which token locations are masked)
            lrn_batch - inputs/targets for vectorizer objective
            orig_words - original word sequence for debugging and restoration
            (if get_infer_map)
                infer_replacement_map - mapping of locations that were overriden by vectorizer outputs
        """

        # position embeddings don't need touching, they're set in BertEmbeddings
        if len(in_ids.shape) == 1:
            in_ids = in_ids.view(1, -1)
        in_ids = in_ids.to(self.device)
        seqlen = in_ids.shape[1]

        # token ids to pass as training data
        learn_replacement_map = self.learn_filter_rules(in_ids).to(self.device)
        # token ids to replace with predicted vectors
        infer_replacement_map = self.infer_filter_rules(in_ids).to(self.device)

        # create new index-form inputs with replacements marked
        inf_idcs = []
        inf_vect_inputs = []
        new_inputs = []  # used for downstream inference -- ignores learning pass
        curr_lrn_rep_id = -1
        curr_inf_rep_id = -1
        curr_lrn_inp = ''
        curr_lrn_trgs = []
        curr_inf_inp = ''
        lrn_insts = []
        orig_words = {}

        for i, s in enumerate(in_ids.tolist()):
            # new_s will be passed to mask_tokens(), where rep_tok_for_masking is protected from masking
            new_s = []
            for j, t in enumerate(s):
                if t == self.btok.pad_token_id:
                    break
                t_spctok = t in self.btok.all_special_ids
                inf_repl = infer_replacement_map[i, j]
                lrn_repl = learn_replacement_map[i, j]

                # flush buffer if replacement ended
                if (t_spctok or lrn_repl < 0) and curr_lrn_inp:
                    cvec = self.char_vec(curr_lrn_inp)
                    lrn_insts.append((cvec, self.poolify_inds(curr_lrn_trgs)))
                    curr_lrn_inp = ''
                    curr_lrn_trgs = []
                if (t_spctok or inf_repl < 0) and curr_inf_inp:
                    coord = (i, len(new_s))
                    inf_idcs.append(coord)
                    orig_words[coord] = curr_inf_inp
                    inf_vect_inputs.append(self.char_vec(curr_inf_inp))
                    curr_inf_inp = ''
                    # masking lrn toks is ok, masking inf toks is not
                    new_s.append(self.rep_tok_for_masking)
                if t_spctok or (lrn_repl < 0 and inf_repl < 0):
                    new_s.append(t)

                else:
                    # update buffers
                    if lrn_repl >= 0:
                        if lrn_repl == curr_lrn_rep_id:
                            # append characters and targets
                            curr_lrn_inp += self.token_for_vectorizer_input(t)
                            curr_lrn_trgs.append(t)
                        else:
                            if curr_lrn_inp:  # consecutive but different vectorizations
                                cvec = self.char_vec(curr_lrn_inp)
                                lrn_insts.append((cvec, self.poolify_inds(curr_lrn_trgs)))
                            curr_lrn_rep_id = lrn_repl
                            curr_lrn_inp = self.token_for_vectorizer_input(t)
                            curr_lrn_trgs = [t]
                    if inf_repl >= 0:
                        if inf_repl == curr_inf_rep_id:
                            # append characters
                            curr_inf_inp += self.token_for_vectorizer_input(t)
                        else:
                            if curr_inf_inp:  # consecutive but different vectorizations
                                coord = (i, len(new_s))
                                inf_idcs.append(coord)
                                orig_words[coord] = curr_inf_inp
                                inf_vect_inputs.append(self.char_vec(curr_inf_inp))
                                new_s.append(self.rep_tok_for_masking)
                            curr_inf_rep_id = inf_repl
                            curr_inf_inp = self.token_for_vectorizer_input(t)
                    else:
                        new_s.append(t)

            # the following two shouldn't happen anymore, since sentences end with sep_token
            if curr_lrn_inp:
                cvec = self.char_vec(curr_lrn_inp)
                lrn_insts.append((cvec, self.poolify_inds(curr_lrn_trgs)))
                curr_lrn_inp = ''
                curr_lrn_trgs = []
            if curr_inf_inp:
                coord = (i, len(new_s))
                inf_idcs.append(coord)
                orig_words[coord] = curr_inf_inp
                inf_vect_inputs.append(self.char_vec(curr_inf_inp))
                new_s.append(self.rep_tok_for_masking)
                curr_inf_inp = ''

            new_inputs.append(new_s + [self.btok.pad_token_id] * (seqlen - len(new_s)))
        vec_pad_inp = torch.tensor(new_inputs, device=self.device)

        if mask:
            # compute mask, excluding vectorized replacements
            masked_in_ids, mask_labs, attn_mask = self.masker(vec_pad_inp)
        else:
            masked_in_ids = vec_pad_inp  # in_ids
            mask_labs = torch.full_like(vec_pad_inp, IGNORE_INDEX, dtype=torch.long, device=self.device)
            attn_mask = None

        # infer new embeddings and use base embeddings on the rest
        encode_embs = self.bembs(masked_in_ids).to(self.device)
        if len(inf_vect_inputs) > 0:
            # inf_vect_inputs = [self.char_vec(s) for s in inf_strs]
            inf_batch, inf_lens = self.batchify(inf_vect_inputs)
            inferred_embs = self.vectorizer(inf_batch, lengths=inf_lens)
            for k, (i, j) in enumerate(inf_idcs):
                encode_embs[i, j] = inferred_embs[k]

        # prepare encoding learner input and targets
        if len(lrn_insts) > 0:
            lrn_inp_batch, lrn_lens = self.batchify([x[0] for x in lrn_insts])
            lrn_preds_batch = self.vectorizer(lrn_inp_batch, lengths=lrn_lens)
            lrn_trg_batch = torch.cat([x[1] for x in lrn_insts]).to(self.device)
            lrn_batch = (lrn_preds_batch, lrn_trg_batch, lrn_inp_batch)
        else:
            lrn_batch = None

        # construct return tuple
        ret = (masked_in_ids.to(self.device), encode_embs,
               mask_labs.to(self.device), attn_mask,
               lrn_batch, orig_words)
        if get_infer_map:
            ret += (infer_replacement_map, )
        return ret

    def batchify(self, insts: List[torch.Tensor]):
        assert len(insts) > 0
        assert self.char_to_ids[PAD_CHAR] > 0

        lens = [x.shape[0] for x in insts]
        batch_tns = pad_sequence(insts, batch_first=True, padding_value=self.char_to_ids[PAD_CHAR])
        batch_tns = batch_tns.to(self.device)
        return batch_tns, lens

    def poolify_inds(self, ids):
        if self.pool is None:
            return torch.zeros((1, self.bembs.embedding_dim), device=self.device)
        return self.pool(self.bembs(torch.tensor(ids, device=self.device))).view(1, -1).to(self.device)
