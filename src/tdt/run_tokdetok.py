"""
Run a model augmenting a large language model (LLM) with a Tokdetok apparatus (TDT)
See argparse help for individual parameter usage
"""
import argparse
import logging
import os

import torch

from src.tdt.aux_classes import AllMultiToksFilter, StochasticFilter, \
    MaxPooler, AvgPooler, FirstTokPooler
from src.tdt.consts import *
from src.tdt.cycles import TdCycleTrainer, DtCycleTrainer
from src.tdt.data import load_dataset, load_vocab
from src.tdt.embedding import TdtEmbedder
from src.tdt.evaluation import evaluate
from src.tdt.generation import TdtGenerator
from src.tdt.io_utils import load_model_files, write_dataset, load_cached_dataset
from src.tdt.tokdetok import BertTdtWrapper, T5TdtWrapper, GptTdtWrapper, RobertaTdtWrapper
from src.tdt.training import train
from src.tdt.utils import PreTokenizer, set_seed, add_vector_token, add_pad_token

VEC_CHOICES = ['lstm', 'conv', 'transformer']

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.NOTSET)


def main():
    parser = argparse.ArgumentParser()

    # system
    parser.add_argument('--seed', type=int, default=496351,
                        help="Random seed to be set in all modules; enter -1 for no setting")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help="For distributed training: local_rank (do not set in external call)")
    parser.add_argument('--n-gpu', type=int, default=1, help="number of GPUs")
    parser.add_argument('--device', default='cuda', help="torch device")
    parser.add_argument('--no-cuda', action='store_true', help="do not use cuda")
    parser.add_argument("--fp16", action="store_true",
                        help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

    # I/O
    parser.add_argument('--model-type', choices=['bert', 'cbert', 'gpt2', 't5', 'roberta'])
    parser.add_argument('--model-dir', help="location of base model files")
    parser.add_argument('--train-data-file', help="location of second pretraining corpus")
    parser.add_argument('--eval-data-file', help="location of evaluation corpus")
    parser.add_argument('--vocab-file', help="location of vocabulary file (word \t frequency, ordered by frequency)")
    parser.add_argument('--output-dir', help="location of resulting TDT model files")
    parser.add_argument('--logdir', help="location of tensorboard log")

    # Base LM
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE, help="LM transformer block size")
    parser.add_argument('--mlm-probability', type=float, default=0.15, help="proportion of tokens to be masked in MLM")
    parser.add_argument("--lowercase-vocab", action='store_true', help="should vocab be lowercased")
    parser.add_argument("--hashtml", action='store_true', help="data with html gets unescaped first")
    parser.add_argument('--vocab-size', type=int, default=100000,
                        help="vocabulary size for cycle training (-1 => no limit)")

    # Data
    parser.add_argument('--train-data-portion', type=float, default=1.0,
                        help="Portion of lines taken for training (1.0 for all).")
    parser.add_argument('--line-by-line', action="store_true",
                        help="If toggled, distinct lines of text in the dataset"
                             "are to be handled as distinct sequences.")
    parser.add_argument('--shuffle-data', action="store_true", help="Shuffle input data (only in line-by-line mode).")

    # TDT & base
    parser.add_argument('--lrn-tdt', action='store_true', help="is TDT trained in this run")
    parser.add_argument('--lrn-prob', type=float, default=0.5,
                        help="proportion of words to be used for TDT learning")
    parser.add_argument('--per-gpu-train-batch-size', type=int, default=8,
                        help="batch size for each GPU core (training)")
    parser.add_argument('--per-gpu-eval-batch-size', type=int, default=8,
                        help="batch size for each GPU core (evaluation)")
    parser.add_argument("--evaluate-during-training", action="store_true",
                        help="run evaluation during training at each logging step")
    parser.add_argument("--max-steps", default=-1, type=int,
                        help="if > 0: set total number of training steps to perform. Overrides num_train_epochs.")
    parser.add_argument("--warmup-steps", default=0.1, type=float, help="linear warmup over warmup_steps * total.")
    parser.add_argument("--logging-steps", type=int, default=500, help="log every X updates steps.")
    parser.add_argument("--save-steps", type=int, default=500, help="save checkpoint every X updates steps.")
    parser.add_argument("--save-total-limit", type=int, default=None,
                        help="limit the total amount of checkpoints, delete the older"
                             "checkpoints in the output_dir, does not delete by default")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning-rate", default=2e-5, type=float, help="initial learning rate for Adam.")
    parser.add_argument("--weight-decay", default=0.0, type=float, help="weight decay if we apply some.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="epsilon for Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("--num-train-epochs", default=1.0, type=float,
                        help="total number of training epochs to perform.")
    parser.add_argument("--report", action='store_true', help="add reports, such as intermittent generation samples.")

    # TDT flow
    parser.add_argument("--alpha-vec", default=0.25, type=float, help="weight for vectorizer loss")
    parser.add_argument("--alpha-gen", default=0.25, type=float, help="weight for generator loss")
    parser.add_argument("--stochastic-inference", action='store_true', help="use stochastic sampling for inference")
    parser.add_argument('--max-gen', type=int, default=MAX_WORDLEN,
                        help='longest generated word possible')
    parser.add_argument('--char-emb-size', type=int, default=CHAR_EMB_SIZE, help="character embedding size")
    parser.add_argument('--train-cycle-dep', action='store_true', help="toggle if cycle dependency is sought")
    parser.add_argument('--cycle-freq', type=int, default=3000,
                        help="number of LM iterations between cycle dependency loops")
    parser.add_argument('--cycle-batch-iters', type=int, default=1000, help="number of iterations in each cycle loop")
    parser.add_argument('--td-strategy', choices=['uniform', 'sqrt', 'freq'], default='uniform',
                        help="strategy for sampling words for TD loops, given frequency data")
    parser.add_argument("--alpha-cyc-td", default=0.1, type=float,
                        help="weight for cycle dependency loss (T->D, word to word).")
    parser.add_argument("--alpha-cyc-dt", default=0.1, type=float,
                        help="weight for cycle dependency loss (D->T, vector to vector).")

    # vectorizer
    parser.add_argument('--vectorizer-type', choices=VEC_CHOICES, default='lstm',
                        help=f"type of vectorizer, available types are {VEC_CHOICES}")
    parser.add_argument('--pool-policy', choices=['max', 'avg', 'first'], default='max',
                        help="policy for pooling multi-tokens, available flavors are max, avg, first")
    parser.add_argument('--num-mlp-layers', type=int, default=MLP_NUM_LAYERS,
                        help=f"number of perceptron layers, default={MLP_NUM_LAYERS}")
    parser.add_argument('--lstm-hidden-size', type=int, default=LSTM_HIDDEN_SIZE,
                        help=f"hidden size for LSTM vectorizer, default={LSTM_HIDDEN_SIZE}")
    parser.add_argument('--num-lstm-layers', type=int, default=LSTM_NUM_LAYERS,
                        help=f"number of layers for LSTM vectorizer, default={LSTM_NUM_LAYERS}")
    parser.add_argument('--conv-hidden-size', type=int, default=CONV_HIDDEN_SIZE,
                        help=f"hidden size for convnet vectorizer, default={CONV_HIDDEN_SIZE}")
    parser.add_argument('--two-conv-proj', action="store_true", help='two projection layers in conv Tok')
    parser.add_argument('--tform-hidden-size', type=int, default=TRANSFORMER_HIDDEN_SIZE,
                        help=f"hidden size for transformer vectorizer, default={TRANSFORMER_HIDDEN_SIZE}")
    parser.add_argument('--tform-heads', type=int, default=TRANSFORMER_NUM_HEADS,
                        help=f"number of heads for transformer vectorizer, default={TRANSFORMER_NUM_HEADS}")
    parser.add_argument('--num-tform-layers', type=int, default=TRANSFORMER_NUM_LAYERS,
                        help=f"number of layers for transformer vectorizer, default={TRANSFORMER_NUM_LAYERS}")
    parser.add_argument('--max-position-embeddings', type=int, default=MAX_IN_WORDLEN,
                        help=f"maximum word length for transformer vectorizer, default={MAX_IN_WORDLEN}")

    # generator
    parser.add_argument('--gen-lstm-hidden-size', type=int, default=GEN_LSTM_HIDDEN_SIZE,
                        help=f"hidden size for LSTM generator, default={GEN_LSTM_HIDDEN_SIZE}")
    parser.add_argument('--gen-num-lstm-layers', type=int, default=GEN_LSTM_NUM_LAYERS,
                        help=f"number of layers for LSTM generator, default={GEN_LSTM_NUM_LAYERS}")
    parser.add_argument('--generate-all', action="store_true",
                        help="whether to generate all words from vectors while training, otherwise just masks")
    parser.add_argument('--spaces-end', action="store_true", help="all space characters end generation")

    args = parser.parse_args()

    # Set up CUDA, GPU & distributed training
    if args.local_rank < 0 and os.environ.get('RANK'):
        args.local_rank = int(os.environ['RANK'])

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        logger.info(f'Initializing worker {args.local_rank}')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args)

    # here lay the "with IOContext" line, indented until the very end

    # Load pretrained model and tokenizer
    pretok = PreTokenizer(hashtml=args.hashtml)

    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training downloads model & vocab & data
        torch.distributed.barrier()

    temp_mod_dir = "/tmp/{}".format('shared_files')
    logger.info(f'temp directory: {temp_mod_dir}')

    # load pre-trained models
    btok, bmod, _ = load_model_files(args.model_type, args.model_dir, temp_mod_dir)
    if args.model_type == 'gpt2':
        add_pad_token(btok)
    add_vector_token(btok)
    bmod.resize_token_embeddings(len(btok))
    bmod.to(args.device)

    logger.info(f'{args.model_type} vocabulary size is {bmod.get_input_embeddings().num_embeddings}.')
    args.word_emb_dim = bmod.get_input_embeddings().embedding_dim
    logger.info(f'Embedding dimension is {args.word_emb_dim}.')

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training downloads model & vocab & data
        torch.distributed.barrier()

    # load dataset
    if args.local_rank in [-1, 0]:
        train_dataset = load_dataset(args.train_data_file, pretok, btok,
                                     args.line_by_line, args.block_size, args.train_data_portion,
                                     args.shuffle_data)
        logger.info(f'Loaded {len(train_dataset)} training examples.')
        write_dataset(train_dataset, os.path.join(temp_mod_dir, 'tr_ds.b'))

        # End of data barrier
        if args.local_rank == 0:
            torch.distributed.barrier()

    if args.local_rank not in [-1, 0]:
        train_dataset = load_cached_dataset(os.path.join(temp_mod_dir, 'tr_ds.b'))

    train_dataset.set_device(args.device)

    char_vocab = train_dataset.chars
    logger.info(f'Character vocab size:\t{len(char_vocab)}, saving to chars.txt')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'chars.txt'), mode='w') as outf:
        outf.write(''.join(char_vocab[:-len(SPECIAL_CHAR_LIST)]))

    # load vocab, init cycle trainers
    if args.lrn_tdt:
        vocab = load_vocab(args.vocab_file, args.vocab_size, lowercase=args.lowercase_vocab)

        tdcyc = TdCycleTrainer(args, vocab)
        dtcyc = DtCycleTrainer(args)
    else:
        tdcyc = None
        dtcyc = None
        args.train_cycle_dep = False

    if args.lrn_tdt:
        # init auxiliary modules
        lrn_rules = StochasticFilter(btok, args.lrn_prob)
        if args.stochastic_inference:
            inf_rules = StochasticFilter(btok, args.lrn_prob)
        else:
            inf_rules = AllMultiToksFilter(btok)
        if args.pool_policy == 'max':
            pooler = MaxPooler()
        elif args.pool_policy == 'avg':
            pooler = AvgPooler()
        elif args.pool_policy == 'first':
            pooler = FirstTokPooler()
        else:
            raise ValueError(f'Multi-token pooling policy not supported: {args.pool_policy}')

        # init TDT modules
        tdemb = TdtEmbedder(base_tok=btok, base_embs=bmod.get_input_embeddings().to(args.device),
                            char_vocab=char_vocab, char_emb_size=args.char_emb_size,
                            learn_filter_rules=lrn_rules, infer_filter_rules=inf_rules,
                            multitoken_pooler=pooler, args=args)
        gen = TdtGenerator(char_vocab=char_vocab,
                           emb_size=args.char_emb_size,
                           vec_size=bmod.get_output_embeddings().in_features,
                           args=args)
    else:
        tdemb = None
        gen = None
        args.alpha_vec = 0.0
        args.alpha_gen = 0.0
    if args.model_type in ['bert', 'cbert']:
        tdt = BertTdtWrapper(base_model=bmod,
                             pretokenizer=pretok,
                             base_tokenizer=btok,
                             tdt_embedder=tdemb,
                             tdt_generator=gen,
                             args=args)
    elif args.model_type == 'gpt2':
        tdt = GptTdtWrapper(base_model=bmod,
                            pretokenizer=pretok,
                            base_tokenizer=btok,
                            tdt_embedder=tdemb,
                            tdt_generator=gen,
                            args=args)
    elif args.model_type == 't5':
        tdt = T5TdtWrapper(base_model=bmod,
                           pretokenizer=pretok,
                           base_tokenizer=btok,
                           tdt_embedder=tdemb,
                           tdt_generator=gen,
                           args=args)
    elif args.model_type == 'roberta':
        tdt = RobertaTdtWrapper(base_model=bmod,
                                pretokenizer=pretok,
                                base_tokenizer=btok,
                                tdt_embedder=tdemb,
                                tdt_generator=gen,
                                args=args)

    logger.info(f'Initialized Tokdetok model.')

    # train
    train(args, train_dataset, tdt, tdcyc, dtcyc, char_vocab)

    # save model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    tdt.save(args.output_dir)
    logger.info(f"Saved model to {args.output_dir}")

    # evaluate
    evaluate(args, tdt)


if __name__ == "__main__":
    main()
