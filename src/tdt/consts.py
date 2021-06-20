# special characters and keys

BOUND_CHAR = '<e>'
UNK_CHAR = '<unk>'
PAD_CHAR = '<pad>'
SPECIAL_CHAR_LIST = [UNK_CHAR, BOUND_CHAR, PAD_CHAR]

VEC_TOKEN = '[VECTORIZED]'
MASK_TOKEN_KEY = 'mask_token'

# default values

IGNORE_INDEX = -100

# tokenization properties

BERT_WORDMID = '##'
GPT_SPACE = 'Ġ'
EASY_SUFFIXES = ['s', 'ed', 'es', 'ing', 'ly', 'al', 'ally', "'m", "'re", "'ve", 'y', 'ive', 'er',
                 "'t", "'ll", 'an', 'ers']

# default params

CHAR_EMB_SIZE = 50
LSTM_HIDDEN_SIZE = 128
GEN_LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
CONV_HIDDEN_SIZE = 128
TRANSFORMER_HIDDEN_SIZE = 256
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_NUM_LAYERS = 6
MAX_IN_WORDLEN = 32

GEN_LSTM_NUM_LAYERS = 2
MLP_NUM_LAYERS = 2
MAX_WORDLEN = 20

BLOCK_SIZE = 512

