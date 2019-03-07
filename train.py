# This script trains the BiLSTM-CRF architecture for syllabification using
# the CELEX English dataset.
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from preprocessing import load_dataset

# Change into the working dir of the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Logging level
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Data preprocessing
datasets = {
    'english':                                            # Name of the dataset. Same as folder name in /celex-data/
        {
            'columns': {0:'raw_tokens', 1:'boundaries'},  # CoNLL format for the input data (tab-delineated). Column 0 contains phones, column 1 contains syllable boundary information
            'label': 'boundaries'                         # Which column we like to predict
        }
}

# Load the embeddings and the dataset. Choose whether or not to pad the words.
embeddings, data, mappings, vocab_size, n_class_labels, word_length = load_dataset(datasets, do_pad_words=True)

"""
EMBEDDINGS (not used)
    - numpy.ndarray holding 300 dimensional embeddings (each numpy.ndarray) that are not normalized to 0-1.
    - there is not an explicit mapping built into the structure, so they must be associated with the mappings data structure
    - embeddings are for the word inputs. word (raw_tokens) -> tokens -> embedding

DATA
    - raw_tokens are phones in DISC format
    shape:
    data = {
        'english': {
            'train_matrix': [
                {
                    'tokens': [int, int, ... , int],
                    'boundaries': [int, int, ... , int],
                    'raw_tokens':[str, str, ..., str]
                }, ...
            ]
            'dev_matrix': same as train_matrix
            'test_matrix': same as train_matrix
        }
    }

MAPPINGS
    - dictionary that maps tokens to a unique integer

VOCAB_SIZE
    - number of possible inputs to the NN.
    - Usually is the number of phones in the langage being used.

N_CLASS_LABELS
    - number of possible types of syllable boundaries. 
    - Default is two: either boundary (1) or no boundary (0)
"""

# parameters 'use_cnn', 'cnn_filter_size', and 'cnn_filter_length' don't do anything yet. CNN development is WIP.
params_to_update = {
    # LSTM related
    'which_rnn': 'LSTM', # either 'LSTM' or 'GRU'
    'lstm_size': [100],
    'dropout': (0.25, 0.25),

    # CNN related
    'use_cnn': True,
    'cnn_layers': 2,
    'cnn_num_filters': 20,
    'cnn_filter_size': 3,
    'cnn_max_pool_size': 2, # if None or False, do not use MaxPooling

    # CRF related
    'classifier': 'crf', # either 'softmax', 'kc-crf' (from keras-contrib) or 'crf' (by Philipp Gross).
    'crf_activation': 'linear', # Only for kc-crf. Possible values: 'linear' (default), 'relu', 'tanh', 'softmax', others. See Keras Activations.

    # general params
    'embedding_size': 100,
    'early_stopping': 7
}

model = BiLSTM(params_to_update)
model.set_vocab_size(vocab_size, n_class_labels, word_length, mappings)
model.set_dataset(datasets, data)
model.store_results('results/english.csv') # Path to store performance scores for dev / test
model.model_save_path = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" # Path to store models
model.fit(epochs = 50)
