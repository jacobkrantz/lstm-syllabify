# This script trains the BiLSTM-CRF architecture for syllabification.

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
embeddings, data, mappings, vocab_size, n_class_labels, word_length = load_dataset(datasets, do_pad_words=True, embedding_size=50) # embedding_size in [50, 100, 200]
# sys.exit()
"""
EMBEDDINGS (GloVe)
    - numpy.ndarray holding n dimensional embeddings (each numpy.ndarray). Not normalized.
    - embeddings are for the 'tokens' input. word (raw_tokens) -> tokens -> embedding

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
PATH = os.getcwd() + '/results/hyper_parameters'
def create_directory(size):
        if(os.path.exists(PATH + "/" + str(size))):
                return
        # Make sure the directory doesn't already exist.
        os.mkdir(PATH + "/" + str(size))

# run_lst: [name, cnn layers, cnn number filters, cnn filter size, cnn max pool size, dropout, lstm size, embedding size]
# run_lst = [
#     ['1-test-layer',        6,  40, 3, 2, 0.25, 100, 100],
#     ['2-test-filter',       2, 100, 3, 2, 0.25, 100, 100],
#     ['3-test-layer-filter', 6, 100, 3, 2, 0.25, 100, 100],
#     ['4-test-filter-size',  2,  40, 2, 2, 0.25, 100, 100]
# ]

# run_lst = [
#     [ '9', 2, 100, 2,    4, 0.25, 100, 100],
#     ['10', 2, 200, 3,    2, 0.25, 100, 100],
#     ['11', 2, 200, 2,    2, 0.25, 100, 100],
#     ['12', 1,  40, 3,    2, 0.25, 100, 100],
#     ['13', 1, 100, 3,    2, 0.25, 100, 100],
#     ['14', 1, 200, 3,    2, 0.25, 100, 100],
#     ['15', 2, 200, 2,    4, 0.25, 100, 100],
#     ['16', 2, 100, 2, None, 0.25, 100, 100],
#     ['17', 2, 200, 2, None, 0.25, 100, 100],
#     ['18', 2, 100, 3, None, 0.25, 100, 100]
# ]

run_lst = [
    ['19', 1, 200, 3, 2, 0.25, 200, 100],
    ['20', 1, 200, 3, 2, 0.25, 100, 200],
    ['21', 1, 200, 3, 2, 0.25, 200, 200],
    ['22', 2, 200, 3, 2, 0.25, 200, 100],
    ['23', 2, 200, 3, 2, 0.25, 100, 200],
    ['24', 2, 200, 3, 2, 0.25, 200, 200]
]

# for run in run_lst:
#     create_directory(run[0])
#     print("Entering run: ", run[0])

#     for iteration in range(0,21): # Run one extra test for an outlier to be removed.
#         file_path = PATH + "/" + str(run[0]) + "/" + str(iteration) + '.csv'
#         print("Run with updated parameters: ", run)
#         params_to_update = {
#             # LSTM related
#             'which_rnn': 'LSTM', # either 'LSTM' or 'GRU'
#             'lstm_size': run[6],
#             'dropout': 0.25, # (0.25, 0.25), # tuple dropout is for recurrent dropout and cannot work with GPU computation.
#             # CNN related
#             'use_cnn': True,
#             'cnn_layers': run[1],
#             'cnn_num_filters': run[2],
#             'cnn_filter_size': run[3],
#             'cnn_max_pool_size': run[4], # if None or False, do not use MaxPooling

#             # CRF related
#             'classifier': 'crf', # either 'softmax', 'kc-crf' (from keras-contrib) or 'crf' (by Philipp Gross).
#             'crf_activation': 'linear', # Only for kc-crf. Possible values: 'linear' (default), 'relu', 'tanh', 'softmax', others. See Keras Activations.

#             # general params
#             'mini_batch_size': 64,
#             'using_gpu': True,
#             'embedding_size': run[7],
#             'use_pretrained_embeddings': True
#             'early_stopping': 10
#         }

#         model = BiLSTM(params_to_update)
#         model.set_vocab_size(vocab_size, n_class_labels, word_length, mappings)
#         model.set_dataset(datasets, data)
#         model.set_embeddings(embeddings)
#         model.store_results(file_path) # Path to store performance scores for dev / test
#         model.model_save_path = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" # Path to store models
#         model.fit(epochs = 120)
#         os.system("rm /home/ubuntu/gulp/lstm-syllabify/models/* -rf")














file_path =  'results/glove-test.csv'
params_to_update = {
    # LSTM related
    'which_rnn': 'LSTM', # either 'LSTM' or 'GRU'
    'lstm_size': 100,
    'dropout': 0.25, # (0.25, 0.25), # tuple dropout is for recurrent dropout and cannot work with GPU computation.
    # CNN related
    'use_cnn': False,
    'cnn_layers': 2,
    'cnn_num_filters': 100,
    'cnn_filter_size': 2,
    'cnn_max_pool_size': 2, # if None or False, do not use MaxPooling

    # CRF related
    'classifier': 'crf', # either 'softmax', 'kc-crf' (from keras-contrib) or 'crf' (by Philipp Gross).
    'crf_activation': 'linear', # Only for kc-crf. Possible values: 'linear' (default), 'relu', 'tanh', 'softmax', others. See Keras Activations.

    # general params
    'mini_batch_size': 64,
    'using_gpu': False,
    'embedding_size': 50,
    'use_pretrained_embeddings': True,
    'early_stopping': 10
}

model = BiLSTM(params_to_update)
model.set_vocab_size(vocab_size, n_class_labels, word_length, mappings)
model.set_dataset(datasets, data)
model.set_embeddings(embeddings)
model.store_results(file_path) # Path to store performance scores for dev / test
model.model_save_path = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" # Path to store models
model.fit(epochs = 120)
