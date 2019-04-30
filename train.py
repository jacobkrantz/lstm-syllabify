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

# Results directories
PATH = os.getcwd() + '/results/m_experiments'
def create_directory(name):
    if(not os.path.exists(PATH)):
        os.mkdir(PATH)
    if(not os.path.exists(PATH + "/" + str(name))):
        os.mkdir(PATH + "/" + str(name))

cnn_optimization_runs = [
    # [name, use_cnn, use_lstm, cnn layers, cnn number filters, cnn filter size, cnn max pool size, dropout, lstm size, embedding size, mini_batch_size, which_rnn, classifier, language]
    [ '1', True, True, 6,  40, 3,    2, 100, 100, 64, 'LSTM', 'crf', 'english'],
    [ '2', True, True, 2, 100, 3,    2, 100, 100, 64, 'LSTM', 'crf', 'english'],
    [ '3', True, True, 6, 100, 3,    2, 100, 100, 64, 'LSTM', 'crf', 'english'],
    [ '4', True, True, 2,  40, 2,    2, 100, 100, 64, 'LSTM', 'crf', 'english'],
    [ '9', True, True, 2, 100, 2,    4, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['10', True, True, 2, 200, 3,    2, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['11', True, True, 2, 200, 2,    2, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['12', True, True, 1,  40, 3,    2, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['13', True, True, 1, 100, 3,    2, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['14', True, True, 1, 200, 3,    2, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['15', True, True, 2, 200, 2,    4, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['16', True, True, 2, 100, 2, None, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['17', True, True, 2, 200, 2, None, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['18', True, True, 2, 100, 3, None, 100, 100, 64, 'LSTM', 'crf', 'english'],
    ['19', True, True, 1, 200, 3,    2, 200, 100, 64, 'LSTM', 'crf', 'english'],
    ['20', True, True, 1, 200, 3,    2, 100, 200, 64, 'LSTM', 'crf', 'english'],
    ['21', True, True, 1, 200, 3,    2, 200, 200, 64, 'LSTM', 'crf', 'english'],
    ['22', True, True, 2, 200, 3,    2, 200, 100, 64, 'LSTM', 'crf', 'english'],
    ['23', True, True, 2, 200, 3,    2, 100, 200, 64, 'LSTM', 'crf', 'english'],
    ['24', True, True, 2, 200, 3,    2, 200, 200, 64, 'LSTM', 'crf', 'english'],
    ['25', True, True, 1, 200, 3,    2, 300, 300, 64, 'LSTM', 'crf', 'english'],
    ['26', True, True, 2, 200, 3,    2, 300, 300, 64, 'LSTM', 'crf', 'english'],
    ['27', True, True, 1, 200, 3,    2, 200, 200, 16, 'LSTM', 'crf', 'english'],
    ['28', True, True, 2, 200, 3,    2, 300, 300, 16, 'LSTM', 'crf', 'english'],
    ['29', True, True, 1, 200, 3,    2, 300, 300, 16, 'LSTM', 'crf', 'english'],
    ['30', True, True, 2, 200, 3,    2, 300, 300, 16, 'LSTM', 'crf', 'english']
]

m_experiment_runs = [
    # [name, use_cnn, use_lstm, cnn layers, cnn number filters, cnn filter size, cnn max pool size, lstm size, embedding size, mini_batch_size, which_rnn, classifier, language]
    # ['Base', True, True, 2, 200, 3, 2, 300, 300, 64, 'LSTM',     'crf',       'english'], # same as cnn_optimization_runs #26.
    [  'M1', True,   True, 2, 200, 3, 2, 300, 300, 64,  'GRU',     'crf',      'english' ],
    [  'M2', False,  True, 2, 200, 3, 2, 300, 300, 64, 'LSTM',     'crf',      'english' ],
    [  'M3', False,  True, 2, 200, 3, 2, 300, 300, 64,  'GRU',     'crf',      'english' ],
    [  'M4', True,   True, 2, 200, 3, 2, 300, 300, 64, 'LSTM', 'softmax',      'english' ],
    [  'M5', True,  False, 2, 200, 3, 2, 300, 300, 64, 'LSTM',     'crf',      'english' ],
    [  'M6', True,   True, 2, 200, 3, 2, 300, 300, 64, 'LSTM',     'crf',      'italian' ],
    [  'M7', True,   True, 2, 200, 3, 2, 300, 300, 64, 'LSTM',     'crf',       'basque' ],
    [  'M8', True,  False, 2, 200, 3, 2, 300, 300, 64, 'LSTM',     'crf',      'NETtalk' ],
    [  'M9', True,  False, 2, 200, 3, 2, 300, 300, 16, 'LSTM',     'crf',      'NETtalk' ],
    [ 'M10', True,  False, 2, 200, 3, 2, 300, 300, 64, 'LSTM',     'crf', 'NETtalkTrain' ],
    [ 'M11', True,  False, 2, 200, 3, 2, 300, 300, 16, 'LSTM',     'crf', 'NETtalkTrain' ],
    [ 'M12', True,  False, 2, 200, 3, 2, 300, 300, 32, 'LSTM',     'crf', 'NETtalkTrain' ],
    [ 'M13', True,  False, 2, 200, 3, 2, 300, 300, 64, 'LSTM',     'crf',        'dutch' ],
    [ 'M14', True,   True, 2, 200, 3, 2, 300, 300, 64, 'LSTM',     'crf',     'manipuri' ]
]

# just do the last for now
m_experiment_runs = [m_experiment_runs[-1]]

# ---------------------------
#  Load data and train model
# ---------------------------

for run in m_experiment_runs:
    # Data preprocessing
    datasets = {
        run[12]:                                             # Name of the dataset. Same as folder name in /celex-data/
            {
                'columns': {0:'raw_tokens', 1:'boundaries'},  # CoNLL format for the input data (tab-delineated). Column 0 contains phones, column 1 contains syllable boundary information
                'label': 'boundaries'                         # Which column we like to predict
            }
    }

    # Load the embeddings and the dataset. Choose whether or not to pad the words.
    embeddings, data, mappings, vocab_size, n_class_labels, word_length = load_dataset(datasets, do_pad_words=True)

    create_directory(run[0])
    print("Entering run: ", run[0])

    for iteration in range(0,21): # Run one extra test for an outlier to be removed.
        file_path = PATH + "/" + str(run[0]) + "/" + str(iteration) + '.csv'
        print("Run with updated parameters: ", run)
        params_to_update = {
            # LSTM related
            'use_lstm': run[2], # True or False
            'which_rnn': run[10], # either 'LSTM' or 'GRU'
            'lstm_size': run[7],

            # CNN related
            'use_cnn': run[1], # True or False
            'cnn_layers': run[3],
            'cnn_num_filters': run[4],
            'cnn_filter_size': run[5],
            'cnn_max_pool_size': run[6], # if None or False, do not use MaxPooling

            # CRF related
            'classifier': run[11], # either 'softmax', 'kc-crf' (from keras-contrib) or 'crf' (by Philipp Gross).

            # general params
            'mini_batch_size': run[9],
            'using_gpu': False,
            'embedding_size': run[8]
        }

        model = BiLSTM(params_to_update)
        model.set_vocab_size(vocab_size, n_class_labels, word_length, mappings)
        model.set_dataset(datasets, data)
        model.store_results(file_path) # Path to store performance scores for dev / test
        model.model_save_path = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" # Path to store models
        model.fit(epochs = 120)
        os.system("rm /home/ubuntu/gulp/lstm-syllabify/models/* -rf")
