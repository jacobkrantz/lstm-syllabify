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
PATH = os.getcwd() + '/results/'
def create_directory(name):
    if(not os.path.exists(PATH)):
        os.mkdir(PATH)
    if(not os.path.exists(PATH + "/" + str(name))):
        os.mkdir(PATH + "/" + str(name))

def train_and_eval_model(run_params, iterations=21):
    """
    Load data and train model
    args:
        run_params: list containing: [
                        0  name,
                        1  use_cnn,
                        2  use_lstm,
                        3  cnn_layers,
                        4  cnn_number_filters,
                        5  cnn_filter_size,
                        6  cnn_max_pool_size,
                        7  lstm_size,
                        8  embedding_size,
                        9  mini_batch_size,
                        10 which_rnn,
                        11 classifier,
                        12 language
                    ]
        iterations: run one extra test for an outlier to be removed.
    """

    # Data preprocessing
    datasets = {
        run_params[12]:                                             # Name of the dataset. Same as folder name in /data/
            {
                'columns': {0:'raw_tokens', 1:'boundaries'},  # CoNLL format for the input data (tab-delineated). Column 0 contains phones, column 1 contains syllable boundary information
                'label': 'boundaries'                         # Which column we like to predict
            }
    }

    # Load the embeddings and the dataset. Choose whether or not to pad the words.
    embeddings, data, mappings, vocab_size, n_class_labels, word_length = load_dataset(datasets, do_pad_words=True)

    create_directory(run_params[0])
    print("Entering run: ", run_params[0])

    for iteration in range(0,iterations):
        file_path = PATH + "/" + str(run_params[0]) + "/" + str(iteration) + '.csv'
        print("Run with updated parameters: ", run_params)
        params_to_update = {
            # LSTM related
            'use_lstm': run_params[2], # True or False
            'which_rnn': run_params[10], # either 'LSTM' or 'GRU'
            'lstm_size': run_params[7],

            # CNN related
            'use_cnn': run_params[1], # True or False
            'cnn_layers': run_params[3],
            'cnn_num_filters': run_params[4],
            'cnn_filter_size': run_params[5],
            'cnn_max_pool_size': run_params[6], # if None or False, do not use MaxPooling

            # CRF related
            'classifier': run_params[11], # either 'softmax', 'kc-crf' (from keras-contrib) or 'crf' (by Philipp Gross).

            # general params
            'mini_batch_size': run_params[9],
            'using_gpu': True,
            'embedding_size': run_params[8]
        }

        model = BiLSTM(params_to_update)
        model.set_vocab_size(vocab_size, n_class_labels, word_length, mappings)
        model.set_dataset(datasets, data)
        model.store_results(file_path) # Path to store performance scores for dev / test
        model.model_save_path = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" # Path to store models
        model.fit(epochs = 120)
        os.system("rm /home/ubuntu/gulp/lstm-syllabify/models/* -rf")

final_params_large = ['Base', True, True, 2, 200, 3, 2, 300, 300, 64, 'LSTM', 'crf', 'english']
final_params_small = ['small-3', True, True, 1, 40, 3, 2,  50, 100, 64, 'LSTM', 'crf', 'english'] # eqivalent to small-3

dataset_names = [
    'english',
    'italian',
    'basque',
    'dutch',
    'manipuri',
    'french'
]

for dataset in dataset_names:
    if dataset == 'french':
        final_params_large[-1] = dataset
        train_and_eval_model(final_params_large)
