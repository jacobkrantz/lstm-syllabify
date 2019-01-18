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

# Load the embeddings and the dataset.
# Once we have this training, we could store these in a pkl file for speed.
embeddings, data, mappings, vocab_size, n_class_labels = load_dataset(datasets)

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

params_to_update = {
    'classifier': ['softmax'], # either 'softmax' or 'crf'. crf is not ready yet.
    'lstm_size': [100],
    'dropout': (0.25, 0.25),
    'embedding_size': 100
}

model = BiLSTM(params_to_update)
model.set_vocab_size(vocab_size, n_class_labels, mappings)
model.set_dataset(datasets, data)
model.store_results('results/english.csv') # Path to store performance scores for dev / test
model.model_save_path = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" # Path to store models
model.fit(epochs=40)
