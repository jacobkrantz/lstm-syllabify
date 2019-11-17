# This script trains the BiLSTM-CRF architecture for syllabification.

import argparse
import os
import logging
import sys

from config import get_cfg_defaults
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
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Results directories
PATH = os.getcwd() + "/results/"


def create_directory(name):
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    if not os.path.exists(PATH + "/" + str(name)):
        os.mkdir(PATH + "/" + str(name))


def train_and_eval_model(cfg):
    """
    Load data and train model
    args:
        cfg (YACS YAML config)
    """

    # Data preprocessing
    dataset = {
        "columns": {0: "raw_tokens", 1: "boundaries"},
        # CoNLL format (tab-delineated)
        #   Column 0: phones
        #   Column 1: syllable boundary
        "label": "boundaries",  # Which column we like to predict
    }

    # Load the embeddings and the dataset. Choose whether or not to pad the words.
    # Right now, padding must be done if CRF is chosen for output layer.
    # The CRF layer does not support masking.
    embeddings, data, mappings, vocab_size, n_class_labels, word_length = load_dataset(
        dataset, dataset_name=cfg.TRAINING.DATASET, do_pad_words=True
    )

    create_directory(cfg.CONFIG_NAME)
    logger.info(f"Starting training of `{cfg.CONFIG_NAME}` on dataset `{dataset}`")

    for training_repeat in range(cfg.TRAINING.TRAINING_REPEATS):
        model = BiLSTM(cfg)
        model.set_vocab(vocab_size, n_class_labels, word_length, mappings)
        model.set_dataset(dataset, data)

        # Path to store performance scores for dev / test
        model.store_results(
            PATH + "/" + cfg.CONFIG_NAME + "/" + str(training_repeat) + ".csv"
        )
        model.fit(epochs=cfg.TRAINING.EPOCHS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/french_large_base.yaml",
        help="filename of config to run experiment with",
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    logging.info(cfg)
    train_and_eval_model(cfg)
