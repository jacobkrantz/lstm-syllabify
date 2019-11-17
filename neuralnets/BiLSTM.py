import numpy as np
import sys
import time
import os
import random
import logging

from .utils import try_tensorflow_import

try_tensorflow_import()

from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, Adadelta, Adagrad, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

from config import get_cfg_defaults
from .keraslayers.ChainCRF import ChainCRF


class BiLSTM:
    """
    A bidirectional LSTM with optional CRF for NLP sequence tagging.

    Author: Jacob Krantz
    Based on work done by Nils Reimers
    TODO: do Apache-2.0 properly

    https://github.com/tensorflow/tensorflow/issues/30263#issuecomment-509010526
    As of TF 2.0, we no longer need to selecte CudnnLSTM vs LSTM. Warning
    "Skipping optimization due to error while loading function libraries"
    can be ignored.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.model_save_path = cfg.TRAINING.MODEL_SAVE_PATH
        self.results_save_path = None

    def set_vocab(self, vocab_size, n_class_labels, word_length, mappings):
        # class labels are syllable boundary labels
        self.vocab_size = vocab_size
        self.n_class_labels = n_class_labels
        self.word_length = word_length
        self.mappings = mappings  # used indirectly during model reload

    def set_dataset(self, dataset, data):
        self.dataset = dataset
        self.data = data

        self.epoch = 0
        self.learning_rate_updates = {"sgd": {1: 0.1, 3: 0.05, 5: 0.01}}
        self.train_mini_batch_ranges = None
        self.train_word_length_ranges = None

        self.label_key = self.dataset["label"]

        logging.info("--- Dataset Details ---")
        logging.info("%d train words" % len(self.data["train_matrix"]))
        logging.info("%d dev words" % len(self.data["dev_matrix"]))
        logging.info("%d test words" % len(self.data["test_matrix"]))

    def build_model(self):
        if self.word_length <= 0:  # variable length words
            self.word_length = None

        tokens_input = Input(
            shape=(self.word_length,),  # use explicit word length for CNNs to work
            dtype="float32",
            name="phones_input",
        )

        # output shape: (batch_size, word_length, embedding size)
        tokens = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.cfg.MODEL.EMBEDDING_SIZE,
            trainable=True,
            name="phone_embeddings",
        )(tokens_input)

        # Add recurrent layers
        if self.cfg.MODEL.USE_RNN:
            assert self.cfg.MODEL.RNN in ["gru", "lstm"]
            rnn_func = GRU if self.cfg.MODEL.RNN == "gru" else LSTM

            recurrent_layer = Bidirectional(
                rnn_func(
                    units=self.cfg.MODEL.RNN_SIZE,
                    return_sequences=True,
                    dropout=self.cfg.MODEL.DROPOUT,
                    recurrent_dropout=self.cfg.MODEL.RECURRENT_DROPOUT,
                ),
                name="Bi" + self.cfg.MODEL.RNN,
            )(tokens)

        # Add CNNs, inspired by Ma and Hovy, 2016. In our case,
        #   the CNNs are parallel to LSTM instead of prior.
        # TODO: add RELU activation function.
        if self.cfg.MODEL.USE_CNN:
            cnn_layer = tokens
            # how to reshape::: re = Reshape((tokens.shape[1],tokens.shape[2],) + (1, ))(tokens) #  + (1, )

            for i in range(self.cfg.MODEL.CNN_LAYERS):
                cnn_layer = Conv1D(
                    filters=self.cfg.MODEL.NUM_FILTERS,
                    kernel_size=self.cfg.MODEL.FILTER_SIZE,
                    padding="same",
                    name="cnn_" + str(i + 1),
                )(cnn_layer)

                if self.cfg.MODEL.MAX_POOL_SIZE:
                    # maintain dimensionality (stride = 1)
                    cnn_layer = MaxPooling1D(
                        pool_size=self.cfg.MODEL.MAX_POOL_SIZE,
                        strides=1,
                        padding="same",
                        name="max_pooling_" + str(i + 1),
                    )(cnn_layer)

            # concatenating the CNN with the LSTM essentially tacks on the cnn vector to the end of each lstm time-step vector.
            if self.cfg.MODEL.USE_RNN:
                concat_layer = concatenate([recurrent_layer, cnn_layer])
            else:
                concat_layer = cnn_layer
        else:
            assert self.cfg.MODEL.USE_RNN, "Either RNN or CNN must be in network."
            concat_layer = recurrent_layer

        # Add output classifier
        output = concat_layer
        assert self.cfg.MODEL.CLASSIFIER in [
            "softmax",
            "crf",
        ], "classifier must be either 'softmax' or 'crf'"
        if self.cfg.MODEL.CLASSIFIER == "softmax":
            output = TimeDistributed(
                Dense(units=self.n_class_labels, activation="softmax"), name="softmax"
            )(output)
            loss_function = "sparse_categorical_crossentropy"

        elif self.cfg.MODEL.CLASSIFIER == "crf":  # use Philipp Gross' ChainCRF
            output = TimeDistributed(
                Dense(units=self.n_class_labels, activation=None),
                name="hidden_lin_layer",
            )(output)
            crf = ChainCRF(name="crf")
            output = crf(output)
            loss_function = crf.sparse_loss

        # :: Parameters for the optimizer ::
        optim_params = {}
        if self.cfg.OPTIMIZER.CLIP_NORM > 0.0:
            optim_params["clipnorm"] = self.cfg.OPTIMIZER.CLIP_NORM
        if self.cfg.OPTIMIZER.CLIP_VALUE > 0:
            optim_params["clipvalue"] = self.cfg.OPTIMIZER.CLIP_VALUE

        optimizer = self.cfg.OPTIMIZER.OPTIMIZER
        if optimizer == "adam":
            opt = Adam(**optim_params)
        elif optimizer == "nadam":
            opt = Nadam(**optim_params)
        elif optimizer == "rmsprop":
            opt = RMSprop(**optim_params)
        elif optimizer == "adadelta":
            opt = Adadelta(**optim_params)
        elif optimizer == "adagrad":
            opt = Adagrad(**optim_params)
        elif optimizer == "sgd":
            opt = SGD(lr=0.1, **optim_params)
        else:
            assert False, "Optmizer not in list of allowable optimizers"

        model = Model(inputs=[tokens_input], outputs=[output])
        model.compile(loss=loss_function, optimizer=opt)
        model.summary(line_length=100)
        self.model = model

    def train_model(self):
        self.epoch += 1

        if (
            self.cfg.OPTIMIZER.OPTIMIZER in self.learning_rate_updates
            and self.epoch in self.learning_rate_updates[self.cfg.OPTIMIZER.OPTIMIZER]
        ):
            lr_update = self.learning_rate_updates[self.cfg.OPTIMIZER.OPTIMIZER][
                self.epoch
            ]
            logging.info("Update Learning Rate to %f" % (lr_update))
            K.set_value(self.model.optimizer.lr, lr_update)

        loss = 0.0
        for batch in self.minibatch_iterate_dataset():
            loss += self.model.train_on_batch(x=batch[1:], y=batch[0])
        return loss

    def minibatch_iterate_dataset(self):
        """
        Create based on word length mini-batches with approx. the same size. 
        Words and mini-batch chunks are shuffled and used to the train the model
        """
        if self.train_word_length_ranges == None:
            """ Create mini batch ranges """
            self.train_word_length_ranges = {}
            self.train_mini_batch_ranges = {}

            train_data = self.data["train_matrix"]
            train_data.sort(
                key=lambda x: len(x["tokens"])
            )  # Sort train data by word length
            train_ranges = []
            old_word_len = len(train_data[0]["tokens"])
            idxStart = 0

            # Find start and end of ranges with words with same length
            for idx in range(len(train_data)):
                word_len = len(train_data[idx]["tokens"])

                if word_len != old_word_len:
                    train_ranges.append((idxStart, idx))
                    idxStart = idx

                old_word_len = word_len

            # Add last word
            train_ranges.append((idxStart, len(train_data)))

            # Break up ranges into smaller mini batch sizes
            mini_batch_ranges = []
            for batch_range in train_ranges:
                range_len = batch_range[1] - batch_range[0]

                bins = int(
                    np.ceil(range_len / float(self.cfg.TRAINING.MINI_BATCH_SIZE))
                )
                bin_size = int(np.ceil(range_len / float(bins)))

                for bin_nr in range(bins):
                    startIdx = bin_nr * bin_size + batch_range[0]
                    endIdx = min(
                        batch_range[1], (bin_nr + 1) * bin_size + batch_range[0]
                    )
                    mini_batch_ranges.append((startIdx, endIdx))

                self.train_word_length_ranges = train_ranges
                self.train_mini_batch_ranges = mini_batch_ranges

        # Shuffle training data
        # 1. Shuffle words that have the same length
        x = self.data["train_matrix"]
        for data_range in self.train_word_length_ranges:
            for i in reversed(range(data_range[0] + 1, data_range[1])):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = random.randint(data_range[0], i)
                x[i], x[j] = x[j], x[i]

        # 2. Shuffle the order of the mini batch ranges
        random.shuffle(self.train_mini_batch_ranges)

        # Iterate over the mini batch ranges
        range_length = len(self.train_mini_batch_ranges)

        batches = {}
        for idx in range(range_length):
            batches.clear()
            train_data = self.data["train_matrix"]
            data_range = self.train_mini_batch_ranges[
                idx % len(self.train_mini_batch_ranges)
            ]
            labels = np.asarray(
                [
                    train_data[idx][self.label_key]
                    for idx in range(data_range[0], data_range[1])
                ]
            )
            labels = np.expand_dims(labels, -1)

            batches = [labels]

            for featureName in self.cfg.TRAINING.FEATURE_NAMES:
                inputData = np.asarray(
                    [
                        train_data[idx][featureName]
                        for idx in range(data_range[0], data_range[1])
                    ]
                )
                batches.append(inputData)

            yield batches

    def fit(self, epochs):
        if self.model is None:
            self.build_model()

        train_time_total = 0
        eval_time_total = 0
        max_dev_score = 0
        max_test_score = 0
        no_improvement_since = 0

        for epoch in range(epochs):
            sys.stdout.flush()
            logging.info("\n--------- Epoch %d -----------" % (epoch + 1))

            start_time_epoch = time.time()
            self.train_model()
            train_time_epoch = time.time() - start_time_epoch
            train_time_total += train_time_epoch
            logging.info(
                "%.2f sec for training (%.2f total)"
                % (train_time_epoch, train_time_total)
            )

            start_time_eval = time.time()
            dev_score, test_score = self.compute_acc_scores(
                self.data["dev_matrix"], self.data["test_matrix"]
            )

            if dev_score > max_dev_score or epoch == 1:
                max_dev_score = dev_score
                max_test_score = test_score
                no_improvement_since = 0
                self.save_model(epoch, dev_score, test_score)
            else:
                no_improvement_since += 1

            eval_time_epoch = time.time() - start_time_eval
            eval_time_total += eval_time_epoch
            logging.info(
                "\nScores from epoch with best dev-scores:\n  Dev-Score: %.4f\n  Test-Score %.4f"
                % (max_dev_score, max_test_score)
            )
            logging.info(
                "%.2f sec for eval (%.2f total)" % (eval_time_epoch, eval_time_total)
            )

            if self.results_save_path != None:
                self.results_save_path.write(
                    "\t".join(
                        map(
                            str,
                            [
                                epoch + 1,
                                dev_score,
                                test_score,
                                max_dev_score,
                                max_test_score,
                                train_time_epoch,
                                train_time_total,
                                eval_time_epoch,
                            ],
                        )
                    )
                )
                self.results_save_path.write("\n")
                self.results_save_path.flush()

            if (
                self.cfg.TRAINING.EARLY_STOPPING > 0
                and no_improvement_since >= self.cfg.TRAINING.EARLY_STOPPING
            ):
                logging.info(
                    "!!! Early stopping, no improvement after "
                    + str(no_improvement_since)
                    + " epochs !!!"
                )
                break

    def tagWords(self, words):
        """
        words: [{
                    'raw_tokens': ['S', 'V', 't', 'P', 'd'],
                    'tokens': [11, 5, 43, 36, 8]
                }, ...]
        """
        padded_pred_labels = self.predict_labels(self.model, words)
        pred_labels = []
        for idx in range(len(words)):
            unpadded_pred_labels = []
            for tokenIdx in range(len(words[idx]["tokens"])):
                # Skip padding tokens
                if words[idx]["tokens"][tokenIdx] != 0:
                    unpadded_pred_labels.append(padded_pred_labels[idx][tokenIdx])

            pred_labels.append(unpadded_pred_labels)

        return pred_labels

    def get_word_lengths(self, words):
        word_lengths = {}
        for idx in range(len(words)):
            word = words[idx]["tokens"]
            if len(word) not in word_lengths:
                word_lengths[len(word)] = []
            word_lengths[len(word)].append(idx)

        return word_lengths

    def predict_labels(self, model, words):
        pred_labels = [None] * len(words)
        word_lengths = self.get_word_lengths(words)

        for indices in word_lengths.values():
            nn_input = []
            for feature_name in self.cfg.TRAINING.FEATURE_NAMES:
                input_data = np.asarray([words[idx][feature_name] for idx in indices])
                nn_input.append(input_data)

            predictions = model.predict(nn_input, verbose=False)
            predictions = predictions.argmax(axis=-1)  # Predict classes

            predIdx = 0
            for idx in indices:
                pred_labels[idx] = predictions[predIdx]
                predIdx += 1

        return pred_labels

    def compute_acc_scores(self, dev_data, test_data):
        """
        Accuracy scores are reported at the word level. This means that if a single
        syllable boundary was incorrectly placed, the entire word is marked incorrect.
        
        Logs the boundary level accuracy as well.
        """
        dev_acc, dev_bound = self.compute_acc(dev_data)
        test_acc, test_bound = self.compute_acc(test_data)

        logging.info("-- Epoch Accuracies --")
        logging.info("Word-Level Accuracy")
        logging.info("Dev: %.4f" % (dev_acc))
        logging.info("Test: %.4f" % (test_acc))

        logging.info("\nBoundary-Level Accuracy")
        logging.info("Dev: %.4f" % (dev_bound))
        logging.info("Test: %.4f" % (test_bound))

        return dev_acc, test_acc

    def compute_acc(self, words):
        """
        Returns:
            float: word level accuracy. Range: [0.,1.]
            float: boundary_level_acc. Range: [0.,1.]
        """
        correct_labels = [words[idx][self.label_key] for idx in range(len(words))]
        pred_labels = self.predict_labels(self.model, words)

        num_labels = 0
        num_corr_labels = 0
        num_corr_words = 0

        for word_id in range(len(correct_labels)):
            word_was_wrong = False
            for tokenId in range(len(correct_labels[word_id])):
                num_labels += 1
                if correct_labels[word_id][tokenId] == pred_labels[word_id][tokenId]:
                    num_corr_labels += 1
                else:
                    word_was_wrong = True

            if not word_was_wrong:
                num_corr_words += 1

        boundary_level_acc = num_corr_labels / float(num_labels)
        word_level_acc = num_corr_words / len(words)
        return word_level_acc, boundary_level_acc

    def store_results(self, results_path):
        if results_path != None:
            directory = os.path.dirname(results_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            self.results_save_path = open(results_path, "w")
        else:
            self.results_save_path = None

    def save_model(self, epoch, dev_score, test_score):
        import json
        import h5py

        save_path = (
            self.model_save_path.replace("[DATASET]", self.cfg.TRAINING.DATASET)
            .replace("[DevScore]", "%.4f" % dev_score)
            .replace("[TestScore]", "%.4f" % test_score)
            .replace("[Epoch]", str(epoch + 1))
        )

        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.isfile(save_path):
            logging.info(f"Model {save_path} already exists. Model will be overwritten")

        res = keras.models.save_model(
            model=self.model, filepath=save_path, overwrite=True, save_format="h5"
        )

        with h5py.File(save_path, "a") as h5file:
            h5file.attrs["mappings"] = json.dumps(self.mappings)
            h5file.attrs["label_key"] = self.dataset["label"]
            h5file.attrs["vocab_size"] = self.vocab_size
            h5file.attrs["n_class_labels"] = self.n_class_labels
            h5file.attrs["word_length"] = (
                self.word_length if self.word_length != None else -1
            )

    @staticmethod
    def load_model(model_path, cfg_path):
        import h5py
        import json

        cfg = get_cfg_defaults()
        cfg.merge_from_file(cfg_path)
        cfg.freeze()
        logging.info(cfg)

        with h5py.File(model_path, "r") as f:
            mappings = json.loads(f.attrs["mappings"])
            label_key = f.attrs["label_key"]
            vocab_size = f.attrs["vocab_size"]
            n_class_labels = f.attrs["n_class_labels"]
            word_length = f.attrs["word_length"]

        if cfg.MODEL.CLASSIFIER == ["crf"]:
            from .keraslayers.ChainCRF import create_custom_objects

            custom_objects = create_custom_objects()

        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        bilstm = BiLSTM(cfg)
        bilstm.set_vocab(vocab_size, n_class_labels, word_length, mappings)
        bilstm.model = model
        bilstm.label_key = label_key
        return bilstm
