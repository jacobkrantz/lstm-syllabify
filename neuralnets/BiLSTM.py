
import math
import numpy as np
import sys
import time
import os
import random
import logging

import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from .keraslayers.ChainCRF import ChainCRF

# from keras_contrib.layers import CRF
# from keras_contrib.losses import crf_loss
import keras.backend as K
import tensorflow as tf

class BiLSTM:
    """
    A bidirectional LSTM with optional CRF for NLP sequence tagging.

    Author: Jacob Krantz
    Based on work done by Nils Reimers
    TODO: do Apache-2.0 properly
    """

    def __init__(self, params=None):
        self.models = None
        self.model_save_path = None # Path for storing models
        self.results_save_path = None # Path for storing output labels while training

        # Hyperparameters for the network
        default_params = {
            'dropout': (0.5,0.5),
            'classifier': ['Softmax'],
            'which_rnn': 'GRU', # either 'LSTM' or 'GRU'
            'lstm_size': (100,),
            'use_cnn': False,
            'cnn_layers': 2,
            'cnn_num_filters': 20,
            'cnn_filter_size': 3,
            'cnn_max_pool_size': 2, # if None or False, do not use MaxPooling
            'optimizer': 'adam',
            'clipvalue': 0,
            'clipnorm': 1,
            'early_stopping': 5,
            'mini_batch_size': 32,
            'feature_names': ['tokens'],
            'embedding_size': 50,
            'crf_activation':'linear',
            'using_gpu': True
        }
        if params != None:
            default_params.update(params)
        self.params = default_params


    def set_vocab_size(self, vocab_size, n_class_labels, word_length, mappings):
        # class labels are syllable boundary labels
        self.vocab_size = vocab_size
        self.n_class_labels = n_class_labels
        self.word_length = word_length
        self.mappings = mappings # used indirectly during model reload


    def set_dataset(self, datasets, data):
        self.datasets = datasets
        self.data = data

        # Create some helping variables
        self.main_model_name = None
        self.epoch = 0
        self.learning_rate_updates = {'sgd': {1: 0.1, 3: 0.05, 5: 0.01}}
        self.model_names = list(self.datasets.keys())
        self.label_keys = {}
        self.train_mini_batch_ranges = None
        self.train_word_length_ranges = None

        for model_name in self.model_names:
            label_key = self.datasets[model_name]['label']
            self.label_keys[model_name] = label_key
            
            logging.info("--- %s ---" % model_name)
            logging.info("%d train words" % len(self.data[model_name]['train_matrix']))
            logging.info("%d dev words" % len(self.data[model_name]['dev_matrix']))
            logging.info("%d test words" % len(self.data[model_name]['test_matrix']))

        self.main_model_name = self.model_names[0]

        
    def build_model(self):
        self.models = {}

        if self.word_length <= 0: # variable length words
            self.word_length = None

        tokens_input = Input(
            shape = (self.word_length,), # use explicit word length for CNNs to work
            dtype = 'float32',
            name = 'phones_input'
        )

        tokens = Embedding(
            input_dim = self.vocab_size,
            output_dim = self.params['embedding_size'],
            trainable = True,
            name = 'phone_embeddings'
        )(tokens_input) # output shape: (batch_size, word_length, embedding size)

        # Add recurrent layers
        if(self.params['which_rnn'] == 'GRU'):
            rnn_func = CuDNNGRU if self.params['using_gpu'] else GRU
        elif(self.params['which_rnn'] == 'LSTM'):
            rnn_func = CuDNNLSTM if self.params['using_gpu'] else LSTM
        else:
            assert(False) # invalid rnn type
        recurrent_layer = tokens
        logging.info("lstm_size: %s" % str(self.params['lstm_size']))
        cnt = 1
        for size in self.params['lstm_size']:
            if isinstance(self.params['dropout'], (list, tuple)):
                if(self.params['using_gpu']):
                    raise ValueError('recurrent_dropout only works with CPU computation. Use simple dropout for GPU.')

                recurrent_layer = Bidirectional(rnn_func(
                        units = size,
                        return_sequences = True,
                        dropout = self.params['dropout'][0],
                        recurrent_dropout = self.params['dropout'][1]
                ), name = 'Bi'+ self.params['which_rnn'] +'_' + str(cnt)
                )(recurrent_layer)

            else:
                """ Naive dropout """
                recurrent_layer = Bidirectional(rnn_func(
                        units = size,
                        return_sequences = True
                    ), name = 'LSTM_' + str(cnt)
                )(recurrent_layer)

                if self.params['dropout'] > 0.0:
                    recurrent_layer = TimeDistributed(Dropout(
                            rate = self.params['dropout']
                        ), name = 'dropout_' + str(self.params['dropout']) + "_"+str(cnt)
                    )(recurrent_layer)

            cnt += 1

        # Add CNNs, inspired by Ma and Hovy, 2016. CNNs are parallel to LSTM instead of prior.
        if(self.params['use_cnn'] and self.params['cnn_layers'] > 0):
            cnn_layer = tokens
            # print('tokens shape initial:\t\t\t', tokens.shape) 
            # how to reshape::: re = Reshape((tokens.shape[1],tokens.shape[2],) + (1, ))(tokens) #  + (1, )

            for i in range(self.params['cnn_layers']):
                cnn_layer = Conv1D(
                    filters = self.params['cnn_num_filters'],
                    kernel_size = self.params['cnn_filter_size'],
                    padding = 'same',
                    name = "cnn_" + str(i+1)
                )(cnn_layer)
                
                if(self.params['cnn_max_pool_size']):
                    # maintain dimensionality (stride = 1)
                    cnn_layer = MaxPooling1D(
                        pool_size = self.params['cnn_max_pool_size'],
                        strides = 1,
                        padding = 'same',
                        name = 'max_pooling_' + str(i+1)
                    )(cnn_layer)

        # concatenating the CNN with the LSTM essentially tacks on the cnn vector to the end of each lstm time-step vector.
        if(self.params['use_cnn'] and self.params['cnn_layers'] > 0):
            concat_layer = concatenate([recurrent_layer, cnn_layer])
        else:
            concat_layer = recurrent_layer

        # Add output classifier
        for model_name in self.model_names:
            output = concat_layer
            for classifier in [self.params['classifier']]:
                if classifier == 'softmax':
                    output = TimeDistributed(Dense(
                            units = self.n_class_labels,
                            activation = 'softmax'
                        ), name = model_name + '_softmax'
                    )(output)
                    lossFct = 'sparse_categorical_crossentropy'
                
                elif classifier == 'crf': # use Philipp Gross' ChainCRF
                    output = TimeDistributed(Dense(
                            units = self.n_class_labels,
                            activation = None
                        ), name = model_name + '_hidden_lin_layer'
                    )(output)
                    crf = ChainCRF(name=model_name+'_crf')
                    output = crf(output)
                    lossFct = crf.sparse_loss
                
                elif classifier == 'kc-crf': # use keras-contrib CRF
                    output = TimeDistributed(Dense(
                            units = self.n_class_labels,
                            activation = None
                        ), name = model_name + '_hidden_lin_layer'
                    )(output)

                    crf = CRF(
                        units = self.n_class_labels,
                        learn_mode = 'join',
                        activation = self.params['crf_activation'],
                        sparse_target = True
                    )
                    output = crf(output)
                    lossFct = crf_loss
                    
                else:
                    assert(False) # bad classifier option

            # :: Parameters for the optimizer ::
            optimizerParams = {}
            if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
                optimizerParams['clipnorm'] = self.params['clipnorm']
            
            if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
                optimizerParams['clipvalue'] = self.params['clipvalue']
            
            if self.params['optimizer'].lower() == 'adam':
                opt = Adam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'nadam':
                opt = Nadam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'rmsprop': 
                opt = RMSprop(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adadelta':
                opt = Adadelta(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adagrad':
                opt = Adagrad(**optimizerParams)
            elif self.params['optimizer'].lower() == 'sgd':
                opt = SGD(lr=0.1, **optimizerParams)
            
            model = Model(inputs=[tokens_input], outputs=[output])
            model.compile(loss=lossFct, optimizer=opt)
            model.summary(line_length=125)
            
            self.models[model_name] = model


    def train_model(self):
        self.epoch += 1
        
        if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:       
            logging.info("Update Learning Rate to %f" % (self.learning_rate_updates[self.params['optimizer']][self.epoch]))
            for model_name in self.model_names:            
                K.set_value(self.models[model_name].optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch]) 

        for batch in self.minibatch_iterate_dataset():
            for model_name in self.model_names:         
                nn_labels = batch[model_name][0]
                nn_input = batch[model_name][1:]
                self.models[model_name].train_on_batch(nn_input, nn_labels)  


    def minibatch_iterate_dataset(self, model_names = None):
        """
        Create based on word length mini-batches with approx. the same size. 
        Words and mini-batch chunks are shuffled and used to the train the model
        """
        
        if self.train_word_length_ranges == None:
            """ Create mini batch ranges """
            self.train_word_length_ranges = {}
            self.train_mini_batch_ranges = {}            
            for model_name in self.model_names:
                train_data = self.data[model_name]['train_matrix']
                train_data.sort(key=lambda x:len(x['tokens'])) #Sort train matrix by word length
                train_ranges = []
                old_word_len = len(train_data[0]['tokens'])            
                idxStart = 0
                
                #Find start and end of ranges with words with same length
                for idx in range(len(train_data)):
                    word_len = len(train_data[idx]['tokens'])
                    
                    if word_len != old_word_len:
                        train_ranges.append((idxStart, idx))
                        idxStart = idx
                    
                    old_word_len = word_len
                
                #Add last word
                train_ranges.append((idxStart, len(train_data)))

                #Break up ranges into smaller mini batch sizes
                mini_batch_ranges = []
                for batch_range in train_ranges:
                    range_len = batch_range[1] - batch_range[0]

                    bins = int(math.ceil(range_len / float(self.params['mini_batch_size'])))
                    bin_size = int(math.ceil(range_len / float(bins)))
                    
                    for bin_nr in range(bins):
                        startIdx = bin_nr * bin_size + batch_range[0]
                        endIdx = min(batch_range[1],(bin_nr+1) * bin_size + batch_range[0])
                        mini_batch_ranges.append((startIdx, endIdx))
                      
                self.train_word_length_ranges[model_name] = train_ranges
                self.train_mini_batch_ranges[model_name] = mini_batch_ranges
                
        if model_names == None:
            model_names = self.model_names
            
        #Shuffle training data
        for model_name in model_names:      
            #1. Shuffle words that have the same length
            x = self.data[model_name]['train_matrix']
            for data_range in self.train_word_length_ranges[model_name]:
                for i in reversed(range(data_range[0]+1, data_range[1])):
                    # pick an element in x[:i+1] with which to exchange x[i]
                    j = random.randint(data_range[0], i)
                    x[i], x[j] = x[j], x[i]
               
            #2. Shuffle the order of the mini batch ranges       
            random.shuffle(self.train_mini_batch_ranges[model_name])

        #Iterate over the mini batch ranges
        if self.main_model_name != None:
            range_length = len(self.train_mini_batch_ranges[self.main_model_name])
        else:
            range_length = min([len(self.train_mini_batch_ranges[model_name]) for model_name in model_names])

        batches = {}
        for idx in range(range_length):
            batches.clear()
            
            for model_name in model_names:   
                trainMatrix = self.data[model_name]['train_matrix']
                data_range = self.train_mini_batch_ranges[model_name][idx % len(self.train_mini_batch_ranges[model_name])] 
                labels = np.asarray([trainMatrix[idx][self.label_keys[model_name]] for idx in range(data_range[0], data_range[1])])
                labels = np.expand_dims(labels, -1)
                
                
                batches[model_name] = [labels]
                
                for featureName in self.params['feature_names']:
                    inputData = np.asarray([trainMatrix[idx][featureName] for idx in range(data_range[0], data_range[1])])
                    batches[model_name].append(inputData)
            
            yield batches   


    def store_results(self, results_path):
        if results_path != None:
            directory = os.path.dirname(results_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            self.results_save_path = open(results_path, 'w')
        else:
            self.results_save_path = None


    def fit(self, epochs):
        if self.models is None:
            self.build_model()

        total_train_time = 0
        max_dev_score = {model_name:0 for model_name in self.models.keys()}
        max_test_score = {model_name:0 for model_name in self.models.keys()}
        no_improvement_since = 0
        
        for epoch in range(epochs):      
            sys.stdout.flush()           
            logging.info("\n--------- Epoch %d -----------" % (epoch+1))
            
            start_time = time.time()
            self.train_model()
            time_diff = time.time() - start_time
            total_train_time += time_diff
            logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))
            
            
            start_time = time.time() 
            for model_name in self.model_names:
                logging.info("-- %s --" % (model_name))
                dev_score, test_score = self.compute_acc_scores(model_name, self.data[model_name]['dev_matrix'], self.data[model_name]['test_matrix'])

                if dev_score > max_dev_score[model_name]:
                    max_dev_score[model_name] = dev_score
                    max_test_score[model_name] = test_score
                    no_improvement_since = 0

                    #Save the model
                    if self.model_save_path != None:
                        self.save_model(model_name, epoch, dev_score, test_score)
                else:
                    no_improvement_since += 1
                    
                if self.results_save_path != None:
                    self.results_save_path.write(
                        "\t".join(map(str, [
                            epoch + 1,
                            model_name,
                            dev_score,
                            test_score,
                            max_dev_score[model_name],
                            max_test_score[model_name],
                            time_diff, # training time for this epoch
                            total_train_time, # training time for all epochs
                            time.time() - start_time # time for evaluation during this epoch
                        ]))
                    )
                    self.results_save_path.write("\n")
                    self.results_save_path.flush()
                
                logging.info("\nScores from epoch with best dev-scores:\n  Dev-Score: %.4f\n  Test-Score %.4f" % (max_dev_score[model_name], max_test_score[model_name]))
                logging.info("")
                
            logging.info("%.2f sec for evaluation" % (time.time() - start_time))
            
            if self.params['early_stopping']  > 0 and no_improvement_since >= self.params['early_stopping']:
                logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
                break


    def tagWords(self, words):
        """
        words: [{'raw_tokens': ['S', 'V', 't', 'P', 'd'], 'tokens': [11, 5, 43, 36, 8]}, ...]
        """
        labels = {}
        for model_name, model in self.models.items():
            padded_pred_labels = self.predict_labels(model, words)
            pred_labels = []
            for idx in range(len(words)):
                unpadded_pred_labels = []
                for tokenIdx in range(len(words[idx]['tokens'])):
                    if words[idx]['tokens'][tokenIdx] != 0:  # Skip padding tokens
                        unpadded_pred_labels.append(padded_pred_labels[idx][tokenIdx])

                pred_labels.append(unpadded_pred_labels)

            labels[model_name] = pred_labels
        return labels
            
    
    def get_word_lengths(self, words):
        word_lengths = {}
        for idx in range(len(words)):
            word = words[idx]['tokens']
            if len(word) not in word_lengths:
                word_lengths[len(word)] = []
            word_lengths[len(word)].append(idx)
        
        return word_lengths

    def predict_labels(self, model, words):
        pred_labels = [None]*len(words)
        word_lengths = self.get_word_lengths(words)
        
        for indices in word_lengths.values():   
            nnInput = []                  
            for feature_name in self.params['feature_names']:
                input_data = np.asarray([words[idx][feature_name] for idx in indices])
                nnInput.append(input_data)
            
            predictions = model.predict(nnInput, verbose=False)
            predictions = predictions.argmax(axis=-1) # Predict classes            
           
            
            predIdx = 0
            for idx in indices:
                pred_labels[idx] = predictions[predIdx]    
                predIdx += 1   
        
        return pred_labels

    def compute_acc_scores(self, model_name, devMatrix, testMatrix):
        """
        Accuracy scores are reported at the word level. This means that if a single
        syllable boundary was incorrectly placed, the entire word is marked incorrect.
        
        Logs the boundary level accuracy as well.
        """
        dev_acc, dev_bound = self.compute_acc(model_name, devMatrix)
        test_acc, test_bound = self.compute_acc(model_name, testMatrix)

        logging.info("Word-Level Accuracy")
        logging.info("Dev: %.4f" % (dev_acc))
        logging.info("Test: %.4f" % (test_acc))

        logging.info("\nBoundary-Level Accuracy")
        logging.info("Dev: %.4f" % (dev_bound))
        logging.info("Test: %.4f" % (test_bound))
        
        return dev_acc, test_acc   
    
    def compute_acc(self, model_name, words):
        """
        Returns:
            float: word level accuracy. Range: [0.,1.]
            float: boundary_level_acc. Range: [0.,1.]
        """
        correct_labels = [words[idx][self.label_keys[model_name]] for idx in range(len(words))]
        pred_labels = self.predict_labels(self.models[model_name], words) 
        
        num_labels = 0
        num_corr_labels = 0
        num_words = 0
        num_corr_words = 0
        word_was_wrong = False
        for word_id in range(len(correct_labels)):
            num_words += 1
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
        word_level_acc = num_corr_words / float(num_words)
        return word_level_acc, boundary_level_acc

    def save_model(self, model_name, epoch, dev_score, test_score):
        import json
        import h5py

        if self.model_save_path == None:
            raise ValueError('model_save_path not specified.')

        savePath = self.model_save_path.replace("[DevScore]", "%.4f" % dev_score).replace("[TestScore]", "%.4f" % test_score).replace("[Epoch]", str(epoch+1)).replace("[ModelName]", model_name)

        directory = os.path.dirname(savePath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.isfile(savePath):
            logging.info("Model " + savePath + " already exists. Model will be overwritten")

        self.models[model_name].save(savePath, True)

        with h5py.File(savePath, 'a') as h5file:
            h5file.attrs['params'] = json.dumps(self.params)
            h5file.attrs['mappings'] = json.dumps(self.mappings)
            h5file.attrs['model_name'] = model_name
            h5file.attrs['label_key'] = self.datasets[model_name]['label']
            h5file.attrs['vocab_size'] = self.vocab_size
            h5file.attrs['n_class_labels'] = self.n_class_labels
            h5file.attrs['word_length'] = self.word_length if self.word_length != None else -1

    @staticmethod
    def load_model(model_path):
        import h5py
        import json

        with h5py.File(model_path, 'r') as f:
            params = json.loads(f.attrs['params'])
            mappings = json.loads(f.attrs['mappings'])
            model_name = f.attrs['model_name']
            label_key = f.attrs['label_key']
            vocab_size = f.attrs['vocab_size']
            n_class_labels = f.attrs['n_class_labels']
            word_length = f.attrs['word_length']

        if params['classifier'] == ['kc-crf']:
            from keras_contrib.layers import CRF
            from keras_contrib.losses import crf_loss
            keras.losses.crf_loss = crf_loss # HACK: integrate crf_loss directly into Keras. Sorry.
            custom_objects = {'CRF': CRF, 'loss': crf_loss}
        else:
            from .keraslayers.ChainCRF import create_custom_objects
            custom_objects=create_custom_objects()
            
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        bilstm = BiLSTM(params)
        bilstm.set_vocab_size(vocab_size, n_class_labels, word_length, mappings)
        bilstm.models = {model_name: model}
        bilstm.label_keys = {model_name: label_key}
        return bilstm
