"""
A bidirectional LSTM with optional CRF for NLP sequence tagging.

Author: Jacob Krantz
Based on work done by Nils Reimers
License: Apache-2.0

TODO: clean up evaluateModelNames. we will always evaluate a model.
"""

from __future__ import print_function

import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
import math
import numpy as np
import sys
import gc
import time
import os
import random
import logging

from .keraslayers.ChainCRF import ChainCRF

class BiLSTM:
    def __init__(self, params=None):
        # modelSavePath = Path for storing models, resultsSavePath = Path for storing output labels while training
        self.models = None
        self.modelSavePath = None
        self.resultsSavePath = None

        # Hyperparameters for the network
        defaultParams = {
            'dropout': (0.5,0.5),
            'classifier': ['Softmax'],
            'LSTM-Size': (100,),
            'customClassifier': {},
            'optimizer': 'adam',
            'clipvalue': 0,
            'clipnorm': 1,
            'earlyStopping': 5,
            'miniBatchSize': 32,
            'featureNames': ['tokens'],
            'addFeatureDimensions': 10,
            'embedding-size': 50
        }
        if params != None:
            defaultParams.update(params)
        self.params = defaultParams

    def setVocabSize(self, vocab_size, n_class_labels):
        # class labels are syllable boundary labels
        self.vocab_size = vocab_size
        self.n_class_labels = n_class_labels

    def setDataset(self, datasets, data):
        self.datasets = datasets
        self.data = data

        # Create some helping variables
        self.mainModelName = None
        self.epoch = 0
        self.learning_rate_updates = {'sgd': {1: 0.1, 3: 0.05, 5: 0.01}}
        self.modelNames = list(self.datasets.keys())
        self.evaluateModelNames = []
        self.labelKeys = {}
        self.trainMiniBatchRanges = None
        self.trainWordLengthRanges = None

        for modelName in self.modelNames:
            labelKey = self.datasets[modelName]['label']
            self.labelKeys[modelName] = labelKey
            self.evaluateModelNames.append(modelName)
            
            logging.info("--- %s ---" % modelName)
            logging.info("%d train words" % len(self.data[modelName]['train_matrix']))
            logging.info("%d dev words" % len(self.data[modelName]['dev_matrix']))
            logging.info("%d test words" % len(self.data[modelName]['test_matrix']))

        if len(self.evaluateModelNames) == 1:
            self.mainModelName = self.evaluateModelNames[0]

        
    def buildModel(self):
        self.models = {}

        tokens_input = Input(shape=(None,), dtype='int32', name='phones_input')
        tokens = Embedding(input_dim=self.vocab_size, output_dim=self.params['embedding-size'], trainable=True, name='phone_embeddings')(tokens_input)

        inputNodes = [tokens_input]
        mergeInputLayers = [tokens]

        if len(mergeInputLayers) >= 2:
            merged_input = concatenate(mergeInputLayers)
        else:
            merged_input = mergeInputLayers[0]

        # Add LSTMs
        shared_layer = merged_input
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:      
            if isinstance(self.params['dropout'], (list, tuple)):  
                shared_layer = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name='shared_varLSTM_'+str(cnt))(shared_layer)
            else:
                """ Naive dropout """
                shared_layer = Bidirectional(LSTM(size, return_sequences=True), name='shared_LSTM_'+str(cnt))(shared_layer) 
                if self.params['dropout'] > 0.0:
                    shared_layer = TimeDistributed(Dropout(self.params['dropout']), name='shared_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(shared_layer)

            cnt += 1
            
        for modelName in self.modelNames:
            output = shared_layer
            
            modelClassifier = self.params['customClassifier'][modelName] if modelName in self.params['customClassifier'] else self.params['classifier']

            if not isinstance(modelClassifier, (tuple, list)):
                modelClassifier = [modelClassifier]
            
            cnt = 1
            for classifier in modelClassifier:
                if classifier == 'Softmax':
                    output = TimeDistributed(Dense(self.n_class_labels, activation='softmax'), name=modelName+'_softmax')(output)
                    lossFct = 'sparse_categorical_crossentropy'
                elif classifier == 'CRF':
                    output = TimeDistributed(Dense(self.n_class_labels, activation=None),
                                             name=modelName + '_hidden_lin_layer')(output)
                    crf = ChainCRF(name=modelName+'_crf')
                    output = crf(output)
                    lossFct = crf.sparse_loss
                elif isinstance(classifier, (list, tuple)) and classifier[0] == 'LSTM':
                            
                    size = classifier[1]
                    if isinstance(self.params['dropout'], (list, tuple)): 
                        output = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name=modelName+'_varLSTM_'+str(cnt))(output)
                    else:
                        """ Naive dropout """ 
                        output = Bidirectional(LSTM(size, return_sequences=True), name=modelName+'_LSTM_'+str(cnt))(output) 
                        if self.params['dropout'] > 0.0:
                            output = TimeDistributed(Dropout(self.params['dropout']), name=modelName+'_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(output)                    
                else:
                    assert(False) #Wrong classifier
                    
                cnt += 1
                
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
            
            
            model = Model(inputs=inputNodes, outputs=[output])
            model.compile(loss=lossFct, optimizer=opt)
            
            model.summary(line_length=125)
            #logging.info(model.get_config())
            #logging.info("Optimizer: %s - %s" % (str(type(model.optimizer)), str(model.optimizer.get_config())))
            
            self.models[modelName] = model


    def trainModel(self):
        self.epoch += 1
        
        if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:       
            logging.info("Update Learning Rate to %f" % (self.learning_rate_updates[self.params['optimizer']][self.epoch]))
            for modelName in self.modelNames:            
                K.set_value(self.models[modelName].optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch]) 
                
            
        for batch in self.minibatch_iterate_dataset():
            for modelName in self.modelNames:         
                nnLabels = batch[modelName][0]
                nnInput = batch[modelName][1:]
                self.models[modelName].train_on_batch(nnInput, nnLabels)  


    def minibatch_iterate_dataset(self, modelNames = None):
        """ Create based on word length mini-batches with approx. the same size. Words and 
        mini-batch chunks are shuffled and used to the train the model """
        
        if self.trainWordLengthRanges == None:
            """ Create mini batch ranges """
            self.trainWordLengthRanges = {}
            self.trainMiniBatchRanges = {}            
            for modelName in self.modelNames:
                trainData = self.data[modelName]['train_matrix']
                trainData.sort(key=lambda x:len(x['tokens'])) #Sort train matrix by word length
                trainRanges = []
                oldSentLength = len(trainData[0]['tokens'])            
                idxStart = 0
                
                #Find start and end of ranges with words with same length
                for idx in range(len(trainData)):
                    sentLength = len(trainData[idx]['tokens'])
                    
                    if sentLength != oldSentLength:
                        trainRanges.append((idxStart, idx))
                        idxStart = idx
                    
                    oldSentLength = sentLength
                
                #Add last word
                trainRanges.append((idxStart, len(trainData)))

                #Break up ranges into smaller mini batch sizes
                miniBatchRanges = []
                for batchRange in trainRanges:
                    rangeLen = batchRange[1]-batchRange[0]

                    bins = int(math.ceil(rangeLen/float(self.params['miniBatchSize'])))
                    binSize = int(math.ceil(rangeLen / float(bins)))
                    
                    for binNr in range(bins):
                        startIdx = binNr*binSize+batchRange[0]
                        endIdx = min(batchRange[1],(binNr+1)*binSize+batchRange[0])
                        miniBatchRanges.append((startIdx, endIdx))
                      
                self.trainWordLengthRanges[modelName] = trainRanges
                self.trainMiniBatchRanges[modelName] = miniBatchRanges
                
        if modelNames == None:
            modelNames = self.modelNames
            
        #Shuffle training data
        for modelName in modelNames:      
            #1. Shuffle words that have the same length
            x = self.data[modelName]['train_matrix']
            for dataRange in self.trainWordLengthRanges[modelName]:
                for i in reversed(range(dataRange[0]+1, dataRange[1])):
                    # pick an element in x[:i+1] with which to exchange x[i]
                    j = random.randint(dataRange[0], i)
                    x[i], x[j] = x[j], x[i]
               
            #2. Shuffle the order of the mini batch ranges       
            random.shuffle(self.trainMiniBatchRanges[modelName])
     
        
        #Iterate over the mini batch ranges
        if self.mainModelName != None:
            rangeLength = len(self.trainMiniBatchRanges[self.mainModelName])
        else:
            rangeLength = min([len(self.trainMiniBatchRanges[modelName]) for modelName in modelNames])

        batches = {}
        for idx in range(rangeLength):
            batches.clear()
            
            for modelName in modelNames:   
                trainMatrix = self.data[modelName]['train_matrix']
                dataRange = self.trainMiniBatchRanges[modelName][idx % len(self.trainMiniBatchRanges[modelName])] 
                labels = np.asarray([trainMatrix[idx][self.labelKeys[modelName]] for idx in range(dataRange[0], dataRange[1])])
                labels = np.expand_dims(labels, -1)
                
                
                batches[modelName] = [labels]
                
                for featureName in self.params['featureNames']:
                    inputData = np.asarray([trainMatrix[idx][featureName] for idx in range(dataRange[0], dataRange[1])])
                    batches[modelName].append(inputData)
            
            yield batches   


    def storeResults(self, resultsFilepath):
        if resultsFilepath != None:
            directory = os.path.dirname(resultsFilepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            self.resultsSavePath = open(resultsFilepath, 'w')
        else:
            self.resultsSavePath = None

    def fit(self, epochs):
        if self.models is None:
            self.buildModel()

        total_train_time = 0
        max_dev_score = {modelName:0 for modelName in self.models.keys()}
        max_test_score = {modelName:0 for modelName in self.models.keys()}
        no_improvement_since = 0
        
        for epoch in range(epochs):      
            sys.stdout.flush()           
            logging.info("\n--------- Epoch %d -----------" % (epoch+1))
            
            start_time = time.time()
            self.trainModel()
            time_diff = time.time() - start_time
            total_train_time += time_diff
            logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))
            
            
            start_time = time.time() 
            for modelName in self.evaluateModelNames:
                logging.info("-- %s --" % (modelName))
                dev_score, test_score = self.computeAccScores(modelName, self.data[modelName]['dev_matrix'], self.data[modelName]['test_matrix'])

                if dev_score > max_dev_score[modelName]:
                    max_dev_score[modelName] = dev_score
                    max_test_score[modelName] = test_score
                    no_improvement_since = 0

                    #Save the model
                    if self.modelSavePath != None:
                        self.saveModel(modelName, epoch, dev_score, test_score)
                else:
                    no_improvement_since += 1
                    
                if self.resultsSavePath != None:
                    self.resultsSavePath.write("\t".join(map(str, [epoch + 1, modelName, dev_score, test_score, max_dev_score[modelName], max_test_score[modelName]])))
                    self.resultsSavePath.write("\n")
                    self.resultsSavePath.flush()
                
                logging.info("\nScores from epoch with best dev-scores:\n  Dev-Score: %.4f\n  Test-Score %.4f" % (max_dev_score[modelName], max_test_score[modelName]))
                logging.info("")
                
            logging.info("%.2f sec for evaluation" % (time.time() - start_time))
            
            if self.params['earlyStopping']  > 0 and no_improvement_since >= self.params['earlyStopping']:
                logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
                break
            
            
    def tagWords(self, words):
        labels = {}
        for modelName, model in self.models.items():
            paddedPredLabels = self.predictLabels(model, words)
            predLabels = []
            for idx in range(len(words)):
                unpaddedPredLabels = []
                for tokenIdx in range(len(words[idx]['tokens'])):
                    if words[idx]['tokens'][tokenIdx] != 0:  # Skip padding tokens
                        unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])

                predLabels.append(unpaddedPredLabels)

            # can skip id to label map, as they are one in the same
            # idx2Label = self.idx2Labels[modelName]
            # labels[modelName] = [[idx2Label[tag] for tag in tagWord] for tagWord in predLabels]
            labels[modelName] = predLabels
        return labels
            
    
    def getWordLengths(self, words):
        wordLengths = {}
        for idx in range(len(words)):
            word = words[idx]['tokens']
            if len(word) not in wordLengths:
                wordLengths[len(word)] = []
            wordLengths[len(word)].append(idx)
        
        return wordLengths

    def predictLabels(self, model, words):
        predLabels = [None]*len(words)
        wordLengths = self.getWordLengths(words)
        
        for indices in wordLengths.values():   
            nnInput = []                  
            for featureName in self.params['featureNames']:
                inputData = np.asarray([words[idx][featureName] for idx in indices])
                nnInput.append(inputData)
            
            predictions = model.predict(nnInput, verbose=False)
            predictions = predictions.argmax(axis=-1) # Predict classes            
           
            
            predIdx = 0
            for idx in indices:
                predLabels[idx] = predictions[predIdx]    
                predIdx += 1   
        
        return predLabels

    def computeAccScores(self, modelName, devMatrix, testMatrix):
        """
        Accuracy scores are reported at the word level. This means that if a single
        syllable boundary was incorrectly placed, the entire word is marked incorrect.
        
        Logs the boundary level accuracy as well.
        """
        dev_acc, dev_bound = self.computeAcc(modelName, devMatrix)
        test_acc, test_bound = self.computeAcc(modelName, testMatrix)

        logging.info("Word-Level Accuracy")
        logging.info("Dev: %.4f" % (dev_acc))
        logging.info("Test: %.4f" % (test_acc))

        logging.info("Boundary-Level Accuracy")
        logging.info("Dev: %.4f" % (dev_bound))
        logging.info("Test: %.4f" % (test_bound))
        
        return dev_acc, test_acc   
    
    def computeAcc(self, modelName, words):
        """
        Returns:
            float: word level accuracy. Range: [0.,1.]
            float: boundary_level_acc. Range: [0.,1.]
        """
        correct_labels = [words[idx][self.labelKeys[modelName]] for idx in range(len(words))]
        pred_labels = self.predictLabels(self.models[modelName], words) 
        
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

    def saveModel(self, modelName, epoch, dev_score, test_score):
        import json
        import h5py

        if self.modelSavePath == None:
            raise ValueError('modelSavePath not specified.')

        savePath = self.modelSavePath.replace("[DevScore]", "%.4f" % dev_score).replace("[TestScore]", "%.4f" % test_score).replace("[Epoch]", str(epoch+1)).replace("[ModelName]", modelName)

        directory = os.path.dirname(savePath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.isfile(savePath):
            logging.info("Model " + savePath + " already exists. Model will be overwritten")

        self.models[modelName].save(savePath, True)

        with h5py.File(savePath, 'a') as h5file:
            h5file.attrs['params'] = json.dumps(self.params)
            h5file.attrs['modelName'] = modelName
            h5file.attrs['labelKey'] = self.datasets[modelName]['label']
            h5file.attrs['vocab_size'] = self.vocab_size
            h5file.attrs['n_class_labels'] = self.n_class_labels


    @staticmethod
    def loadModel(modelPath):
        import h5py
        import json
        from .keraslayers.ChainCRF import create_custom_objects

        model = keras.models.load_model(modelPath, custom_objects=create_custom_objects())

        with h5py.File(modelPath, 'r') as f:
            params = json.loads(f.attrs['params'])
            modelName = f.attrs['modelName']
            labelKey = f.attrs['labelKey']
            vocab_size = f.attrs['vocab_size']
            n_class_labels = f.attrs['n_class_labels']

        bilstm = BiLSTM(params)
        bilstm.setVocabSize(vocab_size, n_class_labels)
        bilstm.models = {modelName: model}
        bilstm.labelKeys = {modelName: labelKey}
        return bilstm
