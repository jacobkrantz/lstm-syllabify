from copy import copy
import logging
"""
potential problem: running an old model may fail due to mappings
    being regenerated every time.
    TODO: fix mappings to a specified map by storing them in pkl file.
"""

def load_dataset(datasets):
    embeddings = []
    mappings = {}
    data = {}

    for datasetName, dataset in datasets.items():
        dataset_columns = dataset['columns']

        trainData = 'celex-data/%s/train.txt' % datasetName 
        devData = 'celex-data/%s/dev.txt' % datasetName 
        testData = 'celex-data/%s/test.txt' % datasetName 
        paths = {'train_matrix': trainData, 'dev_matrix': devData, 'test_matrix': testData}

        logging.info(":: Transform " + datasetName + " dataset ::")
        mappings, vocab_size, n_class_labels = make_mappings(paths.values())
        data[datasetName] = process_data(paths, dataset_columns, dataset, mappings)

    # currently do not have pre-trained phonetic embeddings. 
    # returning embeddings = []. Embeddings mst be trained.
    return (embeddings, data, mappings, vocab_size, n_class_labels)

def make_mappings(paths):
    """
    creates a unique mapping from phone to integer.
    Args:
        paths (list<str>): file paths that hold all possible phones
    Returns:
        dict phone->int,
        int: vocab size (# phone inputs)
        int: number of class labels (possible boundary classifications)
    """
    all_phones = set()
    class_labels = set()

    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.split('\t') # (phone, boundary)
                if len(line) == 1:
                    continue
                all_phones.add(line[0])
                class_labels.add(line[1])

    mappings = {}
    for i, phone in enumerate(all_phones):
        mappings[phone] = i

    return mappings, len(mappings), len(class_labels)

def process_data(paths, dataset_columns, dataset, mappings):
    """
    hardcoded for certain columns. Mst be changed by hand.
    TODO: leverage details from dataset_columns and dataset.
    """
    data = {}

    for name, path in paths.items():

        # add 'raw_tokens' and 'boundaries' to data
        entries = []
        entry = {'raw_tokens':[], 'boundaries': []}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                if len(line) == 1:
                    entries.append(copy(entry))
                    entry['raw_tokens'] = []
                    entry['boundaries'] = []
                    continue
                
                entry['raw_tokens'].append(line[0])
                entry['boundaries'].append(int(line[1]))
            
        data[name] = entries

        # add 'tokens' to data
        for i, entry in enumerate(data[name]):
            data[name][i]['tokens'] = [mappings[raw] for raw in entry['raw_tokens']]

    return data
