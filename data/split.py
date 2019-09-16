import fire
import json
from random import shuffle

def split(file_name, train_percent, test_percent, dev_percent):
    """
    Args:
        file_name (str): input file name in json format. Must have form:
            [
                {
                    "pronunciation": "",
                    "syllabification": "",
                    "word": ""
                },
                ...
            ]
        train_percent (int): percentage of entries to be included in train.txt
        test_percent (int): percentage of entries to be included in test.txt
        dev_percent (int): percentage of entries to be included in dev.txt
    Outputs:
        generates files train.txt, test.txt, dev.txt
        each line has two columns [phone, syllable-boundary]
        if syllable-boundary is 1, there is a syllable boundary after the given phone.
        words are separated by newline.
    """
    assert(train_percent + test_percent + dev_percent == 100), 'all percentages must add to 100'
    assert(train_percent != 0 and test_percent != 0 and dev_percent != 0), 'all percentages must be greater than 0'
    with open(file_name, 'r') as json_in:
        data = json.load(json_in)
    
    shuffle(data)
    
    # make data splits 
    train_size = int(train_percent * len(data) * 0.01)
    test_size = int(test_percent * len(data) * 0.01)
    train = data[0:train_size]
    test = data[train_size:train_size+test_size]
    dev = data[train_size+test_size:]

    print('train.txt: ', len(train))
    print('test.txt: ', len(test))
    print('dev.txt: ', len(dev))

    # save each subset to their respective file
    for ds in [(train,'train.txt'), (test, 'test.txt'), (dev,'dev.txt')]:
        with open(ds[1], 'w') as f:
            syl_column_name = 'PhonSylDISC' # 'syllabification'
            for entry in ds[0]:
                for i,phone in enumerate(entry[syl_column_name]):
                    if(phone == '-'):
                        continue

                    # second column: 1 if there is a syllable boundary after the phone, else 0
                    if(i+1 == len(entry[syl_column_name])):
                        has_boundary = '0'
                    elif(entry[syl_column_name][i+1] == '-'):
                        has_boundary = '1'
                    else:
                        has_boundary = '0'

                    line = phone + '\t' + has_boundary + '\n'
                    f.write(line)

                # a newline separates each word
                f.write('\n')

if __name__ == '__main__':
    fire.Fire(split)