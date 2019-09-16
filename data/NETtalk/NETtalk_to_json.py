import fire
import json
import csv

def derive_syllables(word, syllabification):
    """
    Process:
        1. Throw out '-' charactrer from 'word' and update 'syllabification' as necessary.
        2. Derive syllable boundaries using these rules:
            (Rule 1) A syllable boundary exists after the last '<'.
            (Rule 2) A syllable boundary exists before the first '>'.
            (Rule 3) A syllable boundary exists between any two numbers.
            (Rule 4) Boundaries beginning a word or ending a word are ignored.
    Args:
        word (str): word in phonetic representation.
        syllabification (str): same length as word. Uses NETtalk formulation of stress and
            syllable information. See `nettalk.names` for details.
    Returns:
        (str): word in phonetic representation. No '-' characters exist.
        (str): word in phonetic representation with '-' denoting syllable boundaries
    """
    def remove_hyphen(word, syllabification):
        word = list(word)
        syllabification = list(syllabification)
        try:
            assert(len(word) == len(syllabification))
        except AssertionError as err:
            print('Input syllabification different length than input word: skipping entry:')
            print('\tword:', "".join(word))
            print('\tsyll:', "".join(syllabification))
            raise err

        new_word = []
        new_syllabification = []
        for i in range(len(word)):
            if(word[i] != '-'):
                new_word.append(word[i])
                new_syllabification.append(syllabification[i])

        try:
            assert(len(new_syllabification) == len(new_word))
        except AssertionError as err:
            raise ValueError("After removing '-' character, lengths were different.")
        return new_word, new_syllabification

    def convert_syllabification(word, syllabification):
        boundary_after_index = [] # can have duplicates
        previous_chr = ''
        for i in range(len(syllabification)):
            # Rule 1
            if(i < len(syllabification)-1):
                if(syllabification[i] == '<' and syllabification[i+1] != '<'):
                    boundary_after_index.append(i)
            # Rule 2
            if(syllabification[i] == '>' and previous_chr != '>' and i > 0):
                boundary_after_index.append(i-1)
            # Rule 3
            if(syllabification[i] in ['0','1','2'] and previous_chr in ['0','1','2']):
                boundary_after_index.append(i-1)
            previous_chr = syllabification[i]
        
        # if(len(boundary_after_index)):
        #     print('word:', word)
        #     print('syll:', syllabification)
        #     print('boundaries: ', boundary_after_index)
        #     print('')

        # use boundary_after_index to build new syllabification
        new_syllab = []
        for i in range(len(word)):
            new_syllab.append(word[i])
            if(i in boundary_after_index):
                new_syllab.append('-')

        return word, new_syllab

    word, syllabification = remove_hyphen(word, syllabification)
    word, syllabification = convert_syllabification(word, syllabification)
    return "".join(word), "".join(syllabification)

def get_file(filename):
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')

        # toss out header
        for _ in range(10):
            reader.__next__()
        for row in reader:
            rows.append(row)

    json_object = "[\n"
    count = 0
    for row in rows:
        word = row[0]
        try:
            phone, phoneSyl = derive_syllables(row[1], row[2])
        except AssertionError:
            continue
        json_row ="""
    {
        "pronunciation": "%s",
        "syllabification": "%s",
        "word": "%s"
    },""" % (phone, phoneSyl, word)
        json_object += json_row
        count += 1

    json_object = json_object.rstrip(',') + '\n]'
    print('Processed', count, 'entries successfully.')
    return json_object

def convert(filename="nettalk.data"):
    json_file = get_file(filename)
    with open("nettalk.json","w") as f:
        f.write(json_file)

if __name__ == '__main__':
    fire.Fire(convert)
