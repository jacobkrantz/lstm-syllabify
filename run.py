#!/usr/bin/python

import fire

from preprocessing import read_conll_single, create_data_matrix
from neuralnets.BiLSTM import BiLSTM


def run(model_path, input_path, config_path):
    """
    This script loads a pretrained model and a input file in CoNLL format (each line a token, words separated by an empty line).
    The input words are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
    model_path (string): path to a pretrained model in .h5 format.
    input_path (string): path to the input file in CoNLL format of words to be syllabified.
    """
    words = read_conll_single(
        input_path
    )  # words: list [ { tokens: [ raw_tokens, ... ] } ... ]

    model = BiLSTM.load_model(model_path, config_path)
    data_matrix = create_data_matrix(words, model.mappings)
    tags = model.tagWords(data_matrix)["english"]

    print("\nTagged Words: ")
    for i, word in enumerate(words):
        joined = []
        for j, ch in enumerate(word["tokens"]):
            # pad tags with 0 to length of word.
            if len(tags[i]) < len(word["tokens"]):
                tags[i] += [0] * (len(word["tokens"]) - len(tags[i]))
            joined.append((ch, tags[i][j]))

        for tup in joined:
            print(tup[0], end="")
            if tup[1] == 1:
                print("-", end="")

        print("")


if __name__ == "__main__":
    fire.Fire(run)
