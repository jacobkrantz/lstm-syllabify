# CSV To JSON
import json
import pandas as pd


def get_file(filename):

    json_object = "[\n"
    df = pd.read_csv(filename)
    for index,row in df.iterrows():
        word,phone,phoneSyl = row
        json_row ="""
        {
                "pronunciation": "%s",
                "syllabification": "%s",
                "word": "%s"

        },""" % (phone, phoneSyl, word)
        json_object += json_row
        if(index % 1000 == 0):
            print(index)
    json_object += '\n]'

    return json_object

json_file = get_file("Basque.csv")
file = open("optimized_basque.json","w")
file.write(json_file)
file.close()
