import sys
reload(sys)
sys.setdefaultencoding('utf-8')

'''
This uses the freely available dataset on the Lexique website.
I define a unique word as something with the same phonetic representation. So, not duplicate entries.
This also gets rid of words with commas, spaces and dashes.

This script turns this into a json file, for the split.py to convert the files properly.
'''

import json
import io
f = io.open("French.txt","r")
contents = f.readlines()


chars = {} # List of the all the characters
word_list = []
word_dict = {}
for index in range(len(contents)):
    line = contents[index].strip()
    line = line.split('\t')
    word = line[0]
    pronounce =  line[1]
    syllabification = line[2]
    for p in pronounce:
        if p not in chars:
            chars[p] = 1
        else:
            chars[p] = chars[p] + 1

    if("'" not in word and " " not in word and "-" not in word and not word in word_dict):
        word_list.append((word, pronounce,syllabification))
        word_dict[word] = 1


json_object = "[\n"
for entry in word_list:
    json_row ="""
{
    "pronunciation": "%s",
    "syllabification": "%s",
    "word": "%s"
},""" % (entry[1], entry[2], entry[0])

    json_object += json_row

json_object = json_object.rstrip(',') + '\n]'
with open("french.json","w") as f:
    f.write(json_object)

# Write this to a sql table...
# Convert this to json...
# Then, use Jacob's special form...
