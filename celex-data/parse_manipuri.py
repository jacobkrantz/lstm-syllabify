# -*- coding: utf-8 -*-
'''
There is only the symbol and the pronounciation here
So, we are going to add the third category for consistency, by using the word twice (one for the word and the other for the unsyllabified phonetic representation)

The original dataset (found http://www.iitg.ac.in/cseweb/osint/resources.php) has the the words labeled using the Bengali representation of Manipuri. The first row has the word. Then, it has the words grouped into syllables with either beg, mid or end after it.

To fix this, we change all of the _beg/_mid to be dashes and the _ends to be blank. This creates a normal way for us.

This script turns this into a json file, for the split.py to conver the files properly.
'''

import json
f = open("manipuri_raw.txt","r")
contents = f.readlines()


chars = {} # List of the all the characters
word_list = []
for index in range(len(contents)):
    line = contents[index].strip()
    line = line.split('\t')
    word = line[0]
    pronounce =  line[1]
    pronounce = pronounce.replace('_beg','-')
    pronounce = pronounce.replace('_mid','-')
    pronounce = pronounce.replace('_end','')
    pronounce = pronounce.replace(' ','')
    for p in pronounce:
        if p not in chars:
            chars[p] = 1
        else:
            chars[p] = chars[p] + 1

    word_list.append((word, pronounce))

json_object = "[\n"
for entry in word_list:
    json_row ="""
{
    "pronunciation": "%s",
    "syllabification": "%s",
    "word": "%s"
},""" % (entry[0], entry[1], entry[0])

    json_object += json_row
    
json_object = json_object.rstrip(',') + '\n]'
with open("manipuri.json","w") as f:
    f.write(json_object)

# Write this to a sql table...
# Convert this to json...
# Then, use Jacob's special form...
