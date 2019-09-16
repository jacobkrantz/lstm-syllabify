"""
Converts a CSV file to a JSON file. Each row must have the form:
    (word: str, pronunciation: str, syllabification: str)

python csv_to_json.py FILE_IN FILE_OUT
FILE_IN: name of the CSV file to convert to JSON
FILE_OUT: name of the output JSON file
Example:
    >>> python csv_to_json.py basque.csv basque.json
    Or:
    >>> python csv_to_json.py --file-in basque.csv --file-out basque.json
"""
import pandas as pd
import fire

def get_csv_as_json(filename):
    json_object = "[\n"
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
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


def main(file_in, file_out):
    with open(file_out,'w') as f:
        f.write(get_csv_as_json(file_in))

if __name__ == "__main__":
    fire.Fire(main)
