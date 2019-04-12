import json
import fire


def convert(f_in, f_out):
    """
    converts a JSON file to a format that GloVe can recognize for embedding training.
    Args:
        f_in (str): file name of the json file to read in. Format:
            [
                {
                    "pronunciation": str
                    "syllabification": str
                    "word": str
                }, ...
            ]
        f_out (str): file name for the generated output file
    """
    word_list = [] # list of words which are strings of phones with spaces between each phone
    with open(f_in) as json_file:
        data = json.load(json_file)
        cnt = 0
        for word in data:
            cnt += 1
            word_list.append(' '.join(list(word['pronunciation'])))

        out_str = '\n'.join(word_list)
        with open(f_out, 'w') as f:
            f.write(out_str)
        
        print('Added', cnt, 'words to', f_out)

fire.Fire(convert)