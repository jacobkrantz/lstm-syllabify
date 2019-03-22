import os
import csv
import statistics
import numpy as np
import fire

"""
Analyze a directory of results
We throw out the lowest dev score because occasionally there is a bad experiment.

How to call:
    >>> python3 analyze_res_folder.py (folder-name)
    where folder-name is the directory within ./results/ to be analyzed
Example:
    >>> python3 analyze_res_folder.py 2048

csv column structure:
    (
        0 epoch,
        1 model name,
        2 dev word accuracy, 
        3 word accuracy,
        4 best dev word accuracy,
        5 best test word accuracy,
        6 training time for epoch (s),
        7 total training time thus far (s),
        8 evaluation time
    )
"""

def analyze(res_dir_name):
    res_dir_name = str(res_dir_name)
    epoch_finished_at = []
    training_times = []
    best_dev_accuracy = []
    test_accuracy = []

    res_dir = os.getcwd() + '/' + res_dir_name + '/'

    if not os.path.isdir(res_dir):
        raise ValueError(res_dir + ' is not a valid directory')

    for filename in os.listdir(res_dir):
        with open(res_dir + filename, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')

            # ignore all lines but the last
            for epoch in csv_reader:
                continue
            
            epoch_finished_at.append(int(epoch[0]))
            training_times.append(int(float(epoch[7])))
            best_dev_accuracy.append(float(epoch[4]))
            test_accuracy.append(float(epoch[5]))

    # throw out lowest dev score
    i_lowest = np.argmin(best_dev_accuracy)
    epoch_finished_at.pop(i_lowest)
    training_times.pop(i_lowest)
    best_dev_accuracy.pop(i_lowest)
    test_accuracy.pop(i_lowest)

    # display analysis
    print('---------------------------------------')
    print('Directory analyzed:\t\t', res_dir_name)
    print('---------------------------------------')
    print('Average dev accuracy:\t\t',  '{number:.{digits}f}'.format(number=100*np.mean(best_dev_accuracy),digits=3))
    print('Standard deviation dev:\t\t ', '{number:.{digits}f}'.format(number=100*statistics.stdev(best_dev_accuracy),digits=3))
    print('Average test accuracy:\t\t', '{number:.{digits}f}'.format(number=100*np.mean(test_accuracy),digits=3))
    print('Standard deviation test\t\t ', '{number:.{digits}f}'.format(number=100*statistics.stdev(test_accuracy),digits=3))
    print()
    print('Average number of total epochs:\t', '{number:.{digits}f}'.format(number=np.mean(epoch_finished_at),digits=2))
    print('number of experiments analyzed:\t', len(epoch_finished_at))

if __name__ == '__main__':
    fire.Fire(analyze)