import os
import csv
import statistics
import numpy as np
import fire

"""
Analyze a directory of results. Each file is a repeated run. Can determine
average scores and standard deviation across runs.

How to call:
    >>> python3 analyze_res_folder.py (folder-name)
    where folder-name is the directory within ./results/ to be analyzed
Example:
    >>> python3 analyze_res_folder.py batch_and_variation/2048

csv column structure:
    (
        0 epoch,
        1 dev word accuracy, 
        2 word accuracy,
        3 best dev word accuracy,
        4 best test word accuracy,
        5 training time for epoch (s),
        6 total training time thus far (s),
        7 evaluation time for epoch (s)
    )
"""


def analyze(res_dir_name):
    res_dir = os.getcwd() + "/" + str(res_dir_name) + "/"
    epoch_finished_at = []
    training_times = []
    epoch_training_time = []
    epoch_eval_time = []
    best_dev_accuracy = []
    test_accuracy = []

    if not os.path.isdir(res_dir):
        raise ValueError(res_dir + " is not a valid directory")

    for filename in os.listdir(res_dir):
        with open(res_dir + filename, "r") as f:
            csv_reader = csv.reader(f, delimiter="\t")

            # ignore all lines but the last
            for epoch in csv_reader:
                epoch_training_time.append(float(epoch[5]))
                epoch_eval_time.append(float(epoch[7]))

            epoch_finished_at.append(int(epoch[0]))
            training_times.append(int(float(epoch[6])))
            best_dev_accuracy.append(float(epoch[3]))
            test_accuracy.append(float(epoch[4]))

    # display analysis
    print("---------------------------------------")
    print("Directory analyzed:\t\t", res_dir_name)
    print("---------------------------------------")
    print(
        "Average dev accuracy:\t\t",
        "{number:.{digits}f}".format(number=100 * np.mean(best_dev_accuracy), digits=3),
    )
    if len(best_dev_accuracy) > 1:
        print(
            "Standard deviation dev:\t\t ",
            "{number:.{digits}f}".format(
                number=100 * statistics.stdev(best_dev_accuracy), digits=3
            ),
        )
    print(
        "Average test accuracy:\t\t",
        "{number:.{digits}f}".format(number=100 * np.mean(test_accuracy), digits=3),
    )
    if len(test_accuracy) > 1:
        print(
            "Standard deviation test\t\t ",
            "{number:.{digits}f}".format(
                number=100 * statistics.stdev(test_accuracy), digits=3
            ),
        )
    print("")
    print(
        "Average total training time:\t",
        "{number:.{digits}f}".format(number=np.mean(training_times), digits=3),
    )
    print(
        "Average epoch training time:\t",
        "{number:.{digits}f}".format(number=np.mean(epoch_training_time), digits=3),
    )
    print(
        "Average epoch eval time:\t",
        "{number:.{digits}f}".format(number=np.mean(epoch_eval_time), digits=3),
    )
    print(
        "Average number of total epochs:\t",
        "{number:.{digits}f}".format(number=np.mean(epoch_finished_at), digits=2),
    )
    print("number of experiments analyzed:\t", len(epoch_finished_at))


if __name__ == "__main__":
    fire.Fire(analyze)
