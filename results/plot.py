import fire
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

font = {
    'weight' : 'bold',
    'size' : 22
}
matplotlib.rc('font', **font)

def plot(csv_file):
    """
    Plots a graph of training accuracies as given by the model results output file.
    Args:
        csv_file (str): file name with csv extension.
    """
    df = pd.read_csv(csv_file, delimiter='\t', header=None, usecols=[1,2,3,4,5])

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.set_title(next(iter(set(df[1].values))) + ' syllabification training accuracy over epochs')    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Word-Level Accuracy')
    ax1.plot(list(range(len(df[2].values))), df[2].values, c='r', label='Dev Score', linewidth=5)
    ax1.plot(list(range(len(df[2].values))), df[3].values, c='b', label='Test Score', linewidth=5)
    ax1.grid()
    ax1.legend()

    plt.show()


if __name__ == '__main__':
    fire.Fire(plot)
