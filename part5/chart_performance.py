import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import sys

if __name__ == "__main__":
    folder_paths = sys.argv[1:]
    seeds = [15,36,41,82,121]
    #chart training performance
    fig, axis = plt.subplots(1, len(folder_paths))
    fig.set_figwidth(20)
    file_path = "/train_performance/training_record.csv"
    for i,folder_path in enumerate(folder_paths):
        df = pd.read_csv(folder_path+file_path, header=None)
        x_values = range(500)
        y_values = df.iloc[:,1]
        sns.lineplot(x=x_values, y=y_values, ax=axis[i])
        axis[i].set_title("Seed "+str(seeds[i]))
    fig.savefig('train_performance.png')
    #chart testing performance
    fig, axis = plt.subplots(1, len(folder_paths))
    fig.set_figwidth(20)
    file_path = "/test_performance/testing_record.csv"
    for i,folder_path in enumerate(folder_paths):
        df = pd.read_csv(folder_path+file_path, header=None)
        x_values = range(25)
        y_values = df.iloc[:,1]
        sns.lineplot(x=x_values, y=y_values, ax=axis[i])
        axis[i].set_title("Seed "+str(seeds[i]))
    fig.savefig('test_performance.png')