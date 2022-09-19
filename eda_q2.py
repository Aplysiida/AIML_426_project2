import pandas as pd

import sys

"""
Parse files into tuple (num of items, bag capacity, dataframe)
"""
def parse_data(filepath):
    df = pd.read_table(filepath, sep=' ')
    M = df.columns[0]
    Q = df.columns[1]
    df.columns.values[0], df.columns.values[1] = 'value','weight'
    return (M,Q,df)  

if __name__ == "__main__":
    filepaths = sys.argv[1:]
    #store dataset as (num of items, bag capacity, data)
    datasets = [parse_data(filepath=filepath) for filepath in filepaths] #parse files