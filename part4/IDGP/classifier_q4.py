from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pandas as pd

import time
import sys

def read_datasets(folder_path):
    train_data = pd.read_csv(folder_path+'_train_pattern_file.csv')
    train_labels = train_data.loc[:,'class']
    train_data.drop(columns='class', inplace=True)
    
    test_data = pd.read_csv(folder_path+'_test_pattern_file.csv')
    test_labels = test_data.loc[:,'class']
    test_data.drop(columns='class', inplace=True)

    return train_data, train_labels, test_data, test_labels

def train_test_classifier(train_data, train_labels, test_data, test_labels):
    start = time.time()
    classifier = KNeighborsClassifier().fit(train_data, train_labels)
    end = time.time()
    predicted = classifier.predict(test_data)

    #use metrics to evaluate classifier
    accuracy = accuracy_score(y_true=test_labels, y_pred=predicted)
    print('\tExecution Time = ',(end-start),'s')
    print('\tAccuracy = ',accuracy)

if __name__ == "__main__":
    folder_paths = sys.argv[1:]
    #f1
    print('For dataset F1')
    f1_train_data, f1_train_labels, f1_test_data, f1_test_labels = read_datasets(folder_paths[0])
    print('\tF1 Shape = ', f1_train_data.shape)
    train_test_classifier(train_data=f1_train_data, train_labels=f1_train_labels, test_data=f1_test_data, test_labels=f1_test_labels)
    #f2
    print('For dataset F2')