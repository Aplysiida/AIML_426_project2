import pandas as pd
import numpy as np

import feature_function as fe_fs

"""
Convert the image data into feature vectors using the FLGP program generated from IDGP_main.py
    dataset = raw image dataset
    labels = labels for dataset
    gp_func = FLGP program to convert to feature vector
"""
def img_to_pattern(dataset, labels, gp_func):
    data = []
    for img in dataset:
        vec = gp_func(img)
        data.append(vec)
    df = pd.DataFrame(data=data)
    df['class'] = labels
    return df

if __name__ == "__main__":
    randomSeeds=2
    #get best for f1 Local_SIFT(Region_R(Image0, 102, 38, 43, 45))
    best_gp_tree_f1 = lambda x : fe_fs.all_sift(fe_fs.regionR(x, 102, 38, 43, 45))
    train_f1_data = np.load('f1_train_data.npy') /255.0
    train_f1_labels = np.load('f1_train_label.npy')
    test_f1_data = np.load('f1_test_data.npy') /255.0
    test_f1_labels = np.load('f1_test_label.npy')

    #get best for f2
    best_gp_tree_f2 = ''
    train_f2_data = np.load('f2_train_data.npy') /255.0
    train_f2_labels = np.load('f2_train_label.npy')
    test_f2_data = np.load('f2_test_data.npy') /255.0
    test_f2_labels = np.load('f2_test_label.npy')

    #convert f1 npy to csv
    f1_train_df = img_to_pattern(train_f1_data, train_f1_labels, best_gp_tree_f1)
    f1_train_df.to_csv('data/f1_train_pattern_file.csv')
    f1_test_df = img_to_pattern(train_f1_data, train_f1_labels, best_gp_tree_f1)
    f1_test_df.to_csv('data/f1_test_pattern_file.csv')

    #convert f2 npy to csv