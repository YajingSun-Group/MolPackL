#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/06/1 10:07:52
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib
import numpy as np
import pandas as pd

def load_data():
    # Load data from file
    data0 = np.load('../dataset/density_HS0.npy').reshape(-1, 64,64, 1)
    data1 = np.load('../dataset/density_HS1.npy').reshape(-1, 64,64, 1)
    data2 = np.load('../dataset/density_HS_pro.npy').reshape(-1, 64,64, 1)
    labels = pd.read_csv('../dataset/band_gap_label.csv')['bandgap_energy_ev'].values

    group = pd.read_csv('../dataset/band_gap_label.csv')['group'].values
    

    return data0, data1, data2, labels, group

def data_preprocess(data0, data1, data2):
    """
    Preprocesses the input data by normalizing and clipping the values.

    Args:
        data0 (ndarray): Input data 0.
        data1 (ndarray): Input data 1.
        data2 (ndarray): Input data 2.

    Returns:
        ndarray: Preprocessed data with shape (data0.shape[0], 64, 64, 3).
    """

    # Normalize data0, data1, data2
    data0 = data0 / data0.max()
    data1 = data1 / data1.max()
    data2 = data2 / data2.max()

    # Clip values between 0 and 1
    data0 = data0.clip(0, 1)
    data1 = data1.clip(0, 1)
    data2 = data2.clip(0, 1)

    # Concatenate data0, data1, data2 along the last axis
    data = np.concatenate([data0, data1, data2], axis=-1)

    return data

def split_data(data, labels, group):
    """
    Split the data and labels into train, test, and validation sets based on the given group.

    Parameters:
    data (numpy.ndarray): The input data.
    labels (numpy.ndarray): The corresponding labels.
    group (numpy.ndarray): The group array indicating the category of each data point.

    Returns:
    tuple: A tuple containing train_x, test_x, val_x, train_y, test_y, and val_y arrays.
    """
    train_x = data[group == 'train']
    test_x = data[group == 'test']
    val_x = data[group == 'val']
    
    train_y = labels[group == 'train']
    test_y = labels[group == 'test']
    val_y = labels[group == 'val']
    
    return train_x, test_x, val_x, train_y, test_y, val_y

def data_normalize(train_x, test_x, val_x, data):
    """
    Normalize the data using mean and standard deviation.

    Parameters:
    train_x (numpy.ndarray): Training data.
    test_x (numpy.ndarray): Testing data.
    val_x (numpy.ndarray): Validation data.
    data (numpy.ndarray): Data to calculate mean and standard deviation.

    Returns:
    tuple: A tuple containing the normalized training data, testing data, and validation data.
    """

    mean_lt0 = np.mean(data)
    std_lt0 = np.std(data)

    train_x = (train_x - mean_lt0) / std_lt0
    val_x = (val_x - mean_lt0) / std_lt0
    test_x = (test_x - mean_lt0) / std_lt0

    return train_x, test_x, val_x





