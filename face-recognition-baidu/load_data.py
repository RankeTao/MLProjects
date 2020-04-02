#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import pickle
import numpy as np
from batch_data import unpickle


def load_data(file_path, shape=32, channel=3):
    assert os.path.isdir(file_path)
    x_tr = np.array([])
    y_tr = []
    x_te = np.array([])
    y_te = []
    for file in os.listdir(file_path):
        if 'test' in file.split('_'):
            test_data_dict = unpickle(os.path.join(file_path, file))
            x, y = test_data_dict['data'], test_data_dict['labels']
            if not x_te.any():
                x_te = x
                y_te = y
                print(len(y_te))
            else:
                x_te = np.vstack((x_te, x))
                y_te.extend(y)

            # x_test = x.reshape([x.shape[0], channel, shape*shape]).transpose([0,2,1]).reshape(x.shape[0], shape, shape, channel)
            # y_test = y
        else:
            train_data_dict = unpickle(os.path.join(file_path, file))
            x, y = train_data_dict['data'], train_data_dict['labels']
            if not x_tr.any():
                x_tr = x
                y_tr = y
            else:
                x_tr = np.vstack((x_tr, x))
                y_tr.extend(y)
            # x_train = x.reshape([x.shape[0], channel, shape*shape]).transpose([0,2,1]).reshape(x.shape[0], shape, shape, channel)
            # y_train = y  
    x_train = x_tr.reshape([x_tr.shape[0], channel, shape*shape]).transpose([0, 2, 1]).reshape(x_tr.shape[0], shape, shape, channel)
    y_train = y_tr
    x_test = x_te.reshape([x_te.shape[0], channel, shape*shape]).transpose([0, 2, 1]).reshape(x_te.shape[0], shape, shape, channel)
    y_test = y_te
    return (x_train, y_train), (x_test, y_test)

# if __name__ == "__main__":
#     dataset_path = "C:/Users/iweut/PythonTest/deeplearning/face-recognition-baidu/dataset"
#     (x_train, y_train), (x_test, y_test) = load_data(dataset_path)
#     print(x_train.shape, len(y_train), x_test.shape, len(y_test))