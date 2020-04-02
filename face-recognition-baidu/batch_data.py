#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import pickle

BATCH_COUNTS = 1


def pickled(savepath, dataset, label_lst, fnames, batch_num=BATCH_COUNTS, mode="train"):
    '''
        savepath (str): save path of batched data.\n
        dataset (array): all image data, a nx3072 array.\n
        label (list): all image label, a list with length n.\n
        fnames (str list): all the image names, a list with length n. \n
        batch_num (int): divide dataset into batch_num bins.\n
        mode (str): {'train', 'test'}.
    '''
    assert os.path.isdir(savepath)
    total_num = len(fnames)
    samples_per_bin = int(total_num / batch_num)
    assert samples_per_bin > 0
    idx = 0
    for i in range(batch_num): 
        start = i*samples_per_bin
        end = (i+1)*samples_per_bin
        
        if end <= total_num:
            dict = {'data': dataset[start:end, :],
                    'labels': label_lst[start:end],
                    'filenames': fnames[start:end]
                    }
        else:
            dict = {'data': dataset[start:, :],
                    'labels': label_lst[start:],
                    'filenames': fnames[start:]
                    }
        
        if mode == "train":
            dict['batch_label'] = "training batch {} of {}".format(idx, batch_num-1)
            with open(os.path.join(savepath, 'data_batch_'+str(idx)), 'wb') as fi:
                pickle.dump(dict, fi)
            idx = idx + 1
        else:
            dict['batch_label'] = "testing batch {} of {}".format(idx, batch_num-1)
            with open(os.path.join(savepath, 'test_batch_'+str(idx)), 'wb') as fi:
                pickle.dump(dict, fi)
            idx = idx + 1


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
