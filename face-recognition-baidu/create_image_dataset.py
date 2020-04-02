#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import os
import numpy as np
from batch_data import pickled

# DATA_LEN = 3*CHANNEL_LEN = 3*SHAPE*SHAPE
DATA_LEN = 3072  # length of an image after flattened
CHANNEL_LEN = 1024 # channel length such as "R" of a colorful image
SHAPE = 32 # suppose an image as a square image, height = width = SHAPE


def read_image(img_path, shape=None, color="RGB", mode=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if color == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if shape != None:
        assert isinstance(shape, int)
        img = cv2.resize(img, (shape, shape))
    
    img = img.transpose([2, 0, 1]).flatten()
    #The first 1024 entries contain the red channel values, 
    # the next 1024 the green, and the final 1024 the blue. 
    # The image is stored in row-major order, 
    # so that the first 32 entries of the array 
    # are the red channel values of the first row of the image.
    return img


def read_data(filename, shape=None, color="RGB" ):
    '''
    filename(str) is a file not a folder.
    every per line have an image path and  a label.
    return (numpy): an array of image and an array of label
    '''
    if os.path.isdir(filename):
        print("Please check filename! Make sure it is a file instead of a folder!")
    else:
        with open(filename, 'r') as f:
            lines = f.readlines()
            img_count = len(lines)
            dataset = np.zeros((img_count, DATA_LEN), dtype=np.uint8)
            #label = np.zeros(img_count, dtype=np.uint8)
            #img_path_lst = [line.strip().split('\t')[0] for line in lines]
            #label = [[line.strip().split('\t')[1] for line in lines]]
            
            img_name_lst = []
            label_lst = []
            index = 0
            s = SHAPE
            for line in lines:
                img_path, label = line.strip().split('\t')
                img = read_image(img_path, shape=s, color='RGB')
                dataset[index, :] = img
                img_name_lst.append(img_path.split('/')[-1])
                label_lst.append(int(label))
                index +=1
    
    return dataset, label_lst, img_name_lst


if __name__ == '__main__':
    train_file_list = "face/trainer.list"
    test_file_list = "face/test.list"
    savepath = "dataset"
    
    train_dataset, train_label, train_imgnames = read_data(train_file_list, shape=32, color='RGB')
    test_dataset, test_label, test_imgnames = read_data(test_file_list, shape=32, color='RGB')
    pickled(savepath, train_dataset, train_label, train_imgnames, batch_num=2, mode='train')
    pickled(savepath, test_dataset, test_label, test_imgnames, batch_num=2, mode='test')
    print("Done！")
