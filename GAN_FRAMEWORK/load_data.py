# -*- coding: utf-8 -*-
# author: Christian Arcos

import os
import pandas as pd
import numpy as np
from scipy.io import loadmat


def Input_image(image):
    images = loadmat(image).get('rad')
    images = images.astype('float32')/np.max(images)
    return images


def Oput_image(image):
    images = loadmat(image).get('rad')
    images = images.astype('float32')/np.max(images)
    return images


def load_sambles(data):
    data = data[['inimg']]
    inimg_name = list(data.iloc[:, 0])
    samples = []
    for samp in inimg_name:
        samples.append(samp)
    return samples

def conv_array(samples, lenData, PATH, IMG_WIDTH, IMG_HEIGHT, L_imput, L_bands, shuffle=True):

    X = np.empty((lenData, IMG_WIDTH, IMG_HEIGHT, L_imput )) 
    y = np.empty((lenData, IMG_WIDTH, IMG_HEIGHT, L_bands))
    
    for i, file_name in enumerate(samples):
        # Store sample
        X[i,] = Input_image(PATH + file_name)
        # Store class
        y[i,] = Oput_image(PATH  + file_name)
    
    return X,y

def Build_data_set(IMG_WIDTH=500, IMG_HEIGHT=500, L_bands=31, L_imput=12, BATCH_SIZE=4, PATH=None):
    # Random split
    #data_dir_list = os.listdir(PATH)
    data_dir_list = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]
    N = len(data_dir_list)
    train_df = pd.DataFrame(columns=['inimg'])
    test_df = pd.DataFrame(columns=['inimg'])
    randurls = np.copy(data_dir_list)
    train_n = round(N * 0.80)
    np.random.shuffle(randurls)
    tr_urls = randurls[:train_n]
    ts_urls = randurls[train_n:N]
    for i in tr_urls:
        train_df = train_df.append({'inimg': i}, ignore_index=True)
    for i in ts_urls:
        test_df = test_df.append({'inimg': i}, ignore_index=True)
        
    partition_Train = load_sambles(train_df)
    partition_Test = load_sambles(test_df)  
    
    params = {'IMG_WIDTH': IMG_WIDTH,
          'IMG_HEIGHT': IMG_WIDTH,
          'L_bands':L_bands,
          'L_imput':L_imput,
          'PATH': PATH,
          'shuffle': True}
    
    train_data = conv_array(partition_Train, len(partition_Train), **params)
    test_data = conv_array(partition_Test, len(partition_Test), **params)
    
    return train_data, test_data