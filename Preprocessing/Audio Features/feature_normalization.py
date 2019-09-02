# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:41:48 2019

@author: Rubén
"""

###############################################################################
##                                                                           ##
## Script to normalize the audio features to range [0,1].                    ##
##                                                                           ##
###############################################################################

import numpy as np
import pandas as pd
import os
#from tqdm import tqdm

''' My computer paths '''
#train_path = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/4Labels/train/'
#val_path   = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/4Labels/validation/'
#test_path  = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/4Labels/test/'
#
#max_csv = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/4Labels/features_max.csv'
#min_csv = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/4Labels/features_min.csv'

''' Server paths '''
train_path = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train/'
val_path   = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation/'
test_path  = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/test/'

max_csv = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/features_max.csv'
min_csv = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/features_min.csv'

# Useful variables
labels = ['ang', 'hap', 'neu', 'sad']
n_features = 6373

''' This first part is to check if there are wrong files '''

print('Checking if there are broken files...'.format(n_features))
broken_files = 0
bf_list = []

for label in labels:
    print(label+'...')
    files = os.listdir(train_path + label + '/')
    files.sort()
    for file in tqdm(files):
        csv_file = train_path + label + '/' + file
        try:
            features = pd.read_csv(csv_file, header=None)
        except:
            bf_list.append(csv_file)
            broken_files += 1
        
for label in labels:
    print(label+'...')
    files = os.listdir(val_path + label + '/')
    files.sort()
    for file in tqdm(files):
        csv_file = val_path + label + '/' + file
        features = pd.read_csv(csv_file, header=None)
        try:
            features = pd.read_csv(csv_file, header=None)
        except:
            bf_list.append(csv_file)
            broken_files += 1
            
for label in labels:
    print(label+'...')
    files = os.listdir(test_path + label + '/')
    files.sort()
    for file in tqdm(files):
        csv_file = test_path + label + '/' + file
        features = pd.read_csv(csv_file, header=None)
        try:
            features = pd.read_csv(csv_file, header=None)
        except:
            bf_list.append(csv_file)
            broken_files += 1


print('\n\n\n Number of broken files: {}'.format(broken_files))



''' This second part is to obtain the mamimum and minimum values of the features using only training and validation sets'''

minimo = np.ones((n_features,1))*np.inf
maximo = -np.ones((n_features,1))*np.inf

print('Extracting maximums and minimums for {} features...'.format(n_features))

for label in labels:
    print(label+'...')
    files = os.listdir(train_path + label + '/')
    files.sort()
    for file in tqdm(files):
        csv_file = train_path + label + '/' + file
        features = pd.read_csv(csv_file, header=None).values
        minimo = np.minimum(minimo,features)
        maximo = np.maximum(maximo,features)
        
for label in labels:
    print(label+'...')
    files = os.listdir(val_path + label + '/')
    files.sort()
    for file in tqdm(files):
        csv_file = val_path + label + '/' + file
        features = pd.read_csv(csv_file, header=None).values
        minimo = np.minimum(minimo,features)
        maximo = np.maximum(maximo,features)

pd.DataFrame(maximo).to_csv(max_csv, header=None, index=None)
pd.DataFrame(minimo).to_csv(min_csv, header=None, index=None)


''' In this third part we normalize all the feature vectors '''

print('Normalizing feature vectors...')

maximo = pd.read_csv(max_csv, header=None).values
minimo = pd.read_csv(min_csv, header=None).values

comp = maximo != minimo  ## Quitamos las features que siempre tienen el mismo valor: maximo y mínimo son iguales, porque no aportan información.
n_norm_features = sum(comp)[0]

'''
################
##   TRAIN    ##
################
'''
print('Doing training set...')
for label in labels:
    print('\t {}...'.format(label))
    files = os.listdir(train_path + label + '/')
    files.sort()
    for file in files:
        # Load previous features
        norm_features = np.zeros((n_norm_features,1))
        csv_file = train_path + label + '/' + file
        features = pd.read_csv(csv_file, header=None).values
        
        # Compute new normaliza features
        cont = 0
        for i in range(n_features):
            if maximo[i] != minimo[i]:
                norm_features[cont] = (features[i]-minimo[i])/(maximo[i]-minimo[i])
                cont += 1
        
        # Delete previous file and save new one
        os.remove(csv_file)
        pd.DataFrame(norm_features).to_csv(csv_file, header=None, index=None)


'''
################
## VALIDATION ##
################
'''
print('Doing validation set...')
for label in labels:
    print('\t {}...'.format(label))
    files = os.listdir(val_path + label + '/')
    files.sort()
    for file in files:
        # Load previous features
        norm_features = np.zeros((n_norm_features,1))
        csv_file = val_path + label + '/' + file
        features = pd.read_csv(csv_file, header=None).values
        
        # Compute new normaliza features
        cont = 0
        for i in range(n_features):
            if maximo[i] != minimo[i]:
                norm_features[cont] = (features[i]-minimo[i])/(maximo[i]-minimo[i])
                cont += 1
        
        # Delete previous file and save new one
        os.remove(csv_file)
        pd.DataFrame(norm_features).to_csv(csv_file, header=None, index=None)


'''
################
##    TEST    ##
################
'''
print('Doing test set...')
for label in labels:
    print('\t {}...'.format(label))
    files = os.listdir(test_path + label + '/')
    files.sort()
    for file in files:
        # Load previous features
        norm_features = np.zeros((n_norm_features,1))
        csv_file = test_path + label + '/' + file
        features = pd.read_csv(csv_file, header=None).values
        
        # Compute new normaliza features
        cont = 0
        for i in range(n_features):
            if maximo[i] != minimo[i]:
                norm_features[cont] = (features[i]-minimo[i])/(maximo[i]-minimo[i])
                cont += 1
        
        # Delete previous file and save new one
        os.remove(csv_file)
        pd.DataFrame(norm_features).to_csv(csv_file, header=None, index=None)

