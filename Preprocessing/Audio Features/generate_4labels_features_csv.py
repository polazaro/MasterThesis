# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 20:55:32 2019

@author: Rubén
"""

###############################################################################
##                                                                           ##
## Script to generate the csv with the features files (we use the audio csv  ##
## and change the extension). We generate the normal and the flipped csv.    ##
##                                                                           ##
###############################################################################


import pandas as pd

test_data       = pd.read_csv('D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_training_66/4Labels/test_4labels.csv')
test_data_flip  = pd.read_csv('D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_training_66/4Labels/test_4labels_flip.csv')
val_data        = pd.read_csv('D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_training_66/4Labels/validation_4labels.csv')
val_data_flip   = pd.read_csv('D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_training_66/4Labels/validation_4labels_flip.csv')
train_data      = pd.read_csv('D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_training_66/4Labels/training_4labels.csv')
train_data_flip = pd.read_csv('D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_training_66/4Labels/training_4labels_flip.csv')
path            = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_66/4Labels/'


'''
################
##   TRAIN    ##
################
'''

print('Doing Training CSVs...')

train = []
train_flip = []

for index, line in train_data.iterrows():
    train.append([line['audio_name'][:-4] + '.csv', line['label']])

pd_train = pd.DataFrame(train, columns=['csv_name','label'])
pd_train.to_csv(path + 'training_4labels.csv', index=False)


for index, line in train_data_flip.iterrows():
    train_flip.append([line['audio_name'][:-4] + '.csv', line['label']])

pd_train_flip = pd.DataFrame(train_flip, columns=['csv_name','label'])
pd_train_flip.to_csv(path + 'training_4labels_flip.csv', index=False)

print('Training CSVs finished.\n')


'''
################
## VALIDATION ##
################
'''

print('Doing Validation CSVs...')

val = []
val_flip = []

for index, line in val_data.iterrows():
    val.append([line['audio_name'][:-4] + '.csv', line['label']])

pd_val = pd.DataFrame(val, columns=['csv_name','label'])
pd_val.to_csv(path + 'validation_4labels.csv', index=False)


for index, line in val_data_flip.iterrows():
    val_flip.append([line['audio_name'][:-4] + '.csv', line['label']])

pd_val_flip = pd.DataFrame(val_flip, columns=['csv_name','label'])
pd_val_flip.to_csv(path + 'validation_4labels_flip.csv', index=False)

print('Validation CSVs finished.\n')


'''
################
##    TEST    ##
################
'''

print('Doing Test CSVs...')

test = []
test_flip = []

for index, line in test_data.iterrows():
    test.append([line['audio_name'][:-4] + '.csv', line['label']])

pd_test = pd.DataFrame(test, columns=['csv_name','label'])
pd_test.to_csv(path + 'test_4labels.csv', index=False)


for index, line in test_data_flip.iterrows():
    test_flip.append([line['audio_name'][:-4] + '.csv', line['label']])

pd_test_flip = pd.DataFrame(test_flip, columns=['csv_name','label'])
pd_test_flip.to_csv(path + 'test_4labels_flip.csv', index=False)

print('Test CSVs finished.\n')
