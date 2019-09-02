# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 19:52:43 2019

@author: Rubén
"""

import pandas as pd

test_data  = pd.read_csv('C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_faces_2_training/test_4labels_flip.csv')
train_data = pd.read_csv('C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_faces_2_training/training_4labels_flip.csv')
val_data   = pd.read_csv('C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_faces_2_training/validation_4labels_flip.csv')
path       = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_faces_2_training/'


'''
################
##   TRAIN    ##
################
'''

print('Doing Training CSV...')
train = []

for index, line in train_data.iterrows():
    if 'flip' not in line['im_name']:
        train.append([line['im_name'], line['label']])

pd_train = pd.DataFrame(train, columns=['im_name','label'])
pd_train.to_csv(path + 'training_4labels.csv', index=False)
print('Training CSV finished.\n')


'''
################
## VALIDATION ##
################
'''

print('Doing Validation CSV...')
val = []

for index, line in val_data.iterrows():
    if 'flip' not in line['im_name']:
        val.append([line['im_name'], line['label']])

pd_val = pd.DataFrame(val, columns=['im_name','label'])
pd_val.to_csv(path + 'validation_4labels.csv', index=False)
print('Validation CSV finished.\n')


'''
################
##    TEST    ##
################
'''

print('Doing Test CSV...')
test = []

for index, line in test_data.iterrows():
    if 'flip' not in line['im_name']:
        test.append([line['im_name'], line['label']])

pd_test = pd.DataFrame(test, columns=['im_name','label'])
pd_test.to_csv(path + 'test_4labels.csv', index=False)
print('Test CSV finished.\n')