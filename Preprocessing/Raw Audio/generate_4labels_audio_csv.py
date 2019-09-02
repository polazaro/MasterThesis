# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:55:03 2019

@author: Rubén
"""

###############################################################################
##                                                                           ##
## In this script we create a csv for training, validation and test with     ##
## the names of the files and the associated label. These csv files will be  ##
## useful for the data generator during the training.                        ##
##                                                                           ##
###############################################################################


import pandas as pd
import os

dest_path  = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/'
train_path = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train/'
val_path   = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation/'
test_path  = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test/'

ind_to_label = {
        0: "hap",
        1: "sad",
        2: "neu",
        3: "ang"
        }


'''
################
##   TRAIN    ##
################
'''
print('Doing Training CSV...')
train = []

for i in range(len(ind_to_label)):
    emotion = ind_to_label[i]
    path = train_path + emotion + '/'
    faces_names = os.listdir(path)
    faces_names.sort()
    for face in faces_names:
        train.append([emotion+'/'+face,i])
        train.append([emotion+'/'+face,i]) # We introduce two times each audio because we are going to have the corresponding frame and the same frame flipped

pd_train = pd.DataFrame(train, columns=['audio_name','label'])
pd_train.to_csv(dest_path+'training_4labels_flip.csv', index=False)
print('Training CSV finished.\n')

'''
################
## VALIDATION ##
################
'''
print('Doing Validation CSV...')
validation = []

for i in range(len(ind_to_label)):
    emotion = ind_to_label[i]
    path = val_path + emotion + '/'
    faces_names = os.listdir(path)
    faces_names.sort()
    for face in faces_names:
        validation.append([emotion+'/'+face,i])
        validation.append([emotion+'/'+face,i]) # We introduce two times each audio because we are going to have the corresponding frame and the same frame flipped

pd_val = pd.DataFrame(validation, columns=['audio_name','label'])
pd_val.to_csv(dest_path+'validation_4labels_flip.csv', index=False)
print('Validation CSV finished.\n')

'''
################
##    TEST    ##
################
'''
print('Doing Test CSV...')
test = []

for i in range(len(ind_to_label)):
    emotion = ind_to_label[i]
    path = test_path + emotion + '/'
    faces_names = os.listdir(path)
    faces_names.sort()
    for face in faces_names:
        test.append([emotion+'/'+face,i])
        test.append([emotion+'/'+face,i]) # We introduce two times each audio because we are going to have the corresponding frame and the same frame flipped

pd_test = pd.DataFrame(test, columns=['audio_name','label'])
pd_test.to_csv(dest_path+'test_4labels_flip.csv', index=False)
print('Test CSV finished.\n')
