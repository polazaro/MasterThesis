# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:04:02 2019

@author: Rubén
"""

'''

Just to create a histogram of the lengths of the utterances in terms 
of number of frames.

'''


import matplotlib.pyplot as plt

test_ang_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/test_ang_dict.csv'
test_hap_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/test_hap_dict.csv'
test_neu_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/test_neu_dict.csv'
test_sad_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/test_sad_dict.csv'
train_ang_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/train_ang_dict.csv'
train_hap_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/train_hap_dict.csv'
train_neu_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/train_neu_dict.csv'
train_sad_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/train_sad_dict.csv'
val_ang_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/val_ang_dict.csv'
val_hap_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/val_hap_dict.csv'
val_neu_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/val_neu_dict.csv'
val_sad_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/val_sad_dict.csv'


utt_len_train = list()
utt_len_test = list()
utt_len_val = list()

with open(test_ang_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_test.append(len(value[1:-2].split(',')))
        else:
            pass
        
with open(test_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_test.append(len(value[1:-2].split(',')))
        else:
            pass
        
with open(test_neu_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_test.append(len(value[1:-2].split(',')))
        else:
            pass

with open(test_sad_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_test.append(len(value[1:-2].split(',')))
        else:
            pass

with open(train_ang_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_train.append(len(value[1:-2].split(',')))
            if len(value[1:-2].split(',')) == 1:
                print('ang:'+key)
        else:
            pass

with open(train_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_train.append(len(value[1:-2].split(',')))
            if len(value[1:-2].split(',')) == 1:
                print('hap:'+key)
        else:
            pass

with open(train_neu_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_train.append(len(value[1:-2].split(',')))
            if len(value[1:-2].split(',')) == 1:
                print('neu:'+key)
        else:
            pass

with open(train_sad_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_train.append(len(value[1:-2].split(',')))
            if len(value[1:-2].split(',')) == 1:
                print('sad:'+key)
        else:
            pass

with open(val_ang_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_val.append(len(value[1:-2].split(',')))
        else:
            pass

with open(val_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_val.append(len(value[1:-2].split(',')))
        else:
            pass

with open(val_neu_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_val.append(len(value[1:-2].split(',')))
        else:
            pass

with open(val_sad_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            utt_len_val.append(len(value[1:-2].split(',')))
        else:
            pass
        

plt.figure()
plt.title('Utterance Length - Training Set')
plt.ylim(0,250)
plt.xlim(0,1050)
plt.xlabel('# Frames')
plt.ylabel('# Utterances')
plt.hist(utt_len_train, bins=100)
plt.grid(True)
plt.show()

plt.figure()
plt.title('Utterance Length - Validation Set')
plt.ylim(0,250)
plt.xlim(0,1050)
plt.xlabel('# Frames')
plt.ylabel('# Utterances')
plt.hist(utt_len_val, bins=100)
plt.grid(True)
plt.show()

plt.figure()
plt.title('Utterance Length - Test Set')
plt.ylim(0,250)
plt.xlim(0,1050)
plt.xlabel('# Frames')
plt.ylabel('# Utterances')
plt.hist(utt_len_test, bins=100)
plt.grid(True)
plt.show()

print('Number of utterances in TRAIN set with =< 5 frames: {}'.format(len([i for i in utt_len_train if i <= 5])))
print('Number of utterances in VALIDATION set with =< 5 frames: {}'.format(len([i for i in utt_len_val if i <= 5])))
print('Number of utterances in TEST set with =< 5 frames: {}'.format(len([i for i in utt_len_test if i <= 5])))


