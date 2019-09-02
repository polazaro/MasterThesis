# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 01:08:45 2019

@author: Rubén
"""

''' 

This script is used to create lists of the lengths of the utterances for 
the rnn scenario, because now we are considering sequences of x frames/audios/
feature vectors. 

So, for example, if the length of the utterance is 50. We are taking only every
second (uno de cada dos), so we have 25 frames. If we pass through the network
sequences of 10 frames, we are going to have 16 consecutive sequences in this
utterance, so we have to use 16 predictions to compute the predicted label of 
this utterance. 

'''

import pandas as pd

ind_to_label = {
        0: "hap",
        1: "sad",
        2: "neu",
        3: "ang"
        }

test_ang_dict  = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/test_ang_dict.csv'
test_hap_dict  = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/test_hap_dict.csv'
test_neu_dict  = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/test_neu_dict.csv'
test_sad_dict  = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/test_sad_dict.csv'
train_ang_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/train_ang_dict.csv'
train_hap_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/train_hap_dict.csv'
train_neu_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/train_neu_dict.csv'
train_sad_dict = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/train_sad_dict.csv'
val_ang_dict   = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/val_ang_dict.csv'
val_hap_dict   = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/val_hap_dict.csv'
val_neu_dict   = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/val_neu_dict.csv'
val_sad_dict   = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_dict_2/val_sad_dict.csv'

dest_path = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_RNN_66/'

seq_train_len_hap = list()
seq_train_len_sad = list()
seq_train_len_neu = list()
seq_train_len_ang = list()
seq_test_len_hap  = list()
seq_test_len_sad  = list()
seq_test_len_neu  = list()
seq_test_len_ang  = list()
seq_val_len_hap   = list()
seq_val_len_sad   = list()
seq_val_len_neu   = list()
seq_val_len_ang   = list()
cont_bad_utt_train = 0
cont_bad_utt_val   = 0
cont_bad_utt_test  = 0

seq_len = 10

with open(train_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_train += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_train_len_hap.append(len(files_list)-(seq_len-1))
        else:
            pass

with open(train_sad_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_train += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_train_len_sad.append(len(files_list)-(seq_len-1))
        else:
            pass
        
with open(train_neu_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_train += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_train_len_neu.append(len(files_list)-(seq_len-1))
        else:
            pass
        
with open(train_ang_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_train += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_train_len_ang.append(len(files_list)-(seq_len-1))
        else:
            pass

with open(val_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_val += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_val_len_hap.append(len(files_list)-(seq_len-1))
        else:
            pass

with open(val_sad_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_val += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_val_len_sad.append(len(files_list)-(seq_len-1))
        else:
            pass
        
with open(val_neu_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_val += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_val_len_neu.append(len(files_list)-(seq_len-1))
        else:
            pass
        
with open(val_ang_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_val += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_val_len_ang.append(len(files_list)-(seq_len-1))
        else:
            pass  
        
with open(test_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_test += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_test_len_hap.append(len(files_list)-(seq_len-1))
        else:
            pass
        
with open(test_sad_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_test += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_test_len_sad.append(len(files_list)-(seq_len-1))
        else:
            pass
        
with open(test_neu_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_test += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_test_len_neu.append(len(files_list)-(seq_len-1))
        else:
            pass
        
with open(test_ang_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_test += 1
            else:
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                seq_test_len_ang.append(len(files_list)-(seq_len-1))
        else:
            pass
        
''' We save txt files with the list of lengths of the utterances per emotion '''
with open(dest_path + "train_hap_long.txt", 'w') as f:
    for s in seq_train_len_hap:
        f.write(str(s) + '\n')
        
with open(dest_path + "train_sad_long.txt", 'w') as f:
    for s in seq_train_len_sad:
        f.write(str(s) + '\n')

with open(dest_path + "train_neu_long.txt", 'w') as f:
    for s in seq_train_len_neu:
        f.write(str(s) + '\n')

with open(dest_path + "train_ang_long.txt", 'w') as f:
    for s in seq_train_len_ang:
        f.write(str(s) + '\n')

with open(dest_path + "test_hap_long.txt", 'w') as f:
    for s in seq_test_len_hap:
        f.write(str(s) + '\n')

with open(dest_path + "test_sad_long.txt", 'w') as f:
    for s in seq_test_len_sad:
        f.write(str(s) + '\n')

with open(dest_path + "test_neu_long.txt", 'w') as f:
    for s in seq_test_len_neu:
        f.write(str(s) + '\n')

with open(dest_path + "test_ang_long.txt", 'w') as f:
    for s in seq_test_len_ang:
        f.write(str(s) + '\n')

with open(dest_path + "val_hap_long.txt", 'w') as f:
    for s in seq_val_len_hap:
        f.write(str(s) + '\n')

with open(dest_path + "val_sad_long.txt", 'w') as f:
    for s in seq_val_len_sad:
        f.write(str(s) + '\n')

with open(dest_path + "val_neu_long.txt", 'w') as f:
    for s in seq_val_len_neu:
        f.write(str(s) + '\n')

with open(dest_path + "val_ang_long.txt", 'w') as f:
    for s in seq_val_len_ang:
        f.write(str(s) + '\n')

print('Total number of train sequences:{}'.format(sum(seq_train_len_hap)+sum(seq_train_len_sad)+sum(seq_train_len_neu)+sum(seq_train_len_ang)))
print('Total number of val   sequences:{}'.format(sum(seq_val_len_hap)+sum(seq_val_len_sad)+sum(seq_val_len_neu)+sum(seq_val_len_ang)))
print('Total number of test  sequences:{}'.format(sum(seq_test_len_hap)+sum(seq_test_len_sad)+sum(seq_test_len_neu)+sum(seq_test_len_ang)))


