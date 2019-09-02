# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:32:47 2019

@author: Rubén
"""

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

dest_path = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_faces_2_training/'

seq_train = list()
seq_test  = list()
seq_val  = list()
cont_bad_utt_train = 0
cont_bad_utt_val   = 0
cont_bad_utt_test  = 0

seq_len = 10


'''

TRAIN

'''

with open(train_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_train += 1
            else:
                files_list = ['hap/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_train.append([files_list[i:i+seq_len], 0])
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
                files_list = ['sad/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_train.append([files_list[i:i+seq_len], 1])
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
                files_list = ['neu/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_train.append([files_list[i:i+seq_len], 2])
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
                files_list = ['ang/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_train.append([files_list[i:i+seq_len], 3])
        else:
            pass
        
with open(train_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_train += 1
            else:
                files_list = ['hap/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_train.append([files_list[i:i+seq_len], 0])
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
                files_list = ['sad/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_train.append([files_list[i:i+seq_len], 1])
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
                files_list = ['neu/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_train.append([files_list[i:i+seq_len], 2])
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
                files_list = ['ang/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_train.append([files_list[i:i+seq_len], 3])
        else:
            pass
        
'''

VALIDATION

'''

with open(val_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_val += 1
            else:
                files_list = ['hap/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_val.append([files_list[i:i+seq_len], 0])
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
                files_list = ['sad/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_val.append([files_list[i:i+seq_len], 1])
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
                files_list = ['neu/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_val.append([files_list[i:i+seq_len], 2])
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
                files_list = ['ang/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_val.append([files_list[i:i+seq_len], 3])
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
                files_list = ['hap/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_val.append([files_list[i:i+seq_len], 0])
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
                files_list = ['sad/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_val.append([files_list[i:i+seq_len], 1])
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
                files_list = ['neu/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_val.append([files_list[i:i+seq_len], 2])
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
                files_list = ['ang/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_val.append([files_list[i:i+seq_len], 3])
        else:
            pass  
        
'''

TEST

'''
        
with open(test_hap_dict, 'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key, value = item.split(':',1)
            files_list = value[1:-2].split(',')
            if len(files_list) < 20:
                cont_bad_utt_test += 1
            else:
                files_list = ['hap/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_test.append([files_list[i:i+seq_len], 0])
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
                files_list = ['sad/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_test.append([files_list[i:i+seq_len], 1])
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
                files_list = ['neu/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_test.append([files_list[i:i+seq_len], 2])
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
                files_list = ['ang/'+file+'.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_test.append([files_list[i:i+seq_len], 3])
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
                files_list = ['hap/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_test.append([files_list[i:i+seq_len], 0])
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
                files_list = ['sad/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_test.append([files_list[i:i+seq_len], 1])
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
                files_list = ['neu/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_test.append([files_list[i:i+seq_len], 2])
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
                files_list = ['ang/'+file+'_flip.bmp' for file in files_list]
                files_list = files_list[::2] # Because we are only going to use every second (1 de cada dos)
                files_list = [file.replace(' ','').replace("'",'') for file in files_list]
                for i in range(0, len(files_list)-(seq_len-1)):
                    seq_test.append([files_list[i:i+seq_len], 3])
        else:
            pass
        
        
''' Saving the files '''
        
df_train = pd.DataFrame(seq_train, columns=['seq','label'])   
df_val   = pd.DataFrame(seq_val,   columns=['seq','label'])    
df_test  = pd.DataFrame(seq_test,  columns=['seq','label'])

df_train.to_csv(dest_path + 'train_seqs_flip.csv', index=False)
df_val.to_csv(dest_path + 'val_seqs_flip.csv', index=False)
df_test.to_csv(dest_path + 'test_seqs_flip.csv', index=False)
