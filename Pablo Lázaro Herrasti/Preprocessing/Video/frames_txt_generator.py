# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 01:38:34 2019

@author: Rubén
"""

###############################################################################
##                                                                           ##
## We use the train, validation and test.txt and change the lines in the     ##
## following way:                                                            ##
##                                                                           ##
## faces  --> Session5/1_Ses05F_impro01_aligned/frame_det_00_000108.bmp      ##
## faces_2 --> Session5/1_Ses05F_impro01/1_Ses05F_impro01_000108.bmp         ##
##                                                                           ##
## The rest of the line is the same.                                         ##
##                                                                           ##
###############################################################################


import os
from tqdm import tqdm

# Useful paths
path = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/Utterances_txt_faces/Labels_4/'
path_2 = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/Utterances_txt_faces_2/Labels_4/'


'''
################
##   TRAIN    ##
################
'''

cont_train = 0
print('PREPARING TRAIN TXT\n')
    
with open(path + 'train_data.txt' , "r") as f:
    data = f.readlines()

with open(path_2 + 'train_data.txt' , "w") as f:
    for line in tqdm(data):
        old_frame, sesion_video, audio, label = line[:-1].split('\t')
        f.write(sesion_video + audio[:-4] + '.bmp' + '\t' + sesion_video + '\t' + audio + '\t' + label + '\n')
        
print('\n')     
print('----TRAINING TXT DONE----')     
        

'''
################
## VALIDATION ##
################
'''

cont_train = 0
print('PREPARING VALIDATION TXT\n')
    
with open(path + 'val_data.txt' , "r") as f:
    data = f.readlines()

with open(path_2 + 'val_data.txt' , "w") as f:
    for line in tqdm(data):
        old_frame, sesion_video, audio, label = line[:-1].split('\t')
        f.write(sesion_video + audio[:-4] + '.bmp' + '\t' + sesion_video + '\t' + audio + '\t' + label + '\n')
        
print('\n')     
print('----VALIDATION TXT DONE----')      


'''
################
##    TEST    ##
################
'''

cont_train = 0
print('PREPARING TEST TXT\n')
    
with open(path + 'test_data.txt' , "r") as f:
    data = f.readlines()

with open(path_2 + 'test_data.txt' , "w") as f:
    for line in tqdm(data):
        old_frame, sesion_video, audio, label = line[:-1].split('\t')
        f.write(sesion_video + audio[:-4] + '.bmp' + '\t' + sesion_video + '\t' + audio + '\t' + label + '\n')
        
print('\n')     
print('----TEST TXT DONE----')  

