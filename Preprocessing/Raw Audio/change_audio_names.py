# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 01:41:40 2019

@author: Rubén
"""

###############################################################################
##                                                                           ##
## This script is used to change the names of the audio files in order to    ##
## keep the same order as the faces in the csv that we are going to use to   ##
## train the fusion models.                                                  ##
##                                                                           ##
###############################################################################

import os

audio_path = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_training/4 Labels/'
sets = ['test/','train/','validation/']

ind_to_label = {
        0: "hap",
        1: "sad",
        2: "neu",
        3: "ang"
        }

for s in range(3):
    path1 = audio_path + sets[s]
    print(sets[s] + '\n')
    for i in range(4):
        path = path1 + ind_to_label[i] + '/'
        audio_files = os.listdir(path)
            
        for audio_file in audio_files:
            a = audio_file[:-4].split('_')
            new_name = audio_file[:-(4+len(a[-1]))] + '{:06d}'.format(int(a[-1])) + '.wav'
            os.rename(path+audio_file, path+new_name)
                
        print('Path '+ ind_to_label[i] + 'done\n')
