# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 03:42:16 2019

@author: Rubén
"""

###############################################################################
##                                                                           ##
## We have seen that in some video swe have lost some frames. We don't know  ##
## the reason, so we are goig to check if (the number in) the name of the    ##
## last frame matchs the number of frames of the folder for eac video.       ##
##                                                                           ##
## We print in the terminal the name of the videos where some frames are     ##
## lost.                                                                     ##
##                                                                           ##
###############################################################################


import os

path = 'D:/Máster/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/Utterances_txt/Labels_4/'
frames_path = 'D:/Máster/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_faces/Session'
cont = 0

with open('errors.txt','w') as f:
    for i in range(1,6):
        session_path = frames_path+str(i)+'/'
        files_path = path+'Session'+str(i)+'/'
        files_names = os.listdir(files_path)
    
        for file_name in files_names:
            folder = file_name[:-4]+'_aligned'
            directory = session_path+folder
            frames_names = os.listdir(directory)
            if len(frames_names) != int(frames_names[-1][13:19]):
                cont +=1
                print(file_name[:-4])
                f.write(file_name[:-4]+'\n')


print('\n\n Nº of files with wrong number of frames: ',cont)
