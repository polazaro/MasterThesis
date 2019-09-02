# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:31:47 2019

@author: Rubén
"""


###############################################################################
##                                                                           ##
## This script is used to move the videos obtained in the detect_errors.py   ##
## to another folder in order to process all together in OpenFace.           ##
##                                                                           ##
###############################################################################

import shutil


database_path = 'D:/Máster/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_cropped/'
dest_path = 'D:/Máster/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_cropped/Videos to correct'


with open('errors.txt','r') as f:
    data = f.readlines()
    
cont = 0
    
for name in data:
    # The sixth position of the names of the videos indicates the session
    if int(name[6]) == 1:
        video_path = database_path+'Session1/'+name[:-1]+'.avi'
    if int(name[6]) == 2:
        video_path = database_path+'Session2/'+name[:-1]+'.avi'
    if int(name[6]) == 3:
        video_path = database_path+'Session3/'+name[:-1]+'.avi'
    if int(name[6]) == 4:
        video_path = database_path+'Session4/'+name[:-1]+'.avi'
    if int(name[6]) == 5:
        video_path = database_path+'Session5/'+name[:-1]+'.avi'
    
    shutil.copy(video_path, dest_path)
    cont +=1
    print(cont)
    
    
print('DONE!')
        
        
        
        
        
        
        