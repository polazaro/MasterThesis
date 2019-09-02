# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 03:52:05 2019

@author: Rubén
"""

import cv2
import os

base_dir_ori = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_cropped/'
base_dir_des = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_frames/'


for i in range(5):
    print('SESIÓN ' + str(i+1) + ':\n')
    ses_dir_ori =  base_dir_ori + 'Session' + str(i+1) + '/'
    ses_dir_des =  base_dir_des + 'Session' + str(i+1) + '/'
    
    videos = os.listdir(ses_dir_ori)
    
    for video in videos:
        
        video_folder = ses_dir_des + video[:-4] + '/'
        os.mkdir(video_folder)
        vidcap = cv2.VideoCapture(ses_dir_ori + video)
        success, image = vidcap.read()
        cont = 1

        while success:
    
            cv2.imwrite(video_folder + video[:-4] + '_{:06d}'.format(cont) + '.bmp', image)     # save frame as JPEG file      
            success, image = vidcap.read()
#            print('Read a new frame: ', success)
            cont += 1
        
        print(video + ': ' + str(cont-1) + ' frames')
        
    print('\n\n')    
    