# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:17:28 2019

@author: polaz
"""

import os
from scipy.io import wavfile
import numpy as np

name_audio = 'Ses01F_script01_1'
dir_audios = 'C:/Users/polaz/Documents/Máster Data Science/Segundo Cuatrimestre/TFM/Code/Segmentation/'
dir_seg = dir_audios + name_audio + '/'

files = os.listdir(dir_seg)
audio_files = files[:-2]
txt_file = files[-2]

file = open(name_audio + '/' + txt_file, 'r') 

window_t = 0.160
overlap = 0.3

for segment in audio_files:
    control = True
    fs, signal = wavfile.read(dir_seg + segment)
    window_len = 0
    cont = 0
    os.mkdir(dir_seg + segment[:-4])
    window_len = int(window_t*fs)
    i = 0
    while control:
        if window_len+cont > len(signal):
            output = np.zeros((window_len,),dtype=np.int16)
            new = signal[cont:]
            output[0:len(new)] = new
            control = False
        else:
            output = signal[cont:window_len+cont]
            cont += int(window_len*(1-overlap))
        i+=1
        wavfile.write(dir_seg + segment[:-4] + '/' + segment[:-4] + '_' + '{:04d}'.format(i) + '.wav',fs,output)

        
    
    
    
        


        
        

