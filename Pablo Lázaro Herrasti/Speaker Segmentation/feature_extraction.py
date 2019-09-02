# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:16:16 2019

@author: polaz
"""

import subprocess
import pandas as pd
import csv
import os
import numpy as np

OpenSMILE_path = 'D:/Data_Master/TFM/opensmile-2.3.0/bin/Win32/SMILExtract_Release.exe'
conf_file = 'D:/Data_Master/TFM/opensmile-2.3.0/config/IS13_ComParE.conf'
name_audio = 'Ses01F_script01_1'
dir_audios = 'C:/Users/polaz/Documents/Máster Data Science/Segundo Cuatrimestre/TFM/Code/Segmentation/'
dir_seg = dir_audios + name_audio + '/'

common_cmd = OpenSMILE_path + " -C " + conf_file + " -I "

files = os.listdir(dir_seg)
folders = []
for f in files:
    if str.find(f,'.')<0:
        folders.append(f)
    
for folder in folders: 
    print(folder)
    folder_dir = dir_seg + folder + '/'
    segments = os.listdir(folder_dir)
    for segment in segments:
        csv_file = folder_dir + segment[:-4] + ".csv"
        audio_file = folder_dir + segment
        cmd = common_cmd + '"' + audio_file + '"' + " -O " + '"' + csv_file + '"'
        subprocess.call(cmd,shell=True)
        with open(csv_file, 'r') as f:
            r = csv.reader(f)
            for row in r:
                features = row[1:-1]  #Quitamos la primera y la última porque no contienen información
                
        os.remove(csv_file)
        features = np.array([float(x) for x in features])
        pd.DataFrame(features).to_csv(csv_file, header=None, index=None)