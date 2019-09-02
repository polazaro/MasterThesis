# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:45:49 2019

@author: Rubén
"""

import subprocess
import pandas as pd
import csv
import os
import numpy as np
from tqdm import tqdm

OpenSMILE_path   = r"C:/Users/ruben/Downloads/opensmile-2.3.0/bin/Win32/SMILExtract_Release.exe"
conf_file        = r"C:/Users/ruben/Downloads/opensmile-2.3.0/config/IS13_ComParE.conf"
audio_train_path = r"D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_training_100/4Labels/train/"
audio_val_path   = r"D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_training_100/4Labels/validation/"
audio_test_path  = r"D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_training_100/4Labels/test/"
dest_train_path  = r"D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/4Labels/train/"
dest_val_path    = r"D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/4Labels/validation/"
dest_test_path   = r"D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/4Labels/test/"

common_cmd = OpenSMILE_path + r" -C " + conf_file + r" -I "

labels = [r'ang', r'hap', r'neu', r'sad']

'''
################
##   TRAIN    ##
################
'''
print('Extracting audio features of the Training set...')

for label in labels:
    print(label+'...')
    files = os.listdir(audio_train_path + label + '/')
    files.sort()
    for file in tqdm(files):
        audio_file = audio_train_path + label + '/' + file
        csv_file = dest_train_path + label + '/' + file[:-4] + r".csv"
        
        cmd = common_cmd + '"' + audio_file + '"' + r" -O " + '"' + csv_file + '"'
        subprocess.call(cmd,shell=True)
        
        with open(csv_file, 'r') as f:
            r = csv.reader(f)
            for row in r:
                features = row[1:-1]  #Quitamos la primera y la última porque no contienen información
                
        os.remove(csv_file)
        features = np.array([float(x) for x in features])
        pd.DataFrame(features).to_csv(csv_file, header=None, index=None)
    
    
    
'''
################
## VALIDATION ##
################
'''
print('Extracting audio features of the Validation set...')

for label in labels:
    print(label+'...')
    files = os.listdir(audio_val_path + label + '/')
    files.sort()
    for file in tqdm(files):
        audio_file = audio_val_path + label + '/' + file
        csv_file = dest_val_path + label + '/' + file[:-4] + r".csv"
        
        cmd = common_cmd + '"' + audio_file + '"' + r" -O " + '"' + csv_file + '"'
        subprocess.call(cmd,shell=True)
        
        with open(csv_file, 'r') as f:
            r = csv.reader(f)
            for row in r:
                features = row[1:-1]  #Quitamos la primera y la última porque no contienen información
                
        os.remove(csv_file)
        features = np.array([float(x) for x in features])
        pd.DataFrame(features).to_csv(csv_file, header=None, index=None)
    
    

'''
################
##    TEST    ##
################
'''
print('Extracting audio features of the Test set...')

for label in labels:
    print(label+'...')
    files = os.listdir(audio_test_path + label + '/')
    files.sort()
    for file in tqdm(files):
        audio_file = audio_test_path + label + '/' + file
        csv_file = dest_test_path + label + '/' + file[:-4] + r".csv"
        
        cmd = common_cmd + '"' + audio_file + '"' + r" -O " + '"' + csv_file + '"'
        subprocess.call(cmd,shell=True)
        
        with open(csv_file, 'r') as f:
            r = csv.reader(f)
            for row in r:
                features = row[1:-1]  #Quitamos la primera y la última porque no contienen información
                
        os.remove(csv_file)
        features = np.array([float(x) for x in features])
        pd.DataFrame(features).to_csv(csv_file, header=None, index=None)
    
    
