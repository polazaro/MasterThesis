# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:27:50 2019

@author: Rubén
"""

import subprocess
import pandas as pd
import csv
import os
import numpy as np
from tqdm import tqdm

OpenSMILE_path    = r"C:/Users/ruben/Downloads/opensmile-2.3.0/bin/Win32/SMILExtract_Release.exe"
conf_file         = r"C:/Users/ruben/Downloads/opensmile-2.3.0/config/IS13_ComParE.conf"
audio_path_1      = r"C:/Users/ruben/Desktop/ProofConcept/Audios/F1_talking/"
audio_path_2      = r"C:/Users/ruben/Desktop/ProofConcept/Audios/F2_talking/"
audio_path_dest_1 = r"C:/Users/ruben/Desktop/ProofConcept/Features/F1_talking/"
audio_path_dest_2 = r"C:/Users/ruben/Desktop/ProofConcept/Features/F2_talking/"

common_cmd = OpenSMILE_path + r" -C " + conf_file + r" -I "

print('Extracting audio features of F1_talking...')

files = os.listdir(audio_path_1)
files.sort()
for file in tqdm(files):
    audio_file = audio_path_1 + file
    csv_file = audio_path_dest_1 + file[:-4] + r".csv"
    
    cmd = common_cmd + '"' + audio_file + '"' + r" -O " + '"' + csv_file + '"'
    subprocess.call(cmd,shell=True)
    
    with open(csv_file, 'r') as f:
        r = csv.reader(f)
        for row in r:
            features = row[1:-1]  #Quitamos la primera y la última porque no contienen información
            
    os.remove(csv_file)
    features = np.array([float(x) for x in features])
    pd.DataFrame(features).to_csv(csv_file, header=None, index=None)
    
print('Extracting audio features of F2_talking...')

files = os.listdir(audio_path_2)
files.sort()
for file in tqdm(files):
    audio_file = audio_path_2 + file
    csv_file = audio_path_dest_2 + file[:-4] + r".csv"
    
    cmd = common_cmd + '"' + audio_file + '"' + r" -O " + '"' + csv_file + '"'
    subprocess.call(cmd,shell=True)
    
    with open(csv_file, 'r') as f:
        r = csv.reader(f)
        for row in r:
            features = row[1:-1]  #Quitamos la primera y la última porque no contienen información
            
    os.remove(csv_file)
    features = np.array([float(x) for x in features])
    pd.DataFrame(features).to_csv(csv_file, header=None, index=None)