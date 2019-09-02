# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:43:14 2019

@author: Rubén
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

''' My computer paths '''
audio_path_dest_1 = r"C:/Users/ruben/Desktop/ProofConcept/Features/F1_talking/"
audio_path_dest_2 = r"C:/Users/ruben/Desktop/ProofConcept/Features/F2_talking/"

max_csv = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/features_max.csv'
min_csv = 'C:/Users/ruben/Documents/Máster Data Science/2º Cuatrimestre/Master Thesis/Database/IEMOCAP_audio_2_features_100/features_min.csv'
''' This first part is to check if there are wrong files '''

print('Checking if there are broken files...')
broken_files = 0
bf_list = []
files = os.listdir(audio_path_dest_1)
files.sort()
for file in tqdm(files):
    csv_file = audio_path_dest_1 + file
    try:
        features = pd.read_csv(csv_file, header=None)
    except:
        bf_list.append(csv_file)
        broken_files += 1

files = os.listdir(audio_path_dest_2)
files.sort()
for file in tqdm(files):
    csv_file = audio_path_dest_2 + file
    try:
        features = pd.read_csv(csv_file, header=None)
    except:
        bf_list.append(csv_file)
        broken_files += 1


''' In this third part we normalize all the feature vectors '''

print('Normalizing feature vectors...')

maximo = pd.read_csv(max_csv, header=None).values
minimo = pd.read_csv(min_csv, header=None).values
n_features = 6373

comp = maximo != minimo  ## Quitamos las features que siempre tienen el mismo valor: maximo y mínimo son iguales, porque no aportan información.
n_norm_features = sum(comp)[0]


files = os.listdir(audio_path_dest_1)
files.sort()
for file in tqdm(files):
    # Load previous features
    norm_features = np.zeros((n_norm_features,1))
    csv_file = audio_path_dest_1 + file
    features = pd.read_csv(csv_file, header=None).values
    
    # Compute new normaliza features
    cont = 0
    for i in range(n_features):
        if maximo[i] != minimo[i]:
            norm_features[cont] = (features[i]-minimo[i])/(maximo[i]-minimo[i])
            cont += 1
    
    # Delete previous file and save new one
    os.remove(csv_file)
    pd.DataFrame(norm_features).to_csv(csv_file, header=None, index=None)
    
files = os.listdir(audio_path_dest_2)
files.sort()
for file in tqdm(files):
    # Load previous features
    norm_features = np.zeros((n_norm_features,1))
    csv_file = audio_path_dest_2 + file
    features = pd.read_csv(csv_file, header=None).values
    
    # Compute new normaliza features
    cont = 0
    for i in range(n_features):
        if maximo[i] != minimo[i]:
            norm_features[cont] = (features[i]-minimo[i])/(maximo[i]-minimo[i])
            cont += 1
    
    # Delete previous file and save new one
    os.remove(csv_file)
    pd.DataFrame(norm_features).to_csv(csv_file, header=None, index=None)