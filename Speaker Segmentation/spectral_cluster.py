# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:57:42 2019

@author: polaz
"""

from spectralcluster import SpectralClusterer
import matplotlib.pyplot as plt
import os
import csv
import numpy as np

#### READING DATA

name_audio = 'Ses01F_script01_1'
dir_audio = 'C:/Users/polaz/Documents/Máster Data Science/Segundo Cuatrimestre/TFM/Code/Segmentation/' + name_audio + '/'
dir_txt = 'D:/Data_Master/TFM/Segmentation/IEMOCAP_1/'

files = os.listdir(dir_audio)
folders = []
for f in files:
    if str.find(f,'.')<0:
        folders.append(f)
        
real_labels = []
with open(dir_txt + 'segment_labels.txt', "r") as f:
    for line in f:
        real_labels.append(int(line.strip()))

order_pred = []
order = []
X = []
real_labels_seg = []
cont = 0
for folder in folders: 
    print(folder)
    folder_dir = dir_audio + folder + '/'
    files = os.listdir(folder_dir)
    csvs = [] 
    for f in files:
        if str.find(f,'wav')<0:
            csvs.append(f)
            real_labels_seg.append(real_labels[cont])
            
    order.append(csvs)
    for f in csvs:
        data_path = folder_dir + f
        with open(data_path, 'r') as csvFile:
            reader = csv.reader(csvFile)
            data = []
            for row in reader:
                data.append(float(row[0]))
        X.append(data)
    cont += 1
        
order_pred = [item for sublist in order for item in sublist]
X_cluster = np.array([np.array(xi) for xi in X])
        
        
#### SPECTRAL CLUSTER

#clusterer = SpectralClusterer(
#    min_clusters=2,
#    max_clusters=3,
#    p_percentile=0.99,
#    gaussian_blur_sigma=3,
#    thresholding_soft_multiplier=0.5,
#    stop_eigenvalue=10e-5)

clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=3,
    p_percentile=0.99,
    gaussian_blur_sigma=3,
    thresholding_soft_multiplier=0.5,
    stop_eigenvalue=10e-5)

labels, embeddings, centers = clusterer.predict(X_cluster)

#### SAVING RESULTS

np.savetxt(dir_audio + 'labels.npy', labels)
with open(dir_audio + 'order_pred.txt', "w") as f:
    for s in order_pred:
        f.write(str(s) +"\n")
        
#### PLOTTING CLUSTER

plt.figure(1)
plt.title('Spectral Cluster prediction')
plt.scatter(embeddings[:, 0], embeddings[:, 1],c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.ylabel('Embeddings 1') 
plt.xlabel('Embeddings 2')

#### REAL LABELS

plt.figure(2)
plt.title('Cluster real labels')
plt.scatter(embeddings[:, 0], embeddings[:, 1],c=real_labels_seg, s=50, cmap='viridis')
plt.ylabel('Embeddings 1') 
plt.xlabel('Embeddings 2')

