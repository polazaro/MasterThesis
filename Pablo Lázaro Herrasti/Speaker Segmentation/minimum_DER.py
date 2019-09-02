# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 23:26:04 2019

@author: polaz
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import simpleder

name_audio = 'Ses01F_script01_1'
dir_audio = 'C:/Users/polaz/Documents/Máster Data Science/Segundo Cuatrimestre/TFM/Code/Segmentation/' + name_audio + '/'
dir_txt = 'D:/Data_Master/TFM/Segmentation/IEMOCAP_1/'
#### LOADING DATA

dir_list = dir_audio + 'order_pred.txt'
dir_label = dir_audio + 'labels.npy'
dir_time = dir_audio + name_audio + '.txt'


order_list = []
with open(dir_list, "r") as f:
    for line in f:
        order_list.append(line.strip())
        
labels = np.loadtxt(dir_label, dtype=int)

times = []
with open(dir_time, "r") as f:
    for line in f:
        times.append(line.strip())
        
speaker_1 = []
speaker_2 = []

window_t = 0.24
overlap = 0.5        

fs, signal = wavfile.read(dir_audio + '/' + name_audio +'.wav')
time=np.linspace(0, len(signal)/fs, num=len(signal))
#### PLOTTING GROUND TRUTH

real_s1 = []
with open(dir_txt+'speaker1.txt', "r") as f:
    for line in f:
        real_s1.append(line.strip().split('\t'))
        
real_s2 = []
with open(dir_txt+'speaker2.txt', "r") as f:
    for line in f:
        real_s2.append(line.strip().split('\t'))
        

plt.figure(1)
plt.title('Speaker1 GT')
plt.plot(time,signal)
plt.plot()
for s1 in real_s1:
    plt.axvspan(float(s1[0]), float(s1[1]), alpha=0.5, color='red')
plt.show()

plt.figure(2)
plt.title('Speaker2 GT')
plt.plot(time,signal)
plt.plot()
for s2 in real_s2:
    plt.axvspan(float(s2[0]), float(s2[1]), alpha=0.5, color='yellow')
plt.show()  
#        
#        


#### COMPUTING REAL SEGMENTS AND DER

new_labels = np.zeros((len(times)))
speaker_1 = 0
speaker_2 = 0
pos_ant = 0
for i in range(len(order_list)):
    pos = int(order_list[i][8:12])-1
    if i+1 == len(order_list):
        pos_next = -1
    else:
        pos_next = int(order_list[i+1][8:12])-1
    if labels[pos] == 0:
        speaker_1 += 1
    else:
        speaker_2 += 1
    if pos != pos_next:
        if speaker_1>speaker_2:
            new_labels[pos] = 1
        else:
            new_labels[pos] = 2
        speaker_1 = 0
        speaker_2 = 0
        
        

#### COMPUTING REAL SEGMENTS AND DER


hyp = []
real_labels = [1,1,1,1,2,1,1,2,1,2,2,2,2,2,1,1,1,2,1,1,2,2,2,1,2,2,1,2,2,2,1]

for i in range(len(new_labels)):
    ini,final = times[i].split('\t')
    hyp.append((str(int(real_labels[i])),float(ini),float(final)))

            
ref = []
for s1 in real_s1:
    ref.append(('A',float(s1[0]),float(s1[1])))
for s2 in real_s2:
    ref.append(('B',float(s2[0]),float(s2[1])))
            

error = simpleder.DER(ref, hyp)

print('************************')
print('************************')
print('************************')
print("DER={:.3f}".format(error))


#### PLOTTING RESULTS


plt.figure(3)
plt.title('Speaker1 predicted')
plt.plot(time,signal)
plt.plot()
for i in range(len(new_labels)):
    ini,final = times[i].split('\t')
    if real_labels[i] == 1:
        plt.axvspan(float(ini), float(final), alpha=0.5, color='red')
plt.show()

plt.figure(4)
plt.title('Speaker2 predicted')
plt.plot(time,signal)
plt.plot()
for i in range(len(new_labels)):
    ini,final = times[i].split('\t')
    if real_labels[i] == 2:
        plt.axvspan(float(ini), float(final), alpha=0.5, color='yellow')
plt.show()
    