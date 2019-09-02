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

window_t = 0.160
overlap = 0.3   

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
        

#plt.figure(1)
#plt.title('Speaker1 GT')
#plt.plot(time,signal)
#plt.plot()
#for s1 in real_s1:
#    plt.axvspan(float(s1[0]), float(s1[1]), alpha=0.5, color='red')
#plt.show()
#
#plt.figure(2)
#plt.title('Speaker2 GT')
#plt.plot(time,signal)
#plt.plot()
#for s2 in real_s2:
#    plt.axvspan(float(s2[0]), float(s2[1]), alpha=0.5, color='yellow')
#plt.show()  
#        
#        
##### PLOTTING RESULTS
#
#
#plt.figure(3)
#plt.title('Speaker1')
#plt.plot(time,signal)
#plt.plot()
#for s1 in speaker_1:
#    plt.axvspan(s1[0], s1[1]-window/2, alpha=0.5, color='red')
#plt.show()
#
#plt.figure(4)
#plt.title('Speaker2')
#plt.plot(time,signal)
#plt.plot()
#for s2 in speaker_2:
#    plt.axvspan(s2[0], s2[1]-window/2, alpha=0.5, color='yellow')
#plt.show()


#### COMPUTING REAL SEGMENTS AND DER

speaker = []

for i in range(len(labels)):
    pos = int(order_list[i][8:12])-1
    num = int(order_list[i][13:17])-1
    ini,final = times[pos].split('\t')
    if round(float(ini)+((num)*(window_t * (1-overlap))+window_t),4)<float(final):
        interval = [round(float(ini)+num*(window_t * (1-overlap)),4),round(float(ini)+((num)*(window_t * (1-overlap))+window_t),4)]
    else:
        interval = [round(float(ini)+num*(window_t * (1-overlap)),4),float(final)]
    speaker.append(interval)
        

#### COMPUTING REAL SEGMENTS AND DER

hyp = []

for i in range(len(speaker)):
    if len(speaker) == i+1:
        pass
    else:
        if speaker[i][1] <= speaker[i+1][0]:
            hyp.append((str(labels[i]+1),speaker[i][0],round(speaker[i][1],4)))
        else:
            hyp.append((str(labels[i]+1),speaker[i][0],round(speaker[i][1]-(overlap)*window_t,4)))

            
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
    
            
            
        
    
        
    

