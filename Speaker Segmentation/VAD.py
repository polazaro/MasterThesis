# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:12:41 2019

@author: polaz
"""

import matplotlib.pyplot as plt
import wave
import numpy as np
import os
from scipy.io import wavfile
from pydub import AudioSegment
import subprocess

#### DIRECTORIES

dir_audio = 'D:/Data_Master/TFM/Segmentation/IEMOCAP_1/'
#dir_audio = 'D:/Data_Master/TFM/Segmentation/s3/video2/'
name_file = 'Ses01F_script01_1.wav'
#grid_search = 'D:/Data_Master/TFM/Segmentation/Grid_search/' + name_file[:-4]
grid_search = 'C:/Users/polaz/Documents/Máster Data Science/Segundo Cuatrimestre/TFM/Code/Segmentation/' + name_file[:-4]

#### CONVERTING STEREO TO MONO
os.mkdir(grid_search)
mysound = AudioSegment.from_wav(dir_audio + name_file)
mysound = mysound.set_channels(1)
mysound.export(grid_search + '/' + name_file, format="wav")
mysound.export()

#### COMPUTING VAD
fs, signal = wavfile.read(grid_search + '/' + name_file)

#50,62
utterances = subprocess.check_output("auditok -e 62 -i " + dir_audio + name_file + " -a 0.02 -m 20 -n 0.4 -r " + str(fs) + " -s 0.3", shell=True)
utterances = utterances.decode()
utterances = utterances.replace('\n','')
utterances = utterances.split('\r')

intervals = []
with open(grid_search + '/' + name_file[:-4] + '.txt', 'w') as f:
    for utt in utterances[:-1]:
        div = utt.split(' ')
        intervals.append((float(div[1]),float(div[2])))
        f.write("%s\t%s\n" % (div[1],div[2]))
        

#### PLOTTING RESULTS

spf = wave.open(dir_audio + name_file,'r')
        
fs2 = fs*2
signal2 = spf.readframes(-1)
signal2 = np.fromstring(signal2, 'Int16')
time=np.linspace(0, len(signal2)/fs2, num=len(signal2))

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(time,signal2)
plt.plot()
for inter in intervals:
    plt.axvspan(inter[0], inter[1], alpha=0.5, color='red')
plt.show()


### SAVING SEGMENTS

for i in range(len(intervals)):
    ini,final = intervals[i][0],intervals[i][1]
#    with open('segment' + str(i) + '.txt', 'w') as f:
    output = signal[int(ini*fs):int(final*fs)]
    wavfile.write(grid_search + '/segment' + '_{:04d}'.format(i+1) + '.wav',fs,output)
