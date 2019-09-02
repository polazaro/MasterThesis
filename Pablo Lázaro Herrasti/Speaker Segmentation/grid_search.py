# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:28:35 2019

@author: polaz
"""

import os
from scipy.io import wavfile
import numpy as np
import subprocess
import pandas as pd
import csv
import simpleder
from spectralcluster import SpectralClusterer

name_audio = 'Ses01F_script01_1'
dir_audio = 'D:/Data_Master/TFM/Segmentation/IEMOCAP_1/'
dir_audios = 'D:/Data_Master/TFM/Segmentation/Grid_search/'
dir_seg = dir_audios + name_audio + '/'
dir_time = dir_seg + name_audio + '.txt'

files = os.listdir(dir_seg)
audio_files = files[:-2]
txt_file = files[-2]

file = open(name_audio + '/' + txt_file, 'r')         
        
windows_t = [0.16,0.20,0.24,0.30,0.50]
overlaps = [0.3,0.5,0.70,0.90]

times = []
with open(dir_time, "r") as f:
    for line in f:
        times.append(line.strip())
        
real_s1 = []
with open(dir_audio+'speaker1.txt', "r") as f:
    for line in f:
        real_s1.append(line.strip().split('\t'))
        
real_s2 = []
with open(dir_audio+'speaker2.txt', "r") as f:
    for line in f:
        real_s2.append(line.strip().split('\t'))
        
ref = []
for s1 in real_s1:
    ref.append(('A',float(s1[0]),float(s1[1])))
for s2 in real_s2:
    ref.append(('B',float(s2[0]),float(s2[1])))
        
        
        
        
best_models = []
print('*******Running \n\n')
for window_t in windows_t:
    for overlap in overlaps:
        best_error = 10000000
        print('Window: ' + str(window_t) + ' and Overlap: ' +str(overlap))
        dir_segment = dir_seg + '/w' + str(window_t) + 'o' + str(overlap) + '/'
        os.mkdir(dir_segment)
        
        #### WINDOW SEGMENTATION
        print('Window Segmentation...')
        
        for segment in audio_files:
            control = True
            fs, signal = wavfile.read(dir_seg + segment)
            window_len = 0
            cont = 0
            os.mkdir(dir_segment + segment[:-4])
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
                wavfile.write(dir_segment + segment[:-4] + '/' + segment[:-4] + '_' + '{:04d}'.format(i) + '.wav',fs,output)
        
        
        #### FEATURE EXTRACTION
        print('Feature Extraction...')
        
        OpenSMILE_path = 'D:/Data_Master/TFM/opensmile-2.3.0/bin/Win32/SMILExtract_Release.exe'
        conf_file = 'D:/Data_Master/TFM/opensmile-2.3.0/config/IS13_ComParE.conf'
        
        common_cmd = OpenSMILE_path + " -C " + conf_file + " -I "
        
        folders = os.listdir(dir_segment)
            
        for folder in folders: 
            folder_dir = dir_segment + folder + '/'
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
                
                
        #### SPECTRAL CLUSTERING
        print('Spectral Clustering...')
        
        folders = os.listdir(dir_segment)
        order_pred = []
        order = []
        X = []
        for folder in folders: 
            folder_dir = dir_segment + folder + '/'
            files = os.listdir(folder_dir)
            csvs = [] 
            for f in files:
                if str.find(f,'wav')<0:
                    csvs.append(f)
            order.append(csvs)
            for f in csvs:
                data_path = folder_dir + f
                with open(data_path, 'r') as csvFile:
                    reader = csv.reader(csvFile)
                    data = []
                    for row in reader:
                        data.append(float(row[0]))
                X.append(data)
                
        order_list = [item for sublist in order for item in sublist]
        X_cluster = np.array([np.array(xi) for xi in X])
        
        p_percentiles = [0.99,0.95,0.90,0.85,0.8]
        gaussian_blur_sigmas = [1,2,3]
        thresholding_soft_multipliers = [0,0.01,0.02,0.05,0.1,0.5]
        stop_eigenvalues = [1e-4,1e-3,1e-2,1e-1]
        
        for p_percentile in p_percentiles:
            for gaussian_blur_sigma in gaussian_blur_sigmas:
                for thresholding_soft_multiplier in thresholding_soft_multipliers:
                    for stop_eigenvalue in stop_eigenvalues:
#                        print('  Values: p_percentile=' + str(p_percentile) + ' gaussian_blur_sigma=' + str(gaussian_blur_sigma) + ' thresholding_soft_multiplier=' + str(thresholding_soft_multiplier) + ' stop_eigenvalue=' + str(stop_eigenvalue))
                        clusterer = SpectralClusterer(
                            min_clusters=2,
                            max_clusters=10,
                            p_percentile=p_percentile,
                            gaussian_blur_sigma=gaussian_blur_sigma,
                            thresholding_soft_multiplier=thresholding_soft_multiplier,
                            stop_eigenvalue=stop_eigenvalue)
                        
                        labels = clusterer.predict(X_cluster)
                                
                        speaker_1 = []
                        speaker_2 = []
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
                        error = simpleder.DER(ref, hyp)
                        if error<best_error:
                            best_params = str(p_percentile)+'_'+str(gaussian_blur_sigma)+'_'+str(thresholding_soft_multiplier)+'_'+str(stop_eigenvalue)
                            best_error = error
                        
        print("             DER={:.3f}".format(best_error))
        best_models.append(('Window: ' + str(window_t*1000) + 'ms-Overlap: ' + str(overlap*100) +'%','Params: ' + best_params,'DER={:.3f}'.format(best_error)))
