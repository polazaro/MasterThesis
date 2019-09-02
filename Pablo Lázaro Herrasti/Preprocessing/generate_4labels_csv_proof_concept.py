# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 02:32:44 2019

@author: Rubén
"""

import pandas as pd
import os

faces_path  = 'C:/Users/ruben/Desktop/ProofConcept/Faces/'
audio_path  = 'C:/Users/ruben/Desktop/ProofConcept/Audios/'
feat_path   = 'C:/Users/ruben/Desktop/ProofConcept/Features/'


ind_to_label = {
        0: "hap",
        1: "sad",
        2: "neu",
        3: "ang"
        }

label_to_ind = {
        "hap": 0,
        "sad": 1,
        "neu": 2,
        "ang": 3
        }

frame_rate = 25

print('Building F1_talking files...')
with open('C:/Users/ruben/Desktop/ProofConcept/LabelsA.txt', 'r') as f:
    labels_1 = f.readlines()
    
labels_1 = [label_to_ind[label[:-1]] for label in labels_1]

with open('C:/Users/ruben/Desktop/ProofConcept/TimeA.txt', 'r') as f:
    times1 = f.readlines()
    

F1_faces = []
F1_audio = []
F1_feat  = []
ok_utt = 0
wrong_utt = 0
for i in range(len(times1)):
    init_time, final_time = times1[i][:-1].split('\t')
    init_frame = round(float(init_time)*frame_rate)
    final_frame = round(float(final_time)*frame_rate)
    
    frames_exist = [os.path.exists(faces_path + 'F1_talking/F1_talking_{:06d}'.format(j) + '.bmp') for j in range(init_frame, final_frame+1)]
    
    if frames_exist.count(False) == 0:
        for j in range(init_frame, final_frame+1):
            frame = 'F1_talking/F1_talking_{:06d}'.format(j) + '.bmp'
            audio = 'F1_talking/F1_talking_{:06d}'.format(j) + '.wav'
            feat  = 'F1_talking/F1_talking_{:06d}'.format(j) + '.csv'
            F1_faces.append([frame, labels_1[i]])
            F1_audio.append([audio, labels_1[i]])
            F1_feat.append([feat, labels_1[i]])
            
        ok_utt += 1
    else:
        wrong_utt += 1
    
df_F1_faces = pd.DataFrame(F1_faces, columns=['im_name','label'])
df_F1_faces.to_csv(faces_path+'F1_faces.csv', index=False)
df_F1_audio = pd.DataFrame(F1_audio, columns=['audio_name','label'])
df_F1_audio.to_csv(audio_path+'F1_audios.csv', index=False)
df_F1_feat  = pd.DataFrame(F1_feat, columns=['csv_name','label'])
df_F1_feat.to_csv(feat_path+'F1_feat.csv', index=False)
print('F1_talking files finished.\n')
    

print('Building F2_talking files...')
with open('C:/Users/ruben/Desktop/ProofConcept/LabelsB.txt', 'r') as f:
    labels_2 = f.readlines()
    
labels_2 = [label_to_ind[label[:-1]] for label in labels_2]

with open('C:/Users/ruben/Desktop/ProofConcept/TimeB.txt', 'r') as f:
    times2 = f.readlines()
    
F2_faces = []
F2_audio = []
F2_feat  = []
for i in range(len(times2)):
    init_time, final_time = times2[i][:-1].split('\t')
    init_frame = round(float(init_time)*frame_rate)
    final_frame = round(float(final_time)*frame_rate)
    
    frames_exist = [os.path.exists(faces_path + 'F2_talking/F2_talking_{:06d}'.format(j) + '.bmp') for j in range(init_frame, final_frame+1)]
    
    if frames_exist.count(False) == 0:
        for j in range(init_frame, final_frame+1):
            frame = 'F2_talking/F2_talking_{:06d}'.format(j) + '.bmp'
            audio = 'F2_talking/F2_talking_{:06d}'.format(j) + '.wav'
            feat  = 'F2_talking/F2_talking_{:06d}'.format(j) + '.csv'
            F2_faces.append([frame, labels_2[i]])
            F2_audio.append([audio, labels_2[i]])
            F2_feat.append([feat, labels_2[i]])
            
        ok_utt += 1
    else:
        wrong_utt += 1
        
df_F2_faces = pd.DataFrame(F2_faces, columns=['im_name','label'])
df_F2_faces.to_csv(faces_path+'F2_faces.csv', index=False)
df_F2_audio = pd.DataFrame(F2_audio, columns=['audio_name','label'])
df_F2_audio.to_csv(audio_path+'F2_audios.csv', index=False)
df_F2_feat  = pd.DataFrame(F2_feat, columns=['csv_name','label'])
df_F2_feat.to_csv(feat_path+'F2_feat.csv', index=False)
print('F2_talking files finished.\n')       