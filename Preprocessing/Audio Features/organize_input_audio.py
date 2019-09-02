# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 02:13:04 2019

@author: Rubén
"""

###############################################################################
##                                                                           ##
## In this script we read the names of the frames from input_data.txt, and   ##
## we organize the audios in folders. We are going to create one folder per  ##
## label and put there all the audios with that label.                       ##
##                                                                           ##
## We are following this schedule because it is clearer to see which of the  ##
## audios belong to each of the labels and then we can create the csv with   ##
## the names of the audios and the labels faster.                            ##
##                                                                           ##
###############################################################################

import os
import numpy as np
import shutil
from collections import OrderedDict

#from tqdm import tqdm


# Initial variables and dictionaries
frame_rate = 29.97
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

# Useful paths
txt_path            = '/app/data/IEMOCAP_dat1/Utterances_txt_faces_2/Labels_4/'
database_path       = '/app/data/IEMOCAP_dat1/IEMOCAP_faces_final_2/'
audio_database_path = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_mono_100/'
dest_path           = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/'

'''
################
##   TRAIN    ##
################
'''

labels = []
ok_audios = 0
ok_utt = 0
wrong_utt = 0

print('Moving training audios...')

for i in range(1,5):
    
    train_utt_txt = os.listdir(txt_path + 'train/' + 'Session{}'.format(i) + '/')
    train_utt_txt.sort()
    print('...Session{}...'.format(i))
   
    for utt_txt in train_utt_txt:
       
        print(utt_txt[:-4])
       
        with open(txt_path + 'train/' + 'Session{}'.format(i) + '/' + utt_txt,"r") as f:
            data = f.readlines()
           
        data = data[1:]
       
        video_path = database_path + 'Session{}'.format(i) + '/' + utt_txt[:-4] + '/'  #video folder with all the faces for that video
        audio_path = audio_database_path + 'Session{}'.format(i) + '/' + utt_txt[:-4] + '/'  #audio folder witg all the audio-segments for that video
        all_frames = os.listdir(video_path)
        all_frames.sort()
        n_frames = int(all_frames[-1][-10:-4])
       
        for line in data:  ## Each line is an utterance
           
            utt = line[:-1].split('\t')
           
            label = label_to_ind[utt[1]]
            init_time = float(utt[2])
            final_time = float(utt[3])
           
            init_frame = round(init_time*frame_rate)
            final_frame = round(final_time*frame_rate)
           
            if final_frame > n_frames:  # This if is because in some cases the number of the last utterance we obtain multiplying the final time times the frame rate is bigger than the number of frames that we have obtained from the openface
                end_frame = n_frames
            else:
                end_frame = final_frame
                        
            frames_exist = [os.path.exists(video_path + utt_txt[:-4] + '_{:06d}'.format(j) + '.bmp') for j in range(init_frame, end_frame+1)]
           
            if frames_exist.count(False) == 0:
               
                for j in range(init_frame, end_frame+1):
                   
                    audio = audio_path + utt_txt[:-4] + '_{:06d}'.format(j) + '.wav'
                    shutil.copy(audio, dest_path + 'train/' + utt[1])
                    ok_audios += 1
                    labels.append(utt[1])
                   
                ok_utt +=1
               
            else:
               
                wrong_utt += 1
                               
with open(dest_path + 'summary_train.txt', 'w') as f:   
    f.write('SESSIONS 1-4 OF TRAINING SET DONE!\n\n')
    f.write('Number of audios: '+str(ok_audios)+'\n')
    f.write('Number of ok utterances: '+str(ok_utt)+'\n')
    f.write('Number of wrong utterances: '+str(wrong_utt)+'\n'+'\n')
   
    f.write('Number of audios in:'+'\n')
    f.write('ang: '+str(labels.count("ang"))+'\n')
    f.write('hap: '+str(labels.count("hap"))+'\n')
    f.write('neu: '+str(labels.count("neu"))+'\n')
    f.write('sad: '+str(labels.count("sad"))+'\n')
    


 
'''
################
## VALIDATION ##
################
'''

labels = []
ok_audios = 0
ok_utt = 0
wrong_utt = 0

print('Moving validation audios...')

for i in range(1,5):
    
    train_utt_txt = os.listdir(txt_path + 'validation/' + 'Session{}'.format(i) + '/')
    train_utt_txt.sort()
    print('...Session{}...'.format(i))
   
    for utt_txt in train_utt_txt:
       
        print(utt_txt[:-4])
       
        with open(txt_path + 'validation/' + 'Session{}'.format(i) + '/' + utt_txt,"r") as f:
            data = f.readlines()
           
        data = data[1:]
       
        video_path = database_path + 'Session{}'.format(i) + '/' + utt_txt[:-4] + '/'  #video folder with all the faces for that video
        audio_path = audio_database_path + 'Session{}'.format(i) + '/' + utt_txt[:-4] + '/'  #audio folder witg all the audio-segments for that video
        all_frames = os.listdir(video_path)
        all_frames.sort()
        n_frames = int(all_frames[-1][-10:-4])
       
        for line in data:  ## Each line is an utterance
           
            utt = line[:-1].split('\t')
           
            label = label_to_ind[utt[1]]
            init_time = float(utt[2])
            final_time = float(utt[3])
           
            init_frame = round(init_time*frame_rate)
            final_frame = round(final_time*frame_rate)
           
            if final_frame > n_frames:  # This if is because in some cases the number of the last utterance we obtain multiplying the final time times the frame rate is bigger than the number of frames that we have obtained from the openface
                end_frame = n_frames
            else:
                end_frame = final_frame
                        
            frames_exist = [os.path.exists(video_path + utt_txt[:-4] + '_{:06d}'.format(j) + '.bmp') for j in range(init_frame, end_frame+1)]
           
            if frames_exist.count(False) == 0:
               
                for j in range(init_frame, end_frame+1):
                   
                    audio = audio_path + utt_txt[:-4] + '_{:06d}'.format(j) + '.wav'
                    shutil.copy(audio, dest_path + 'validation/' + utt[1])
                    ok_audios += 1
                    labels.append(utt[1])
                   
                ok_utt +=1
               
            else:
               
                wrong_utt += 1
                               
with open(dest_path + 'summary_val.txt', 'w') as f:   
    f.write('SESSIONS 1-4 OF VALIDATION SET DONE!\n\n')
    f.write('Number of audios: '+str(ok_audios)+'\n')
    f.write('Number of ok utterances: '+str(ok_utt)+'\n')
    f.write('Number of wrong utterances: '+str(wrong_utt)+'\n'+'\n')
   
    f.write('Number of audios in:'+'\n')
    f.write('ang: '+str(labels.count("ang"))+'\n')
    f.write('hap: '+str(labels.count("hap"))+'\n')
    f.write('neu: '+str(labels.count("neu"))+'\n')
    f.write('sad: '+str(labels.count("sad"))+'\n')
    
    
   

'''
################
##    TEST    ##
################
'''

labels = []
ok_audios = 0
ok_utt = 0
wrong_utt = 0

print('Moving test audios...')

i = 5

train_utt_txt = os.listdir(txt_path + 'test/' + 'Session{}'.format(i) + '/')
train_utt_txt.sort()
print('...Session{}...'.format(i))

for utt_txt in train_utt_txt:
    
    print(utt_txt[:-4])
    
    with open(txt_path + 'test/' + 'Session{}'.format(i) + '/' + utt_txt,"r") as f:
        data = f.readlines()
        
    data = data[1:]
    
    video_path = database_path + 'Session{}'.format(i) + '/' + utt_txt[:-4] + '/'  #video folder with all the faces for that video
    audio_path = audio_database_path + 'Session{}'.format(i) + '/' + utt_txt[:-4] + '/'  #audio folder witg all the audio-segments for that video
    all_frames = os.listdir(video_path)
    all_frames.sort()
    n_frames = int(all_frames[-1][-10:-4])
    
    for line in data:  ## Each line is an utterance
        
        utt = line[:-1].split('\t')
        
        label = label_to_ind[utt[1]]
        init_time = float(utt[2])
        final_time = float(utt[3])
        
        init_frame = round(init_time*frame_rate)
        final_frame = round(final_time*frame_rate)
        
        if final_frame > n_frames:  # This if is because in some cases the number of the last utterance we obtain multiplying the final time times the frame rate is bigger than the number of frames that we have obtained from the openface
            end_frame = n_frames
        else:
            end_frame = final_frame
                     
        frames_exist = [os.path.exists(video_path + utt_txt[:-4] + '_{:06d}'.format(j) + '.bmp') for j in range(init_frame, end_frame+1)]
        
        if frames_exist.count(False) == 0:
            
            for j in range(init_frame, end_frame+1):
                
                audio = audio_path + utt_txt[:-4] + '_{:06d}'.format(j) + '.wav'
                shutil.copy(audio, dest_path + 'test/' + utt[1])
                ok_audios += 1
                labels.append(utt[1])
                
            ok_utt +=1
            
        else:
            
            wrong_utt += 1
                                

with open(dest_path + 'summary_test.txt', 'w') as f:   
    f.write('SESSION 5 OF TEST SET DONE!\n\n')
    f.write('Number of audios: '+str(ok_audios)+'\n')
    f.write('Number of ok utterances: '+str(ok_utt)+'\n')
    f.write('Number of wrong utterances: '+str(wrong_utt)+'\n'+'\n')
   
    f.write('Number of audios in:'+'\n')
    f.write('ang: '+str(labels.count("ang"))+'\n')
    f.write('hap: '+str(labels.count("hap"))+'\n')
    f.write('neu: '+str(labels.count("neu"))+'\n')
    f.write('sad: '+str(labels.count("sad"))+'\n')  
    
    
    
    
    

''' Auxiliary code that is not needed rigth now: '''
#print('PREPARING TRAIN SET')
#
#with open(txt_path+'train_data.txt',"r") as f:
#    input_data = f.readlines()
#    
#input_data = list(set(input_data)) # Because there are about 100 frames replicated
#input_data.sort()
#print('Creating training dict...')
#
#train_dict = OrderedDict()
#for line in tqdm(input_data):
#    f_name, _, _, label = line[:-1].split('\t')
#    train_dict[f_name] = ind_to_label[label]
#
#print('Training dict done.')
    
# This lines go in line 104 (first line after for j in... loop)
#                    search_in_dict = 'Session{}'.format(i) + utt_txt[:-4] + '/' + utt_txt[:-4] + '_{:06d}'.format(j) + '.bmp'
#                    label = train_dict[search_in_dict]