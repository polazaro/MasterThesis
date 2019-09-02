# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:43:13 2019

@author: Rubén
"""

###############################################################################
##                                                                           ##
## In this script we create dictionaries in which each key is the number of  ##
## the utterance and its value is the list with the names of the files that  ##
## belong to that utterance.                                                 ##
##                                                                           ##
###############################################################################

import os
from collections import OrderedDict

dest_path     = '/app/data/IEMOCAP_dat1/Utterances_dict_2/'
train_path    = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_66/4Labels/train/'
val_path      = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_66/4Labels/validation/'
test_path     = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_66/4Labels/test/'
txt_path      = '/app/data/IEMOCAP_dat1/Utterances_txt_faces_2/Labels_4/'
database_path = '/app/data/IEMOCAP_dat1/IEMOCAP_faces_final_2/'

labels = ['hap', 'sad', 'neu', 'ang']
frame_rate = 29.97
label_to_ind = {
        "hap": 0,
        "sad": 1,
        "neu": 2,
        "ang": 3
        }

'''
################
##   TRAIN    ##
################
'''

''' First we create an ordered dictionary with all the faces/audios we have '''

print('Creating files-training-dictionary...')
train_dict = OrderedDict()
for i, label in enumerate(labels):
    directory = train_path + label + '/'
    files = os.listdir(directory)
    files.sort()
    for file in files:
        train_dict[(file,i)] = i
        
print('Length of files-training-dictionary: {}'.format(len(train_dict)))
print('')
print('Creating training dictionaries...')

train_hap_dict = OrderedDict()
train_sad_dict = OrderedDict()
train_neu_dict = OrderedDict()
train_ang_dict = OrderedDict()

cont_utt_hap = 0
cont_utt_sad = 0
cont_utt_neu = 0
cont_utt_ang = 0

cont_files_hap = 0
cont_files_sad = 0
cont_files_neu = 0
cont_files_ang = 0

ok_utt    = 0
wrong_utt = 0

for i in range(1,5):
   
    train_utt_txt = os.listdir(txt_path + 'train/' + 'Session{}'.format(i) + '/')
    train_utt_txt.sort()
    print('\t...Session{}...'.format(i))
    
    for utt_txt in train_utt_txt:
       
        with open(txt_path + 'train/' + 'Session{}'.format(i) + '/' + utt_txt,"r") as f:
            data = f.readlines()
           
        data = data[1:]
       
        video_path = database_path + 'Session{}'.format(i) + '/' + utt_txt[:-4] + '/'  #video folder with all the faces for that video
        all_frames = os.listdir(video_path)
        all_frames.sort()
        n_frames = int(all_frames[-1][-10:-4])
       
        for line in data:  ## Each line is an utterance
           
            audio_files_list = []
            
            utt = line[:-1].split('\t')
           
            label = utt[1]
            label_int = label_to_ind[utt[1]]
            init_time = float(utt[2])
            final_time = float(utt[3])
           
            init_frame = round(init_time*frame_rate)
            final_frame = round(final_time*frame_rate)
           
            if final_frame > n_frames:  # This if is because in some cases the number of the last utterance we obtain multiplying the final time times the frame rate is bigger than the number of frames that we have obtained from the openface
                end_frame = n_frames
            else:
                end_frame = final_frame
                        
            frames_exist = [os.path.exists(video_path + utt_txt[:-4] + '_{:06d}'.format(j) + '.bmp') for j in range(init_frame, end_frame+1)]
           
            if frames_exist.count(False) == 0 and len(frames_exist)>0:
               
                for j in range(init_frame, end_frame+1):
                   
                    audio = utt_txt[:-4] + '_{:06d}'.format(j) + '.wav'
                    if (audio, label_int) in train_dict:
                        audio_files_list.append(audio[:-4])
                        aux = train_dict.pop((audio, label_int))
                
                audio_files_list.sort()
                
                if label == 'hap':
                    cont_utt_hap += 1
                    train_hap_dict['Utterance{:06d}'.format(cont_utt_hap)] = audio_files_list
                    cont_files_hap += len(audio_files_list)
                elif label == 'sad':
                    cont_utt_sad += 1
                    train_sad_dict['Utterance{:06d}'.format(cont_utt_sad)] = audio_files_list
                    cont_files_sad += len(audio_files_list)
                elif label == 'neu':
                    cont_utt_neu += 1
                    train_neu_dict['Utterance{:06d}'.format(cont_utt_neu)] = audio_files_list
                    cont_files_neu += len(audio_files_list)
                elif label == 'ang':
                    cont_utt_ang += 1
                    train_ang_dict['Utterance{:06d}'.format(cont_utt_ang)] = audio_files_list
                    cont_files_ang += len(audio_files_list) 
                   
                ok_utt +=1
               
            else:
               
                wrong_utt += 1

print('Training dictionaries done...')
print('Saving training summary...')
with open(dest_path + 'summary_train.txt', 'w') as f:   
    f.write('SUMMARY OF THE TRAINING SET:\n\n')
    f.write('Number of ok utterances: '+str(ok_utt)+'\n')
    f.write('Number of wrong utterances: '+str(wrong_utt)+'\n'+'\n')
    
    f.write('Number of utterances per label:'+'\n')
    f.write('ang: '+str(cont_utt_ang)+'\n')
    f.write('hap: '+str(cont_utt_hap)+'\n')
    f.write('neu: '+str(cont_utt_neu)+'\n')
    f.write('sad: '+str(cont_utt_sad)+'\n\n')
    
    f.write('Number of files per label:'+'\n')
    f.write('ang: '+str(cont_files_ang)+'\n')
    f.write('hap: '+str(cont_files_hap)+'\n')
    f.write('neu: '+str(cont_files_neu)+'\n')
    f.write('sad: '+str(cont_files_sad)+'\n\n')

print('Saving training dictionaries...')
with open(dest_path+'train_hap_dict.csv', 'w') as f:
    for key, value in train_hap_dict.items():
        f.write('%s:%s\n' % (key, value))

with open(dest_path+'train_sad_dict.csv', 'w') as f:
    for key, value in train_sad_dict.items():
        f.write('%s:%s\n' % (key, value))

with open(dest_path+'train_neu_dict.csv', 'w') as f:
    for key, value in train_neu_dict.items():
        f.write('%s:%s\n' % (key, value))

with open(dest_path+'train_ang_dict.csv', 'w') as f:
    for key, value in train_ang_dict.items():
        f.write('%s:%s\n' % (key, value))

print('TRAINING DICTIONARIES AND SUMMARY DONE!\n\n')
        
        

'''
################
## VALIDATION ##
################
'''

''' First we create an ordered dictionary with all the faces/audios we have '''

print('Creating files-validation-dictionary...')
val_dict = OrderedDict()
for i, label in enumerate(labels):
    directory = val_path + label + '/'
    files = os.listdir(directory)
    files.sort()
    for file in files:
        val_dict[(file,i)] = i
        
print('Length of files-validation-dictionary: {}'.format(len(val_dict)))
print('')
print('Creating validation dictionaries...')

val_hap_dict = OrderedDict()
val_sad_dict = OrderedDict()
val_neu_dict = OrderedDict()
val_ang_dict = OrderedDict()

cont_utt_hap = 0
cont_utt_sad = 0
cont_utt_neu = 0
cont_utt_ang = 0

cont_files_hap = 0
cont_files_sad = 0
cont_files_neu = 0
cont_files_ang = 0

ok_utt    = 0
wrong_utt = 0

for i in range(1,5):
   
    val_utt_txt = os.listdir(txt_path + 'validation/' + 'Session{}'.format(i) + '/')
    val_utt_txt.sort()
    print('\t...Session{}...'.format(i))
    
    for utt_txt in val_utt_txt:
       
        with open(txt_path + 'validation/' + 'Session{}'.format(i) + '/' + utt_txt,"r") as f:
            data = f.readlines()
           
        data = data[1:]
       
        video_path = database_path + 'Session{}'.format(i) + '/' + utt_txt[:-4] + '/'  #video folder with all the faces for that video
        all_frames = os.listdir(video_path)
        all_frames.sort()
        n_frames = int(all_frames[-1][-10:-4])
       
        for line in data:  ## Each line is an utterance
           
            audio_files_list = []
            
            utt = line[:-1].split('\t')
           
            label = utt[1]
            label_int = label_to_ind[utt[1]]
            init_time = float(utt[2])
            final_time = float(utt[3])
           
            init_frame = round(init_time*frame_rate)
            final_frame = round(final_time*frame_rate)
           
            if final_frame > n_frames:  # This if is because in some cases the number of the last utterance we obtain multiplying the final time times the frame rate is bigger than the number of frames that we have obtained from the openface
                end_frame = n_frames
            else:
                end_frame = final_frame
                        
            frames_exist = [os.path.exists(video_path + utt_txt[:-4] + '_{:06d}'.format(j) + '.bmp') for j in range(init_frame, end_frame+1)]
           
            if frames_exist.count(False) == 0 and len(frames_exist)>0:
               
                for j in range(init_frame, end_frame+1):
                   
                    audio = utt_txt[:-4] + '_{:06d}'.format(j) + '.wav'
                    if (audio, label_int) in val_dict:
                        audio_files_list.append(audio[:-4])
                        aux = val_dict.pop((audio, label_int))
                        
                audio_files_list.sort()
                    
                if label == 'hap':
                    cont_utt_hap += 1
                    val_hap_dict['Utterance{:06d}'.format(cont_utt_hap)] = audio_files_list
                    cont_files_hap += len(audio_files_list)
                elif label == 'sad':
                    cont_utt_sad += 1
                    val_sad_dict['Utterance{:06d}'.format(cont_utt_sad)] = audio_files_list
                    cont_files_sad += len(audio_files_list)
                elif label == 'neu':
                    cont_utt_neu += 1
                    val_neu_dict['Utterance{:06d}'.format(cont_utt_neu)] = audio_files_list
                    cont_files_neu += len(audio_files_list)
                elif label == 'ang':
                    cont_utt_ang += 1
                    val_ang_dict['Utterance{:06d}'.format(cont_utt_ang)] = audio_files_list
                    cont_files_ang += len(audio_files_list) 
                   
                ok_utt +=1
               
            else:
               
                wrong_utt += 1

print('Validation dictionaries done...')
print('Saving validation summary...')
with open(dest_path + 'summary_val.txt', 'w') as f:   
    f.write('SUMMARY OF THE VALIDATION SET:\n\n')
    f.write('Number of ok utterances: '+str(ok_utt)+'\n')
    f.write('Number of wrong utterances: '+str(wrong_utt)+'\n'+'\n')
    
    f.write('Number of utterances per label:'+'\n')
    f.write('ang: '+str(cont_utt_ang)+'\n')
    f.write('hap: '+str(cont_utt_hap)+'\n')
    f.write('neu: '+str(cont_utt_neu)+'\n')
    f.write('sad: '+str(cont_utt_sad)+'\n\n')
    
    f.write('Number of files per label:'+'\n')
    f.write('ang: '+str(cont_files_ang)+'\n')
    f.write('hap: '+str(cont_files_hap)+'\n')
    f.write('neu: '+str(cont_files_neu)+'\n')
    f.write('sad: '+str(cont_files_sad)+'\n\n')

print('Saving validation dictionaries...')
with open(dest_path+'val_hap_dict.csv', 'w') as f:
    for key, value in val_hap_dict.items():
        f.write('%s:%s\n' % (key, value))

with open(dest_path+'val_sad_dict.csv', 'w') as f:
    for key, value in val_sad_dict.items():
        f.write('%s:%s\n' % (key, value))

with open(dest_path+'val_neu_dict.csv', 'w') as f:
    for key, value in val_neu_dict.items():
        f.write('%s:%s\n' % (key, value))

with open(dest_path+'val_ang_dict.csv', 'w') as f:
    for key, value in val_ang_dict.items():
        f.write('%s:%s\n' % (key, value))

print('VALIDATION DICTIONARIES AND SUMMARY DONE!\n\n')



'''
################
##    TEST    ##
################
'''

''' First we create an ordered dictionary with all the faces/audios we have '''

print('Creating files-test-dictionary...')
test_dict = OrderedDict()
for i, label in enumerate(labels):
    directory = test_path + label + '/'
    files = os.listdir(directory)
    files.sort()
    for file in files:
        test_dict[(file,i)] = i
        
print('Length of files-test-dictionary: {}'.format(len(test_dict)))
print('')
print('Creating test dictionaries...')

test_hap_dict = OrderedDict()
test_sad_dict = OrderedDict()
test_neu_dict = OrderedDict()
test_ang_dict = OrderedDict()

cont_utt_hap = 0
cont_utt_sad = 0
cont_utt_neu = 0
cont_utt_ang = 0

cont_files_hap = 0
cont_files_sad = 0
cont_files_neu = 0
cont_files_ang = 0

ok_utt    = 0
wrong_utt = 0

i = 5
   
test_utt_txt = os.listdir(txt_path + 'test/' + 'Session{}'.format(i) + '/')
test_utt_txt.sort()
print('\t...Session{}...'.format(i))

for utt_txt in test_utt_txt:
   
    with open(txt_path + 'test/' + 'Session{}'.format(i) + '/' + utt_txt,"r") as f:
        data = f.readlines()
       
    data = data[1:]
   
    video_path = database_path + 'Session{}'.format(i) + '/' + utt_txt[:-4] + '/'  #video folder with all the faces for that video
    all_frames = os.listdir(video_path)
    all_frames.sort()
    n_frames = int(all_frames[-1][-10:-4])
   
    for line in data:  ## Each line is an utterance
       
        audio_files_list = []
        
        utt = line[:-1].split('\t')
       
        label = utt[1]
        label_int = label_to_ind[utt[1]]
        init_time = float(utt[2])
        final_time = float(utt[3])
       
        init_frame = round(init_time*frame_rate)
        final_frame = round(final_time*frame_rate)
       
        if final_frame > n_frames:  # This if is because in some cases the number of the last utterance we obtain multiplying the final time times the frame rate is bigger than the number of frames that we have obtained from the openface
            end_frame = n_frames
        else:
            end_frame = final_frame
                    
        frames_exist = [os.path.exists(video_path + utt_txt[:-4] + '_{:06d}'.format(j) + '.bmp') for j in range(init_frame, end_frame+1)]
       
        if frames_exist.count(False) == 0 and len(frames_exist)>0:
           
            for j in range(init_frame, end_frame+1):
               
                audio = utt_txt[:-4] + '_{:06d}'.format(j) + '.wav'
                if (audio, label_int) in test_dict:
                    audio_files_list.append(audio[:-4])
                    aux = test_dict.pop((audio, label_int))
                    
            audio_files_list.sort()
                
            if label == 'hap':
                cont_utt_hap += 1
                test_hap_dict['Utterance{:06d}'.format(cont_utt_hap)] = audio_files_list
                cont_files_hap += len(audio_files_list)
            elif label == 'sad':
                cont_utt_sad += 1
                test_sad_dict['Utterance{:06d}'.format(cont_utt_sad)] = audio_files_list
                cont_files_sad += len(audio_files_list)
            elif label == 'neu':
                cont_utt_neu += 1
                test_neu_dict['Utterance{:06d}'.format(cont_utt_neu)] = audio_files_list
                cont_files_neu += len(audio_files_list)
            elif label == 'ang':
                cont_utt_ang += 1
                test_ang_dict['Utterance{:06d}'.format(cont_utt_ang)] = audio_files_list
                cont_files_ang += len(audio_files_list) 
               
            ok_utt +=1
           
        else:
           
            wrong_utt += 1

print('Test dictionaries done...')
print('Saving test summary...')
with open(dest_path + 'summary_test.txt', 'w') as f:   
    f.write('SUMMARY OF THE TEST SET:\n\n')
    f.write('Number of ok utterances: '+str(ok_utt)+'\n')
    f.write('Number of wrong utterances: '+str(wrong_utt)+'\n'+'\n')
    
    f.write('Number of utterances per label:'+'\n')
    f.write('ang: '+str(cont_utt_ang)+'\n')
    f.write('hap: '+str(cont_utt_hap)+'\n')
    f.write('neu: '+str(cont_utt_neu)+'\n')
    f.write('sad: '+str(cont_utt_sad)+'\n\n')
    
    f.write('Number of files per label:'+'\n')
    f.write('ang: '+str(cont_files_ang)+'\n')
    f.write('hap: '+str(cont_files_hap)+'\n')
    f.write('neu: '+str(cont_files_neu)+'\n')
    f.write('sad: '+str(cont_files_sad)+'\n\n')

print('Saving test dictionaries...')
with open(dest_path+'test_hap_dict.csv', 'w') as f:
    for key, value in test_hap_dict.items():
        f.write('%s:%s\n' % (key, value))

with open(dest_path+'test_sad_dict.csv', 'w') as f:
    for key, value in test_sad_dict.items():
        f.write('%s:%s\n' % (key, value))

with open(dest_path+'test_neu_dict.csv', 'w') as f:
    for key, value in test_neu_dict.items():
        f.write('%s:%s\n' % (key, value))

with open(dest_path+'test_ang_dict.csv', 'w') as f:
    for key, value in test_ang_dict.items():
        f.write('%s:%s\n' % (key, value))

print('TEST DICTIONARIES AND SUMMARY DONE!\n\n')
