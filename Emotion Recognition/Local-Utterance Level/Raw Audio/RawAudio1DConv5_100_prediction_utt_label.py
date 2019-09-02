
'''

To obtain the accuracy of the best model at utterance level for the
training, validation and test sets for each label. Audio: 100ms.

'''

import os
import tensorflow as tf

from keras.utils import Sequence
from keras.models import Model, load_model
from scipy.io import wavfile

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math
import random

import pandas as pd
import numpy as np

from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

'''Definition of the raw-audio generator class'''

class AudioDataSequence(Sequence):
    
    def __init__(self, df, data_path, batch_size, nb_class):
        self.df = df
        self.bsz = batch_size
        self.nb_class = nb_class

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.audio_list = self.df['audio_name'].apply(lambda x: os.path.join(data_path, x)).tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))
    
    def load_audio(self, audio):
        _, data = wavfile.read(audio)
        return data

    def num_to_vec(self, label):
        v = np.zeros(self.nb_class)
        v[label] = 1
        return v

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.audio_list))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
#        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]
        return np.array([self.num_to_vec(x) for x in self.labels[idx * self.bsz: (idx + 1) * self.bsz]])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([np.expand_dims(self.load_audio(audio), axis=1) for audio in self.audio_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


'''Definition of the variables and the generator and load the data'''

train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/training_4labels.csv')
val   = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation_4labels.csv')
test  = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test_4labels.csv')


batch_size = 252
nb_class = 4
n_audio_vector = 1600

steps_per_epoch_train = math.ceil(len(train)/batch_size)
steps_per_epoch_val   = math.ceil(len(val)/batch_size)
steps_per_epoch_test  = math.ceil(len(test)/batch_size)

seq_train = AudioDataSequence(train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train/',  
                           batch_size = batch_size, nb_class = nb_class)
seq_val   = AudioDataSequence(val, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation/',  
                           batch_size = batch_size, nb_class = nb_class)
seq_test  = AudioDataSequence(test, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test/',  
                           batch_size = batch_size, nb_class = nb_class)

'''Predicting with the loaded model'''

#print('Files')
path_model = '/app/Experiment1_100/weights.000026-0.430-1.224-0.418-1.248.h5'
loaded_model = load_model(path_model)

print('Loaded model')

var = 'test'
predictions = loaded_model.predict_generator(generator = seq_test)

print('Computing {} accuracies...'.format(var))



'''Reading dictionaries'''
hap_long = [] 
sad_long = []
neu_long = []
ang_long = [] 
real_value = []
with open('/app/data/IEMOCAP_dat1/Utterances_dict_2/' + var + '_hap_dict.csv','r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key,value = item.split(':', 1)
            if len(value) == 3:
                pass
            else:
                hap_long.append(value.count(',')+1)
                real_value.append(0)
        else:
            pass # deal with bad lines of text here
            
dict_sad = dict()
with open('/app/data/IEMOCAP_dat1/Utterances_dict_2/' + var + '_sad_dict.csv','r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key,value = item.split(':', 1)
            if len(value) == 3:
                pass
            else:
                sad_long.append(value.count(',')+1)
                real_value.append(1)
        else:
            pass # deal with bad lines of text here
            
dict_neu = dict()
with open('/app/data/IEMOCAP_dat1/Utterances_dict_2/' + var + '_neu_dict.csv','r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key,value = item.split(':', 1)
            if len(value) == 3:
                pass
            else:
                neu_long.append(value.count(',')+1)
                real_value.append(2)
        else:
            pass # deal with bad lines of text here
            
dict_ang = dict()
with open('/app/data/IEMOCAP_dat1/Utterances_dict_2/' + var + '_ang_dict.csv','r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key,value = item.split(':', 1)
            if len(value) == 3:
                pass
            else:
                ang_long.append(value.count(',')+1)
                real_value.append(3)
        else:
            pass # deal with bad lines of text here

cont = 0
true_labels = [] 
pred_labels_maj  = []
pred_labels_mean = []
for long_utt in hap_long:
    true_labels.append(0)
    utt_preds = predictions[cont:cont+long_utt]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
    pred_labels_maj.append(max(set(votes), key = votes.count))
    
    ### Mean
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    for i in range(len(utt_preds)):
        label_1.append(utt_preds[i][0])
        label_2.append(utt_preds[i][1])
        label_3.append(utt_preds[i][2])
        label_4.append(utt_preds[i][3])
   
    mean_label_1 = np.mean(label_1)
    mean_label_2 = np.mean(label_2)
    mean_label_3 = np.mean(label_3)
    mean_label_4 = np.mean(label_4)
    aux = [mean_label_1,mean_label_2,mean_label_3,mean_label_4]
    pred_labels_mean.append(aux.index(max(aux)))
    
    cont += long_utt
    
for long_utt in sad_long:
    true_labels.append(1)
    utt_preds = predictions[cont:cont+long_utt]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
    pred_labels_maj.append(max(set(votes), key = votes.count))
    
    ### Mean
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    for i in range(len(utt_preds)):
        label_1.append(utt_preds[i][0])
        label_2.append(utt_preds[i][1])
        label_3.append(utt_preds[i][2])
        label_4.append(utt_preds[i][3])
   
    mean_label_1 = np.mean(label_1)
    mean_label_2 = np.mean(label_2)
    mean_label_3 = np.mean(label_3)
    mean_label_4 = np.mean(label_4)
    aux = [mean_label_1,mean_label_2,mean_label_3,mean_label_4]
    pred_labels_mean.append(aux.index(max(aux)))
    
    cont += long_utt

for long_utt in neu_long:
    true_labels.append(2)
    utt_preds = predictions[cont:cont+long_utt]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
    pred_labels_maj.append(max(set(votes), key = votes.count))
    
    ### Mean
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    for i in range(len(utt_preds)):
        label_1.append(utt_preds[i][0])
        label_2.append(utt_preds[i][1])
        label_3.append(utt_preds[i][2])
        label_4.append(utt_preds[i][3])
   
    mean_label_1 = np.mean(label_1)
    mean_label_2 = np.mean(label_2)
    mean_label_3 = np.mean(label_3)
    mean_label_4 = np.mean(label_4)
    aux = [mean_label_1,mean_label_2,mean_label_3,mean_label_4]
    pred_labels_mean.append(aux.index(max(aux)))
    
    cont += long_utt

for long_utt in ang_long:
    true_labels.append(3)
    utt_preds = predictions[cont:cont+long_utt]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
    pred_labels_maj.append(max(set(votes), key = votes.count))
    
    ### Mean
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    for i in range(len(utt_preds)):
        label_1.append(utt_preds[i][0])
        label_2.append(utt_preds[i][1])
        label_3.append(utt_preds[i][2])
        label_4.append(utt_preds[i][3])
   
    mean_label_1 = np.mean(label_1)
    mean_label_2 = np.mean(label_2)
    mean_label_3 = np.mean(label_3)
    mean_label_4 = np.mean(label_4)
    aux = [mean_label_1,mean_label_2,mean_label_3,mean_label_4]
    pred_labels_mean.append(aux.index(max(aux)))
    
    cont += long_utt  
    
'''Accuracy'''
print('Set: {}'.format(var))
print('Number of predictions: {}'.format(len(predictions)))
print('Final value of cont:   {}'.format(cont))
print('Number of utterances pred:  {}'.format(len(pred_labels_mean)))
print('Number of utterances true:  {}'.format(len(true_labels)))

correct_mean = 0
for i in range(len(pred_labels_mean)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(len(pred_labels_maj)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1
        
#print('Accuracy of model at local level: {} %'.format(100*(correct_local/len(predictions))))
print('Accuracy of model at utterance level (majority vote): {} %'.format(100*(correct_maj/len(pred_labels_maj))))
print('Accuracy of model at utterance level          (mean): {} %'.format(100*(correct_mean/len(pred_labels_mean))))
print('Accuracy of model at utterance level:')

correct_mean = 0
for i in range(0,len(hap_long)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(0,len(hap_long)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1

print('hap (majority vote): {} %'.format(100*(correct_maj/len(hap_long))))
print('hap          (mean): {} %'.format(100*(correct_mean/len(hap_long))))

correct_mean = 0
for i in range(len(hap_long),len(hap_long)+len(sad_long)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(len(hap_long),len(hap_long)+len(sad_long)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1

print('sad (majority vote): {} %'.format(100*(correct_maj/len(sad_long))))
print('sad          (mean): {} %'.format(100*(correct_mean/len(sad_long))))

correct_mean = 0
for i in range(len(hap_long)+len(sad_long),len(hap_long)+len(sad_long)+len(neu_long)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(len(hap_long)+len(sad_long),len(hap_long)+len(sad_long)+len(neu_long)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1

print('neu (majority vote): {} %'.format(100*(correct_maj/len(neu_long))))
print('neu          (mean): {} %'.format(100*(correct_mean/len(neu_long))))

correct_mean = 0
for i in range(len(hap_long)+len(sad_long)+len(neu_long),len(hap_long)+len(sad_long)+len(neu_long)+len(ang_long)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(len(hap_long)+len(sad_long)+len(neu_long),len(hap_long)+len(sad_long)+len(neu_long)+len(ang_long)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1

print('ang (majority vote): {} %'.format(100*(correct_maj/len(ang_long))))
print('ang          (mean): {} %'.format(100*(correct_mean/len(ang_long))))



    