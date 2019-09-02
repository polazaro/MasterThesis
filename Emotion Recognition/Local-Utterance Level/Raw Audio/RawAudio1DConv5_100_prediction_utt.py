
'''

To obtain the accuracy of the best model at utterance level for the
training, validation and test sets. Audio: 100ms.

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
    
    def __init__(self, df, data_path, batch_size, nb_class, mode = 'val', seed = 127):
        self.df = df
        self.bsz = batch_size
        self.mode = mode
        self.seed = seed
        self.nb_class = nb_class

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.audio_list = self.df['audio_name'].apply(lambda x: os.path.join(data_path, x)).tolist()
        if self.mode == 'train':
            random.seed(self.seed)

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
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

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
val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation_4labels.csv')
test = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test_4labels.csv')


batch_size = 252
nb_class = 4
n_audio_vector = 1064
n_epochs = 1000
steps_per_epoch_train = math.ceil(len(train)/batch_size)
steps_per_epoch_val = math.ceil(len(val)/batch_size)

seq_train = AudioDataSequence(train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train/',  
                           batch_size = batch_size, nb_class = nb_class, mode = 'train')
seq_val = AudioDataSequence(val, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation/',  
                         batch_size = batch_size, nb_class = nb_class, mode = 'val')
seq_test = AudioDataSequence(test, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test/',  
                         batch_size = batch_size, nb_class = nb_class, mode = 'val')

'''Predicting with the loaded model'''

print('Files')
path_model = '/app/Audio2/Experiment1_100/weights.000026-0.430-1.224-0.418-1.248.h5'
loaded_model = load_model(path_model)

print('Loaded model')

predictions = loaded_model.predict_generator(generator = seq_test)

print('Predictions')

'''Reading dictionaries'''

var = 'test'
long_utterances = []
real_value = []
with open('/app/data/IEMOCAP_dat1/Utterances_dict_2/' + var + '_hap_dict.csv','r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key,value = item.split(':', 1)
            if len(value) == 3:
                pass
            else:
                long_utterances.append(value.count(',')+1)
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
                long_utterances.append(value.count(',')+1)
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
                long_utterances.append(value.count(',')+1)
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
                long_utterances.append(value.count(',')+1)
                real_value.append(3)
        else:
            pass # deal with bad lines of text here

print('Dictionaries')


cont = 0
mode = 1
pred_value = []
for len_utt in long_utterances:
    pred = predictions[cont:cont+len_utt]
    votes = []
    if mode == 0:
        ## Majority vote
        for i in range(len(pred)):
            votes.append(np.where(pred[i] == max(pred[i]))[0][0])
        pred_value.append(max(set(votes), key = votes.count))

    else:
        # Mean 
       label_1 = []
       label_2 = []
       label_3 = []
       label_4 = []
       for i in range(len(pred)):
           label_1.append(pred[i][0])
           label_2.append(pred[i][1])
           label_3.append(pred[i][2])
           label_4.append(pred[i][3])
           
       mean_label_1 = np.mean(label_1)
       mean_label_2 = np.mean(label_2)
       mean_label_3 = np.mean(label_3)
       mean_label_4 = np.mean(label_4)
       aux = [mean_label_1,mean_label_2,mean_label_3,mean_label_4]
       pred_value.append(aux.index(max(aux)))
    
    cont += len_utt
'''Accuracy'''
print(cont,len(predictions))
print('Number of total utterance: ' + str(len(real_value)))
print('Number of happy labels real ' + str(real_value.count(0)))
print('Number of happy labels predicted ' + str(pred_value.count(0)))


correct = 0
for i in range(len(pred_value)):
    if pred_value[i] == real_value[i]:
        correct += 1
    
print('Accuracy of model at utterance level is: ' + str(100*(correct/(len(pred_value)))) + '%')
f_score = f1_score(real_value,pred_value, average='weighted')
acc2 = accuracy_score(real_value,pred_value)
print('F1Score of model at utterance level is: ' + str(f_score))
print('Accuracy sklearn is: ' + str(acc2))
    

    