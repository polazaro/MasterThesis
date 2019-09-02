# -*- coding: utf-8 -*-
"""
Created on Mon May  6 23:17:28 2019

@author: polaz
"""

'''

To obtain the accuracy of the best model at utterance level for the
training, validation and test sets.

'''

import os
import tensorflow as tf

from keras.utils import Sequence
from keras.models import Model
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential
from keras.layers.merge import concatenate
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from scipy.io import wavfile
import math
import random

import pandas as pd
import numpy as np

from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

'''Definition of the image-audio generator class'''
    
class ImAudioSequence(Sequence):
    
    def __init__(self, df_audio, df_im, data_path_audio, data_path_im, batch_size, nb_class, rescale = False, mode = 'train', seed = 127):
        self.df_audio = df_audio
        self.df_im = df_im
        self.bsz = batch_size
        self.mode = mode
        self.seed = seed
        self.nb_class = nb_class
        self.rescale = rescale

        # Take labels and a list of image and audio locations in memory
        self.labels = self.df_im['label'].values
        self.audio_list = self.df_audio['csv_name'].apply(lambda x: os.path.join(data_path_audio, x)).tolist()
        self.im_list = self.df_im['im_name'].apply(lambda x: os.path.join(data_path_im, x)).tolist()
        if self.mode == 'train':
            random.seed(self.seed)

    def __len__(self):
        return int(math.ceil(len(self.df_audio) / float(self.bsz)))

    def load_features(self, feat_name):
        data = pd.read_csv(feat_name, header=None).values
        return np.squeeze(data)
    
    def load_image(self, im):
        if self.rescale == True:
            return img_to_array(load_img(im)) / 255.
        else:
            return img_to_array(load_img(im))
        
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
        return np.array([self.num_to_vec(x) for x in self.labels[idx * self.bsz: (idx + 1) * self.bsz]])
    
    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return [np.array([self.load_image(im) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]]), np.array([self.load_features(audio) for audio in self.audio_list[idx * self.bsz: (1 + idx) * self.bsz]])]

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y

''' Functions to load the pre-trained models '''

def model_VGG(model_path_image):

    custom_vgg_model = load_model(model_path_image)
    
    for layer in custom_vgg_model.layers[:]:
        layer.trainable = False
        
    for layer in custom_vgg_model.layers:
        print(layer.name, layer.trainable)
    
    last_conv_layer_output = custom_vgg_model.layers[-2].output
    new_output = Dense(128, activation='relu')(last_conv_layer_output)
    
    new_custom_vgg_model = Model(custom_vgg_model.inputs, new_output)
    
    return(new_custom_vgg_model)
    
def model_audiofeatures(model_path_audio):
    
    audio_model = load_model(model_path_audio)
    
    for layer in audio_model.layers[:]:
        layer.trainable = False
        
    for layer in audio_model.layers:
        if layer.name == 'dense_1':
            layer.name = 'dense_aux'
        print(layer.name, layer.trainable)
        
    new_audio_model = Model(audio_model.inputs, audio_model.layers[-2].output)
    
    return(new_audio_model)


'''Definition of the variables, the generator and load the data'''

im_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/training_4labels.csv')
im_val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/validation_4labels.csv')
im_test = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/test_4labels.csv')
audio_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/training_4labels.csv')
audio_val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation_4labels.csv')
audio_test = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/test_4labels.csv')


im_size = 224
n_audio_vector = 1064
batch_size = 32
nb_class = 4
n_epochs = 20
steps_per_epoch_train = math.ceil(len(im_train)/batch_size)
steps_per_epoch_val = math.ceil(len(im_val)/batch_size)

seq_train = ImAudioSequence(audio_train, im_train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train/', 
                           '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/train/',
                           batch_size = batch_size, nb_class = nb_class, rescale = True, mode = 'train')
seq_val = ImAudioSequence(audio_val, im_val, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation/',  
                           '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/validation/',
                           batch_size = batch_size, nb_class = nb_class, rescale = True, mode = 'val')
seq_test = ImAudioSequence(audio_test, im_test, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/test/',  
                           '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/test/',
                           batch_size = batch_size, nb_class = nb_class, rescale = True, mode = 'val')

'''Predicting with the loaded model'''

print('Files')
path_model = '/app/Fusion2/Fusion_features_100_arq1/weights.000009-0.862-0.356-0.430-2.265.h5'
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
f_score = f1_score(real_value,pred_value, average='macro')
acc2 = accuracy_score(real_value,pred_value)
print('F1Score of model at utterance level is: ' + str(f_score))
print('Accuracy sklearn is: ' + str(acc2))


