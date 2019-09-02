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
    
    def __init__(self, df_audio, df_features, df_im, data_path_audio, data_path_features, data_path_im, batch_size, nb_class, rescale = False, mode = 'train', seed = 127):
        self.df_audio = df_audio
        self.df_features = df_features
        self.df_im = df_im
        self.bsz = batch_size
        self.mode = mode
        self.seed = seed
        self.nb_class = nb_class
        self.rescale = rescale

        # Take labels and a list of image and audio locations in memory
        self.labels = self.df_im['label'].values
        self.feature_list = self.df_features['csv_name'].apply(lambda x: os.path.join(data_path_features, x)).tolist()
        self.audio_list = self.df_audio['audio_name'].apply(lambda x: os.path.join(data_path_audio, x)).tolist()
        self.im_list = self.df_im['im_name'].apply(lambda x: os.path.join(data_path_im, x)).tolist()
        if self.mode == 'train':
            random.seed(self.seed)

    def __len__(self):
        return int(math.ceil(len(self.df_audio) / float(self.bsz)))

    def load_audio(self, audio):
        _, data = wavfile.read(audio)
        return data

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
        return [np.array([self.load_image(im) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]]), np.array([np.expand_dims(self.load_audio(audio), axis=1) for audio in self.audio_list[idx * self.bsz: (1 + idx) * self.bsz]]), np.array([self.load_features(feature) for feature in self.feature_list[idx * self.bsz: (1 + idx) * self.bsz]])]

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

def model_audio(model_path_audio):
    
    audio_model = load_model(model_path_audio)
    
    for layer in audio_model.layers[:]:
        layer.trainable = False
        
    for layer in audio_model.layers:
        print(layer, layer.trainable)
        
    new_audio_model = Model(audio_model.inputs, audio_model.layers[-3].output)
    
    return(new_audio_model)


'''Definition of the variables, the generator and load the data'''

im_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/training_4labels.csv')
im_val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/validation_4labels.csv')
im_test = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/test_4labels.csv')
audio_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/training_4labels.csv')
audio_val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation_4labels.csv')
audio_test = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test_4labels.csv')
feature_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/training_4labels.csv')
feature_val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation_4labels.csv')
feature_test = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/test_4labels.csv')


im_size = 224
n_feature_vector = 5266
n_audio_vector = 1600
batch_size = 32
nb_class = 4
n_epochs = 20
steps_per_epoch_train = math.ceil(len(im_train)/batch_size)
steps_per_epoch_val = math.ceil(len(im_val)/batch_size)

seq_train = ImAudioSequence(audio_train, feature_train, im_train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train/','/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train/', 
                           '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/train/',
                           batch_size = batch_size, nb_class = nb_class, rescale = True, mode = 'train')
seq_val = ImAudioSequence(audio_val, feature_val, im_val, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation/','/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation/',  
                           '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/validation/',
                           batch_size = batch_size, nb_class = nb_class, rescale = True, mode = 'val')

seq_test = ImAudioSequence(audio_test, feature_test, im_test, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test/','/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/test/',  
                           '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/test/',
                           batch_size = batch_size, nb_class = nb_class, rescale = True, mode = 'val')

print('Files')
path_model = '/app/Fusion2/Fusion_faceraw_features_100/weights.000009-0.865-0.348-0.437-2.212.h5'
loaded_model = load_model(path_model)

print('Loaded model')

evaluations = loaded_model.evaluate_generator(generator = seq_test)


print('Evaluated')

print('Accuracy: ' + str(evaluations[1]*100))



