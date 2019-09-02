# -*- coding: utf-8 -*-
"""
Created on Mon May  6 23:17:28 2019

@author: polaz
"""
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

im_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/training_4labels_flip.csv')
im_val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/validation_4labels_flip.csv')
audio_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/training_4labels_flip.csv')
audio_val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation_4labels_flip.csv')

im_size = 224
n_audio_vector = 5266
batch_size = 32
nb_class = 4
n_epochs = 20
steps_per_epoch_train = math.ceil(len(im_train)/batch_size)
steps_per_epoch_val = math.ceil(len(im_val)/batch_size)
model_path_audio = '/app/AudioFeatures2/Experiment1_100_1/weights.000026-0.403-1.253-0.428-1.228.h5'
model_path_image = '/app/Faces2/Experiment1/weights.000004-0.785-0.549-0.392-2.273.h5'

seq_train = ImAudioSequence(audio_train, im_train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train/', 
                           '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/train/',
                           batch_size = batch_size, nb_class = nb_class, rescale = True, mode = 'train')
seq_val = ImAudioSequence(audio_val, im_val, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation/',  
                           '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/validation/',
                           batch_size = batch_size, nb_class = nb_class, rescale = True, mode = 'val')

'''Real definition of the model'''

VGG_model = model_VGG(model_path_image)
audio_model = model_audiofeatures(model_path_audio)

## Concatenating both models
conc_features = concatenate([VGG_model.layers[-1].output, audio_model.layers[-1].output],axis=1)
fc1 = Dense(32, activation='relu',name='fc_class_1')(conc_features)
out = Dense(nb_class, activation='softmax', name='fc_class_2')(fc1)
fusion_model = Model([VGG_model.input,audio_model.input], out)

adam = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
fusion_model.summary()

'''TRAINING'''

filepath = "weights.{epoch:06d}-{acc:.3f}-{loss:.3f}-{val_acc:.3f}-{val_loss:.3f}.h5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

H = fusion_model.fit_generator(
        generator = seq_train, steps_per_epoch = steps_per_epoch_train, epochs = n_epochs,
        callbacks=[checkpointer], validation_data = seq_val, validation_steps = steps_per_epoch_val)


