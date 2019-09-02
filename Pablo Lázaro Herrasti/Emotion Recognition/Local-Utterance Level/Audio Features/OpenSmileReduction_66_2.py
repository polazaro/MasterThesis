import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import Sequence
from keras import backend as K

from scipy.io import wavfile

import pandas as pd
import numpy as np
import math
import random
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

'''Definition of the raw-audio generator class'''

class FeatDataSequence(Sequence):
    
    def __init__(self, df, data_path, batch_size, nb_class, mode = 'train', seed = 127):
        self.df = df
        self.bsz = batch_size
        self.mode = mode
        self.seed = seed
        self.nb_class = nb_class

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.csv_list = self.df['csv_name'].apply(lambda x: os.path.join(data_path, x)).tolist()
        if self.mode == 'train':
            random.seed(self.seed)

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))
    
    def load_features(self, feat_name):
        data = pd.read_csv(feat_name, header=None).values
        return np.squeeze(data)

    def num_to_vec(self, label):
        v = np.zeros(self.nb_class)
        v[label] = 1
        return v

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.csv_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
#        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]
        return np.array([self.num_to_vec(x) for x in self.labels[idx * self.bsz: (idx + 1) * self.bsz]])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([self.load_features(feat_file) for feat_file in self.csv_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y

'''Definition of the variables and the generator and load the data'''

train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_66/4Labels/training_4labels.csv')
val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_66/4Labels/validation_4labels.csv')

batch_size = 128
nb_class = 4
n_audio_vector = 2099
n_epochs = 1000
steps_per_epoch_train = math.ceil(len(train)/batch_size)
steps_per_epoch_val = math.ceil(len(val)/batch_size)

seq_train = FeatDataSequence(train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_66/4Labels/train/',  
                           batch_size = batch_size, nb_class = nb_class, mode = 'train')
seq_val = FeatDataSequence(val, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_66/4Labels/validation/',  
                         batch_size = batch_size, nb_class = nb_class, mode = 'val')


'''Definition of the network'''

## Defining the model
model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(n_audio_vector,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_class, activation='softmax'))

adam = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
model.summary()


filepath = "weights.{epoch:06d}-{acc:.3f}-{loss:.3f}-{val_acc:.3f}-{val_loss:.3f}.h5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

A = model.fit_generator(
        generator = seq_train, steps_per_epoch = steps_per_epoch_train, epochs = n_epochs,
        callbacks=[checkpointer], validation_data = seq_val, validation_steps = steps_per_epoch_val)
