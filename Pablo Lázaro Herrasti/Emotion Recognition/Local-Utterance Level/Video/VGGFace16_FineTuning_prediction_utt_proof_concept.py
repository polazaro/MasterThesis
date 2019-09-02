import os
import tensorflow as tf

from keras.utils import Sequence
from keras.models import Model,load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import math
import random

import pandas as pd
import numpy as np

from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

'''Definition of the image generator class'''

class ImDataSequence(Sequence):
    
    def __init__(self, df, data_path, batch_size, nb_class, rescale = False):
        self.df = df
        self.bsz = batch_size
        self.rescale = rescale
        self.nb_class = nb_class

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.im_list = self.df['im_name'].apply(lambda x: os.path.join(data_path, x)).tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))
    
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
        self.indexes = range(len(self.im_list))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
#        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]
        return np.array([self.num_to_vec(x) for x in self.labels[idx * self.bsz: (idx + 1) * self.bsz]])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([self.load_image(im) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


'''Definition of the variables and the generator and load the data'''

F1_talking = pd.read_csv('/app/data/ProofConcept/Faces/F1_faces.csv')
F2_talking = pd.read_csv('/app/data/ProofConcept/Faces/F2_faces.csv')

im_size = 224
batch_size = 16
nb_class = 4

steps_per_epoch_F1 = math.ceil(len(F1_talking)/batch_size)
steps_per_epoch_F2 = math.ceil(len(F2_talking)/batch_size)

seq_F1 = ImDataSequence(F1_talking, '/app/data/ProofConcept/Faces/',  
                         batch_size = batch_size, nb_class = nb_class, rescale = True)
seq_F2 = ImDataSequence(F2_talking, '/app/data/ProofConcept/Faces/',  
                         batch_size = batch_size, nb_class = nb_class, rescale = True)


'''Predicting with the loaded model'''

path_model = '/app/Experiment1/weights.000004-0.785-0.549-0.392-2.273.h5'
loaded_model = load_model(path_model)

print('Loaded model')
evaluations_F1 = loaded_model.evaluate_generator(generator = seq_F1)
print('Accuracy F1: ' + str(evaluations_F1[1]*100))
evaluations_F2 = loaded_model.evaluate_generator(generator = seq_F2)
print('Accuracy F2: ' + str(evaluations_F2[1]*100))

predictions_F1 = loaded_model.predict_generator(generator = seq_F1)
predictions_F2 = loaded_model.predict_generator(generator = seq_F2)
#print(str(len(predictions_F1)))
pred_labels_F1 = [np.argmax(pred) for pred in predictions_F1]
pred_labels_F2 = [np.argmax(pred) for pred in predictions_F2]
#print(str(len(pred_labels_F1)))

print('F1_talking...')
true_labels_local = []

for i in range(0,81):
    true_labels_local.append([pred_labels_F1[i], str(0)])
for i in range(81,108):
    true_labels_local.append([pred_labels_F1[i], str(2)])
for i in range(108,166):
    true_labels_local.append([pred_labels_F1[i], str(1)])
for i in range(166,220):
    true_labels_local.append([pred_labels_F1[i], str(3)])
for i in range(220,270):
    true_labels_local.append([pred_labels_F1[i], str(2)])

cont = 0
true_labels_utt = [0, 2, 1, 3, 2] 
pred_labels_utt  = []

inits_F1  = [0, 81, 108, 168, 220]
finals_F1 = [81, 108, 166, 220, 270]

#F1_preds_1 = predictions[inits_F1[0]:finals_F1[0]]
#F1_preds_2 = predictions[inits_F1[1]:finals_F1[1]]
#F1_preds_3 = predictions[inits_F1[2]:finals_F1[2]]
#F1_preds_4 = predictions[inits_F1[3]:finals_F1[3]]
#F1_preds_5 = predictions[inits_F1[4]:finals_F1[4]]

for j in range(0,5):
    
    utt_preds = predictions_F1[inits_F1[j]:finals_F1[j]]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
#    pred_labels_maj.append(max(set(votes), key = votes.count))
    
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
#    pred_labels_mean.append(aux.index(max(aux)))
    pred_labels_utt.append([true_labels_utt[j], max(set(votes), key = votes.count), aux.index(max(aux))])
      
'''Accuracy'''
print('Number of predictions: {}'.format(len(predictions_F1)))
print('Number of utterances pred:  {}'.format(len(pred_labels_utt)))


preds_F1_df = pd.DataFrame(true_labels_local, columns=['pred','true'])
preds_F1_df.to_csv('/app/data/ProofConcept/Faces/'+'local_preds_F1.csv', index=False)
preds_F1_utt_df = pd.DataFrame(pred_labels_utt, columns=['true','mean','majority vote'])
preds_F1_utt_df.to_csv('/app/data/ProofConcept/Faces/'+'utt_preds_F1.csv', index=False)


print('F2_talking...')
true_labels_local = []

for i in range(0,52):
    true_labels_local.append([pred_labels_F2[i], 0])
for i in range(52,217):
    true_labels_local.append([pred_labels_F2[i], 2])
for i in range(217,264):
    true_labels_local.append([pred_labels_F2[i], 2])

cont = 0
true_labels_utt = [0, 2, 2] 
pred_labels_utt  = []

inits_F2  = [0, 52, 217]
finals_F2 = [52, 217, 264]

#F2_preds_1 = predictions[inits_F1[0]:finals_F1[0]]
#F2_preds_2 = predictions[inits_F1[1]:finals_F1[1]]
#F2_preds_3 = predictions[inits_F1[2]:finals_F1[2]]


for j in range(0,3):
    
    utt_preds = predictions_F2[inits_F2[j]:finals_F2[j]]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
#    pred_labels_maj.append(max(set(votes), key = votes.count))
    
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
#    pred_labels_mean.append(aux.index(max(aux)))
    pred_labels_utt.append([true_labels_utt[j], max(set(votes), key = votes.count), aux.index(max(aux))])
      
'''Accuracy'''
print('Number of predictions: {}'.format(len(predictions_F2)))
print('Number of utterances pred:  {}'.format(len(pred_labels_utt)))


preds_F2_df = pd.DataFrame(true_labels_local, columns=['pred','true'])
preds_F2_df.to_csv('/app/data/ProofConcept/Faces/'+'local_preds_F2.csv', index=False)
preds_F2_utt_df = pd.DataFrame(pred_labels_utt, columns=['true','mean','majority vote'])
preds_F2_utt_df.to_csv('/app/data/ProofConcept/Faces/'+'utt_preds_F2.csv', index=False)

