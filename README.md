# Multi-modal Local Non-verbal Emotion Recognition in Dyadic scenarios and Speaker Segmentation

This folder contains the repository of the master thesis of **Pablo Lázaro Herrasti** in relation with the **Non-acted Multi-view Audio-Visual Dyadic Interactions Project**.

### Description of the Master Thesis
This Master Thesis is focused on the development of baseline **Emotion Recognition System** in a dyadic environment using raw and handcraft audio features and cropped faces from the videos. This system is analyzed at frame and utterance level without temporal information. As well, a baseline **Speaker Segmentation** system has been developed to facilitate the annotation task. For this reason, an exhaustive study of the state-of-the-art on emotion recognition and speaker segmentation techniques has been conducted, paying particular attention on Deep Learning techniques for emotion recognition and clustering for speaker segmentation.

While studying the state-of-the-art from the theoretical point of view, a dataset consisting of videos of sessions of dyadic interactions between individuals in different scenarios has been recorded. Different attributes were captured and labelled from these videos: body pose, hand pose, emotion, age, gender, etc. Once the architectures for emotion recognition have been trained with other dataset, a proof of concept is done with this new database in order to extract conclusions. In addition, this database can help future systems to achieve better results.

A large number of experiments with audio and video are performed to create the emotion recognition system. The **IEMOCAP database** is used to perform the training and evaluation experiments of the emotion recognition system. Once the audio and video are trained separately with two different architectures, a fusion of both methods is done. In this work, the importance of preprocessing data (i.e. face detection, windows analysis length, handcrafted features, etc.) and choosing the correct parameters for the architectures (i.e. network depth, fusion, etc.) has been demonstrated and studied.

On the other hand, the experiments for the speaker segmentation system are performed with a piece of audio from IEMOCAP database. In this work, the prerprocessing steps, the problems of an unsupervised system such as clustering and the feature representation are studied and discussed.
 

### Repository Explanation

All the scripts are coded in Python and tested and debugged using Spyder (Anaconda). The paths inside the codes vary between the personal paths of the local computer and the paths of the server provided by **HuPBA** (Human Pose Recovery and Behavior Analysis). Normally, all the training and preprocessing files should have the paths from the server because they are computationally expensive. 

The repository is organized in the following way: 

* **Dockerfiles**: the two dockerfiles coded for runing the experiments on the server:
  * The first Dockerfile contains the needed libraries for running the experiments with *Keras* over *Tensorflow* with gpu. We had to add also the *vgg* library and the libraries to load the files (*Pandas*, *Scipy*, *Pillow*)
  * The second Dockerfile contains the libraries for the face alignment process. It contains the corresponding lines to install *OpenCV* and *PyTorch*.
* **Preprocessing**: all the scripts coded for organizing the IEMOCAP database, for the preprocessing of audio, handcraft features and video and for creating all the .txt, .csv and dictionaries for the training process. There are some common files used for all the database in general. This folder is subdivided in three folders:
  * **Audio Features**: preprocessing files for the handcraft audio fetures. 
  * **Raw Audio**: preprocessing files for the rau audio segments of 66-100ms.
  * **Video**: preprocessing files for the obtention of the cropped and alligned faces from the videos.
* **Emotion Recognition**: all the training scripts for the Emotion Recognition system used in the master thesis. This folder is has another folder named **Local-Utterance Level** for the frame level analysis. Both folders are again subdivided in 4 subfolders: **Audio Features**, **Fusion** (with all the codes for the bimodal and trimodal fusion), **Raw Audio** and **Video**. All these strategies follow the same organization of files:
  * A file coded with the generator and all the needed lines for training the models (i.e. `model_name.py`).
  * A similar file to obtain the evaluation of the best model for the training, vaidation and test sets (i.e. `model_name_prediction.py`).
  * Another file to evaluate the best model at utterance level (i.e. `model_name_prediction_utt.py`).
  * One last file to obtain the performance of the best model at utterance level and for each label (i.e. `model_name_prediction_utt_label.py`).
 In some cases there is also a script to compute the Proof of Concept with the video of the **Face-to-face Dyadic Interaction Dataset**. These files follow the name `model_name_prediction_utt_proof_concept.py`


# Contact

* DockerHub: [DockerfilePyTorch](https://hub.docker.com/r/rubenbt/tfm_torch) / [DockerfileKerasTensorflow](https://hub.docker.com/r/rubenbt/tfm_docker1)
* LinkedIn: [PabloLázaroHerrasti](http://www.linkedin.com/in/plazh)
