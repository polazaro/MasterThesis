# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 23:13:02 2019

@author: Rubén
"""

## In this script we are going to pass from the videos to the cropped-faces
## directly because I don't have enough space in my computer and hard disk
## for videos-to-frames and then frames-to-faces.
##
## The script is a combination of videos_to_frames.py and frames_to_faces.py

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 19:41:50 2019

@author: Rubén
"""

''' Libraries, variables and functions '''
import os
import cv2
import torch
import numpy as np
import face_alignment
import pickle
from collections import OrderedDict

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=224, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image):
        # convert the landmark (x, y)-coordinates to a NumPy array
        pred = self.predictor.get_landmarks_from_image(image)

        if pred != None:
            pred = pred[-1]
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEyePts = pred[lStart:lEnd]
            rightEyePts = pred[rStart:rEnd]
    
            # compute the center of mass for each eye
            leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
            rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    
            # compute the angle between the eye centroids
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
    
            # compute the desired right eye x-coordinate based on the
            # desired x-coordinate of the left eye
            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
    
            # determine the scale of the new resulting image by taking
            # the ratio of the distance between eyes in the *current*
            # image to the ratio of distance between eyes in the
            # *desired* image
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist
    
            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input image
            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    
            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    
            # update the translation component of the matrix
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])
    
            # apply the affine transformation
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(image, M, (w, h),
                flags=cv2.INTER_CUBIC)
    
            # return the aligned face
            return output, True, scale,angle,dist
        else:
            return image, False, 0, 0, 0


''' Main part of the code '''


#base_dir_ori = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_cropped/'
#base_dir_des = 'D:/Master/Master Data Science/2 Cuatrimestre/Master Thesis/Database/IEMOCAP_faces_2/'
base_dir_ori = r'/app/data/IEMOCAP_dat1/IEMOCAP_cropped_2/'
base_dir_des = r'/app/data/IEMOCAP_dat1/IEMOCAP_faces_final_2/'
base_dir_des_list = r'/app/data/IEMOCAP_dat1/IEMOCAP_error_final_frames/'

predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=False) #or 'gpu' to use gpu
fa = FaceAligner(predictor, desiredFaceWidth=224)

for i in range(0,2):
    print('SESSION ' + str(i+1) + ':\n')
    ses_dir_ori =  base_dir_ori + r'Session' + str(i+1) + r'/'
    ses_dir_des =  base_dir_des + r'Session' + str(i+1) + r'/'
    ses_dir_des_list =  base_dir_des_list + r'Session' + str(i+1) + r'/'
    
    videos = os.listdir(ses_dir_ori)
    
    for video in videos:
        print(video)
        
        video_folder = ses_dir_des + video[:-4] + '/'
#        os.makedirs(video_folder)
        vidcap = cv2.VideoCapture(ses_dir_ori + video)
        success, image = vidcap.read()
        cont = 1
        error_detection = []
        
        while success:
            
            crop_frame_des = video_folder + video[:-4] + r'_{:06d}'.format(cont) + r'.bmp'
            crop_image, check, scale,angle,dist = fa.align(image)
            if check == False or (abs(angle) > 60 and abs(angle) < 318) or scale < 3 or scale > 30 or dist < 3:
                error_detection.append(video[:-4] + r'_{:06d}'.format(cont))
            else:
                cv2.imwrite(crop_frame_des, crop_image)
            success, image = vidcap.read()
            cont += 1
        with open(ses_dir_des_list+video[:-4]+'.pkl', 'wb') as f:
            pickle.dump(error_detection, f)
        
        print(video + ': ' + str(cont-1) + ' frames')
        
    print('\n\n') 
        

        
        
        