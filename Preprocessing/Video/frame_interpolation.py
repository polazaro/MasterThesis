import os
import pickle
import cv2
import math


n_sessions = 5

dir_images = r'/app/data/IEMOCAP_dat1/IEMOCAP_faces_final_2/'
dir_files = r'/app/data/IEMOCAP_dat1/IEMOCAP_error_final_frames/'
#dir_images = 'D:/Data_Master/TFM/Images/'
#dir_files = 'D:/Data_Master/TFM/Listas/'

utterance_less = 0


for i in range(n_sessions):
    dir_files_session = dir_files + r'Session' + str(i+1) + r'/'
    dir_images_session = dir_images + r'Session' + str(i+1) + r'/'
#    dir_files_session = dir_files + 'Session' + str(i+1) + '/'
#    dir_images_session = dir_images + 'Session' + str(i+1) + '/'
    files = os.listdir(dir_files_session)
    print('Session'+str(i+1))
    for j in range(len(files)):
        print(files[j][:-4])
        with open(dir_files_session+files[j], 'rb') as f:
            bad_images = pickle.load(f)
        dir_images_session_final = dir_images + r'Session' + str(i+1) + r'/' + files[j][:-4] + '/'
#        dir_images_session_final = dir_images + 'Session' + str(i+1) + '/' + files[j][:-4] + '/'
        cont1 = 0
        stop = True
        cont2 = 1
        while stop:
            if cont1+cont2 == len(bad_images):
                    stop = False
            else:
                while int(bad_images[cont1][-6:])+cont2 == int(bad_images[cont1+cont2][-6:]):
                    cont2 += 1
                    if cont1+cont2 == len(bad_images):
                        stop = False
                        break
            if cont2<=30:
                if cont2 == 1:
                    image_inter = cv2.imread(dir_images_session_final+bad_images[cont1][:-6]+\
                                               '{:06d}'.format(int(bad_images[cont1][-6:])-1)+'.bmp')
                    cv2.imwrite(dir_images_session_final+bad_images[cont1]+'.bmp',image_inter)
                else:
                    if cont1+cont2 == len(bad_images):
                        for k in range(cont2):
                                image_inter1 = cv2.imread(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   '{:06d}'.format(int(bad_images[cont1][-6:])-1)+'.bmp')
                                cv2.imwrite(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   bad_images[cont1+k][-6:]+'.bmp',image_inter1)
                    else:
                        if cont2%2 == 0:
                            num = int(cont2/2)
                            for k in range(num):
                                image_inter1 = cv2.imread(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   '{:06d}'.format(int(bad_images[cont1][-6:])-1)+'.bmp')
                                cv2.imwrite(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   bad_images[cont1+k][-6:]+'.bmp',image_inter1)
                                image_inter2 = cv2.imread(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   '{:06d}'.format(int(bad_images[cont1+cont2-1][-6:])+1)+'.bmp')
                                cv2.imwrite(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   bad_images[cont1+cont2-k-1][-6:]+'.bmp',image_inter2)
                        else:
                            num = math.ceil(cont2/2)
                            for k in range(num):
                                image_inter1 = cv2.imread(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   '{:06d}'.format(int(bad_images[cont1][-6:])-1)+'.bmp')
                                cv2.imwrite(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   bad_images[cont1+k][-6:]+'.bmp',image_inter1)
                            for k in range(num-1):
                                image_inter2 = cv2.imread(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   '{:06d}'.format(int(bad_images[cont1+cont2-1][-6:])+1)+'.bmp')
                                cv2.imwrite(dir_images_session_final+bad_images[cont1][:-6]+\
                                                   bad_images[cont1+cont2-k-1][-6:]+'.bmp',image_inter2)
                            
            else:
                utterance_less += 1
            cont1 += cont2
            cont2 = 1
                
print('Number of times missing more than 30 frames: ' + str(utterance_less))
        
        
        
            
            