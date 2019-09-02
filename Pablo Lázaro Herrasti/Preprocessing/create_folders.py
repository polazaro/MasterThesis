import os


dir_ori = 'D:/IEMOCAP_full_release/IEMOCAP/Video_cropped_2/'
dir_dest = 'D:/IEMOCAP_full_release/IEMOCAP/Folders/'

for i in range(5):
    videos = os.listdir(dir_ori+'Session'+str(i+1))
    os.makedirs(dir_dest+'Session'+str(i+1))
    for video in videos:
        os.makedirs(dir_dest+'Session'+str(i+1)+'/'+video[:-4])