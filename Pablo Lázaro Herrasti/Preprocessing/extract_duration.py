from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx
import os
import gc



n_sessions = 5
duration = []

for i in range(n_sessions):
    cont = 1
    print("Session",i+1)
    dir_videos = 'D:/IEMOCAP_full_release/IEMOCAP/Video'
    dir_videos_s = dir_videos+'/Session'+str(i+1)
    files = os.listdir(dir_videos_s)
    for j in range(len(files)):
        gc.collect()
        dir_video_s = dir_videos_s+'/'+files[j]
        clip = VideoFileClip(dir_video_s)
        duration.append(clip.duration)
        clip.close()
with open('duration.txt', 'w') as f:
    for dur in duration:
        f.write(str(dur)+'\n')       