from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx
import os
import gc



n_sessions = 5
var_aux = 0
aux = 0

#for i in range(0,n_sessions):
i = 1
cont = 1
print("Session",i+1)
dir_videos = 'D:/IEMOCAP_full_release/IEMOCAP/Video'
dir_videos_s = dir_videos+'/Session'+str(i+1)
files = os.listdir(dir_videos_s)
dir_videos_cropped = 'D:/IEMOCAP_full_release/IEMOCAP/Video_cropped_2/Session'+str(i+1)
#os.mkdir(dir_videos_cropped)
for j in range(len(files)):
    gc.collect()
    if  j == 2:
        dir_video_s = dir_videos_s+'/'+files[j+var_aux]
        clip = VideoFileClip(dir_video_s)
        # 450,650
        new_clip_2 = vfx.crop(clip, x1=450,x2=645,y1=0,y2=280)        
        video_cropped_2 = dir_videos_cropped+'/2_'+files[j+var_aux][:-3] + 'mp4'
        new_clip_2.write_videofile(video_cropped_2,fps=29.97,audio=False)
        new_clip_2.close()
        clip.close()
            
        