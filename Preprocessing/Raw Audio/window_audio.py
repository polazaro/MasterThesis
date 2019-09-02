from scipy.io import wavfile
import os
from moviepy.editor import VideoFileClip
import gc


t_window = 0.100
t_window_2 = 0.0333
overlap = 0.5
n_sessions = 5
dir_txt = 'D:/IEMOCAP_full_release/IEMOCAP/Video/'
cont = 0
cont_dur = 0

f = open(dir_txt+'duration.txt')
txt = f.read()
txt = txt.split('\n')
f.close()

#for k in range(n_sessions):
#    print('Session'+str(k+1))
#    dir_audio = 'D:/IEMOCAP_full_release/IEMOCAP/Audio_mono/Session'+str(k+1)+'/'
#    dir_final_audio = 'D:/IEMOCAP_full_release/IEMOCAP/Audio_window_mono_5/Session'+str(k+1)+'/'
##    os.mkdir(dir_final_audio[:-1])
#    files = os.listdir(dir_audio)
#    for i in range(len(files)):
#        print('File'+' '+str(i+1)+'/'+str(len(files)))
#        fs, audio = wavfile.read(dir_audio+files[i])
#        real_length = float(txt[cont_dur])
#        window_len = int(fs*t_window)
#        n_div = int(real_length/t_window)
#        cont = 0
##        os.mkdir(dir_final_audio+'1_'+files[i][:-4])
##        os.mkdir(dir_final_audio+'2_'+files[i][:-4])
#        for j in range(n_div):
#            output = audio[cont:(window_len+cont)]
#            cont += window_len
#            wavfile.write(dir_final_audio+'1_'+files[i][:-4]+'/'+'1_'+files[i][:-4]+'_'+str(j+1)+'.wav',fs,output)
#            wavfile.write(dir_final_audio+'2_'+files[i][:-4]+'/'+'2_'+files[i][:-4]+'_'+str(j+1)+'.wav',fs,output)
#        cont_dur += 1

for k in range(n_sessions):
    print('Session'+str(k+1))
    dir_audio = 'D:/IEMOCAP_full_release/IEMOCAP/Audio_mono/Session'+str(k+1)+'/'
    dir_final_audio = 'D:/IEMOCAP_full_release/IEMOCAP/Audio_window_mono_66ms_centered/Session'+str(k+1)+'/'
#    os.mkdir(dir_final_audio[:-1])
    files = os.listdir(dir_audio)
    for i in range(len(files)):
        print('File'+' '+str(i+1)+'/'+str(len(files)))
        fs, audio = wavfile.read(dir_audio+files[i])
        real_length = float(txt[cont_dur])
        window_len = int(fs*t_window)-1
        window_len_2 = int(fs*t_window_2)
        n_div = int(real_length/(t_window_2))
        cont = 0
#        os.mkdir(dir_final_audio+'1_'+files[i][:-4])
#        os.mkdir(dir_final_audio+'2_'+files[i][:-4])
        for j in range(n_div):
            if j == 0 or j == 1 or j == n_div-1 or j == n_div-2:
                output = audio[0:10]
            else:
                output = audio[int(cont-window_len/2):int(window_len/2+cont)]
            cont += int(window_len_2)
            wavfile.write(dir_final_audio+'1_'+files[i][:-4]+'/'+'1_'+files[i][:-4]+'_'+'{:06d}'.format(j+1)+'.wav',fs,output)
            wavfile.write(dir_final_audio+'2_'+files[i][:-4]+'/'+'2_'+files[i][:-4]+'_'+'{:06d}'.format(j+1)+'.wav',fs,output)
        cont_dur += 1
