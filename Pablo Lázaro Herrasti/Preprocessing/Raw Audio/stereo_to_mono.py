from scipy.io import wavfile
import os
import gc
from pydub import AudioSegment

n_sessions = 5
dir_txt = 'D:/IEMOCAP_full_release/IEMOCAP/Video/'
cont = 0

for k in range(n_sessions):
    print('Session'+str(k+1))
    dir_audio = 'D:/IEMOCAP_full_release/IEMOCAP/Audio/Session'+str(k+1)+'/'
    dir_final_audio = 'D:/IEMOCAP_full_release/IEMOCAP/Audio_mono/Session'+str(k+1)+'/'
    os.mkdir(dir_final_audio[:-1])
    files = os.listdir(dir_audio[:-1])
    for i in range(len(files)):
        print('File'+' '+str(i+1)+'/'+str(len(files)))
        dir_file = dir_audio+'/'+files[i]
        mysound = AudioSegment.from_wav(dir_file)
        mysound = mysound.set_channels(1)
        mysound.export(dir_final_audio[:-1]+'/'+files[i], format="wav")
        mysound.export()