import os
import numpy as np

other = 0
n_sessions = 5
labels_count = np.zeros((4))
for i in range(n_sessions):
    dir_labels = 'D:/IEMOCAP_full_release/IEMOCAP/Labels_4'
    dir_labels_s = dir_labels+'/Session'+str(i+1)
    files = os.listdir(dir_labels_s)
    for j in range(len(files)):
        dir_label_s = dir_labels_s+'/'+files[j]
        f = open(dir_label_s, "r")
        lines = []
        for line in f:
            lines.append(line)
        lines = lines[1:]
        
        for line in lines:
            label = line.split('\t')[1]
            if label == 'hap':
                labels_count[0] += 1
            elif label == 'ang':
                labels_count[1] += 1
            elif label == 'sad':
                labels_count[2] += 1
            elif label == 'neu':
                labels_count[3] += 1
            else:
                other += 1