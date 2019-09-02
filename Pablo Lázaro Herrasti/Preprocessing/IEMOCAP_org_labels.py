# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:30:04 2019

@author: polaz
"""

import os

frame_seg = 29.97
n_sessions = 5

for i in range(n_sessions):
    print("Session",i+1)
    dir_labels = 'D:/IEMOCAP_full_release/IEMOCAP/Labels'
    dir_labels_s = dir_labels+'/Session'+str(i+1)
    files = os.listdir(dir_labels_s)
    for j in range(len(files)):
        dir_label_s = dir_labels_s+'/'+files[j]
        f = open(dir_label_s, "r")
        lines = []
        for line in f:
            lines.append(line)
        lines = lines[2:]
        ## PERSON IN THE LEFT
        labels = []
        time1_list = []
        time2_list = []
        for line in lines:
            line_list = line.split('\t')
            if line[0] == '[' and line_list[1][-4] == 'F':
                line_list = line.split('\t')
                label = line_list[2]
                times = line_list[0].split(' ')
                time1 = times[0][1:]
                time2 = times[2][:-1]
                if label == 'xxx' or label == 'fru' or label == 'sur' or label == 'fea' or label == 'oth' or label == 'dis':
                    pass
                else:
                    if label == 'exc':
                        label = 'hap'
                    labels.append(label)
                    time1_list.append(time1)
                    time2_list.append(time2)
                    
            else: 
                pass
        utt = 1
        dir_labels_f = 'D:/IEMOCAP_full_release/IEMOCAP/Labels_4/Session'+str(i+1)+'/1_'+files[j]
        c = 0
        with open(dir_labels_f, 'w') as f:
            f.write("NUMBER\tLABEL\tSTART_T\tFINAL_T\n")
            for item in labels:
                utter = 'Utterance'+str(utt)
                f.write("%s\t%s\t%s\t%s\n" % (utter,item,str(time1_list[c]),str(time2_list[c])))
                utt += 1
                c += 1
        
        ## PERSON IN THE RIGHT
        labels = []
        time1_list = []
        time2_list = []
        for line in lines:
            line_list = line.split('\t')
            if line[0] == '[' and line_list[1][-4] == 'M':
                line_list = line.split('\t')
                label = line_list[2]
                times = line_list[0].split(' ')
                time1 = times[0][1:]
                time2 = times[2][:-1]
                if label == 'xxx' or label == 'fru' or label == 'sur' or label == 'fea' or label == 'oth' or label == 'dis':
                    pass
                else:
                    if label == 'exc':
                        label = 'hap'
                    labels.append(label)
                    time1_list.append(time1)
                    time2_list.append(time2)
                    
            else: 
                pass
        utt = 1
        dir_labels_f = 'D:/IEMOCAP_full_release/IEMOCAP/Labels_4/Session'+str(i+1)+'/2_'+files[j]
        c = 0
        with open(dir_labels_f, 'w') as f:
            f.write("NUMBER\tLABEL\tSTART_T\tFINAL_T\n")
            for item in labels:
                utter = 'Utterance'+str(utt)
                f.write("%s\t%s\t%s\t%s\n" % (utter,item,str(time1_list[c]),str(time2_list[c]),))
                utt += 1
                c += 1