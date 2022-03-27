import csv
import os
import numpy as np

base_path = '/local-ssd/yzhang9/data/CharadesEgo/'

video_list = []
class_list = []
with open(base_path + "CharadesEgo_v1_%s_only%s.csv" % ('train', '3rd')) as f:
#with open("CharadesEgo_v1_%s_only%s.csv" % ('train', '3rd')) as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        if i==0:
            continue
        class_list1 = row[-4]
        class_list1 = class_list1.split(";")
        if len(class_list1[0])>0:
            tmp = []
            for item in class_list1:
                tmp.append(int((item.split(" "))[0][1:]))
            class_list.append(tmp)
            video_list.append(row[0])
f.close()


records = np.zeros((157,))
for i, list1 in enumerate(class_list):
    for ii in list1:
        records[ii] += 1

print(records/len(class_list))
np.save("per_class_freq.npy", records/len(class_list))
