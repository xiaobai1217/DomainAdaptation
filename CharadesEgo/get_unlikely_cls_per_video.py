import csv
import os
import numpy as np

base_path = '/local-ssd/yzhang9/data/CharadesEgo/'

video_list = []
class_list = []
with open(base_path + "CharadesEgo_v1_%s_only%s.csv" % ('train', '1st')) as f:
#with open("CharadesEgo_v1_%s_only%s.csv" % ('train', '1st')) as f:
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

per_cls_freq = np.load("per_class_freq.npy")
num_videos = int(len(video_list) / 2)
video_list = video_list[num_videos:]
class_list = class_list[num_videos:]
num_videos = len(video_list)
records = np.zeros((num_videos, 157))
for ii in range(157):
    pred_list = []
    unlikely_videos_num = int((1-per_cls_freq[ii]) * num_videos * 0.55)
    print(ii)  
    for i, sample1 in enumerate(video_list):
        path1 = '3rd21st_CharadesEgo_audio_on_train/' + sample1 + '.npy'
        if not os.path.isfile(path1):
            pred1 = np.ones((157,))
            #print("missing")
        else:
            #print("yes")
            pred1 = np.load(path1)
        pred_list.append(pred1[ii])
    pred_list = np.array(pred_list)
    idx_list = np.argsort(pred_list)

    idx_list = idx_list[:unlikely_videos_num]
    for id in idx_list:
        records[id, ii] = 1


save_path = 'unlikely_preds/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, sample1 in enumerate(video_list):
    np.save(save_path+ sample1+'.npy', records[i])



