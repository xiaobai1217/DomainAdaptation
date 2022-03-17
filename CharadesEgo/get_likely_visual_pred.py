import csv
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_id', type=int, default=1)
args = parser.parse_args()

base_path = '/home/yzhang8/data/CharadesEgo/'

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
num_videos = int(len(video_list)/2)
video_list = video_list[num_videos:]
class_list = class_list[num_videos:]
num_videos = len(video_list)
records = np.zeros((num_videos, 157))
count = 0
acc = 0
class_wise_acc = np.zeros((157,))
class_wise_count = np.zeros((157,))
for ii in range(157):
    print(ii)
    pred_list = []
    likely_videos_num = int(per_cls_freq[ii] * num_videos * 0.05)

    for i, sample1 in enumerate(video_list):
        path1 = '3rd21st_CharadesEgo_reweighting_att_on_train%02d/'%args.save_id + sample1 + '.npy'
        if not os.path.isfile(path1):
            pred1 = np.ones((157,))
        else:
            pred1 = np.load(path1)
        pred_list.append(pred1[ii])
    pred_list = np.array(pred_list)
    idx_list = np.argsort(pred_list)
    idx_list = idx_list[::-1]

    #idx_list = idx_list[:likely_videos_num]
    for jj, id in enumerate(idx_list):
        
        pred1 = np.load('3rd21st_CharadesEgo_reweighting_att_on_train%02d/'%args.save_id +video_list[id] + '.npy')
        if pred1[ii] > 0.8 or jj < likely_videos_num:
            count += 1
            class_wise_count[ii] += 1
            if ii in class_list[id]:
                acc += 1
                class_wise_acc[ii] += 1
            records[id, ii] = 1

print(acc / count)
print(np.sum(count))
print(class_wise_acc/class_wise_count)
print(class_wise_acc)
save_path = 'likely_visual_preds%02d/'%args.save_id
if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, sample1 in enumerate(video_list):
    np.save(save_path+ sample1+'.npy', records[i])



