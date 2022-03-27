import csv
import os
import numpy as np
import cv2

base_path = '/home/yzhang8/data/CharadesEgo/CharadesEgo_v1_test_only3rd.csv'
video_base_path = "/home/yzhang8/data/CharadesEgo/CharadesEgo_v1_rgb/"

video_list = []
start_list = []
end_list = []
class_list = []
save_path = "/home/yzhang8/data/CharadesEgo/Labels/"
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(base_path) as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        if i == 0:
            continue
        segment_list = row[9]
        segment_list = segment_list.split(';')
        if len(segment_list[0]) == 0:
            continue
        #cap = cv2.VideoCapture(video_base_path+row[0]+".mp4")
        fps = 24
        segment_list2 = []
        for ii, seg in enumerate(segment_list):
            seg1 = seg.split(' ')
            start1 = int(np.round(float(seg1[1])*fps))+1
            end1 = int(np.round(float(seg1[2]) * fps))+1
            
            
            if start1 < end1:
                segment_list2.append([seg1[0], start1, end1])
            else:
                #print(row)
                segment_list2.append([seg1[0], end1, start1])

        save_path1 = save_path + row[0] + '/'
        if not os.path.exists(save_path1):
            os.mkdir(save_path1)

        num_frames = len(os.listdir("/home/yzhang8/data/CharadesEgo/CharadesEgo_v1_rgb/"+row[0]))
        frame_wise_label = []
        frame_wise_label1 = []
        for j in range(1, num_frames+1):
            class1 = []
            for ii in range(len(segment_list2)):
                start1 = segment_list2[ii][1]
                end1 = segment_list2[ii][2]
                if start1 <= j <= end1:
                    class1.append(segment_list2[ii][0])
            label = ''
            for class2 in class1:
                label += class2 + ' '
            label = label[:-1]
            if len(frame_wise_label1) > 0 and len(label) == 0:
                frame_wise_label.append(frame_wise_label1)
                frame_wise_label1 = []
            elif len(label) ==0:
                continue
            else:
                frame_wise_label1.append([j, label])
        if len(frame_wise_label1) >0:
            frame_wise_label.append(frame_wise_label1)
        if len(frame_wise_label) == 0:
            continue
        for frame_wise_label1 in frame_wise_label:
            save_path2 = save_path1 + 'frame_%010d_%010d.csv'%(frame_wise_label1[0][0], frame_wise_label1[-1][0])
            with open(save_path2, 'a') as f2:
                for kk in range(len(frame_wise_label1)):
                    f2.write('{},{}\n'.format(frame_wise_label1[kk][0], frame_wise_label1[kk][1]))
                    f2.flush()
            f2.close()

f.close()

