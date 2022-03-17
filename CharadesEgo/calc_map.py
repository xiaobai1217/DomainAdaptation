import argparse
import numpy as np
import csv
from util1 import AveragePrecisionMeter

parser = argparse.ArgumentParser()
parser.add_argument('--source_domain', type=str, help='input a str', default='3rd')
parser.add_argument('--target_domain', type=str, help='input a str', default='1st')
args = parser.parse_args()

base_path = '/home/yzhang8/data/CharadesEgo/'

video_list = []
class_list = []
with open(base_path + "CharadesEgo_v1_%s_only%s.csv" % ('test', '1st')) as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        if i==0:
            continue
        class_list1 = row[-4]
        class_list1 = class_list1.split(";")
        if len(class_list1[0])>0:
            tmp  = []
            for item in class_list1:
                tmp.append(int((item.split(" "))[0][1:]))
            class_list.append(tmp)
            video_list.append(row[0])
f.close()
ap_meter = AveragePrecisionMeter(False)

ap_meter.reset()

save_path = args.source_domain + '2' + args.target_domain + '_CharadesEgo_projection_07/'
for i, sample1 in enumerate(video_list):
    #try:
    predict1 = np.load(save_path + sample1+".npy")
    #except:
    #    continue
    label1 = class_list[i]
    labels = np.zeros((157,))
    for tmp in label1:
        labels[tmp] = 1
    ap_meter.add(predict1, labels)


map = 100 * ap_meter.value().mean()
print(map)
