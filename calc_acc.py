import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--source_domain', type=str, help='input a str', default='D3')
parser.add_argument('--target_domain', type=str, help='input a str', default='D2')
args = parser.parse_args()

base_path = '/home/xxx/data/EPIC_KITCHENS_UDA/frames_rgb_flow/flow/'
test_file = pd.read_pickle(args.target_domain + "_test.pkl")


data1 = []
class_dict = {}
for _, line in test_file.iterrows():
    image = [args.target_domain + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
             line['stop_timestamp']]
    labels = line['verb_class']
    data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
    if line['verb'] not in list(class_dict.keys()):
        class_dict[line['verb']] = line['verb_class']

acc = 0
for i, sample1 in enumerate(data1):
    video_id = sample1[0].split("/")[-1]
    start_index = int(np.ceil(sample1[1]/2))
    pred1 = np.load("preds_flow/"+ '%s2%s_transformer4cls/'%(args.source_domain, args.target_domain)+video_id+ "_%010d.npy"%start_index)
    start_index = int(sample1[1])
    pred2 = np.load("preds_rgb/"+ '%s2%s_transformer4cls/'%(args.source_domain, args.target_domain)+video_id+ "_%010d.npy"%start_index)

    pred = (pred1 + pred2) / 2.0

    if np.argmax(pred) == sample1[-1]:
        acc+=1

print( acc / len(data1))
