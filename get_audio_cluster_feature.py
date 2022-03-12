import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from kneed import KneeLocator
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--source_domain', type=str, help='input a str', default='D1')
parser.add_argument('--target_domain', type=str, help='input a str', default='D2')
args = parser.parse_args()

test_file = pd.read_pickle('/home/xxx/data/EPIC_KITCHENS_UDA/' + args.domain + "_train.pkl")
feature_wise_data = []
for ii in range(8):
    feature_wise_data.append([])
class_dict = {}
for _, line in test_file.iterrows():
    image = [args.source_domain + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
             line['stop_timestamp']]
    labels = line['verb_class']
    feature_wise_data[int(labels)].append([image[0], image[1], image[2], image[3], image[4], int(labels)])

cls_wise_feature_list = []
for ii in range(8):
    cls_wise_feature_list.append([])
for ii in range(8):
    for file_id, image in enumerate(feature_wise_data[ii]):
        video_id = image[0].split("/")[-1]
        start_index = int(np.ceil(image[1] / 2))
        feat1 = np.load(args.source_domain + '_sound_features/' + video_id + "_%010d.npy" % start_index)
        cls_wise_feature_list[ii].append(feat1)

for ii in range(8):
    cls_wise_feature_list[ii] = np.stack(cls_wise_feature_list[ii])

pred_cluster = []
cluster_centers = []
for ii in range(8):
    pred_cluster.append([])
cluster_num_list = []
for label_id in range(8):
    feature1 = cls_wise_feature_list[label_id]
    loss_list = []
    for cluster_num in range(1, 21):
        kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(feature1)
        pred_y = kmeans.fit_predict(feature1)
        loss_list.append(kmeans.inertia_)
    x = range(1, len(loss_list) + 1)
    kn = KneeLocator(x, loss_list, curve='convex', direction='decreasing')
    # print(label_id, kn.knee)
    kmeans = KMeans(n_clusters=kn.knee, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(feature1)
    source_cluster_centers = kmeans.cluster_centers_
    cluster_centers.append(source_cluster_centers)
    pred_y = kmeans.fit_predict(feature1)
    pred_cluster[label_id] = pred_y
    cluster_num_list.append(kn.knee)

save_path = args.source_domain + '_audio_clusters/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
cluster_num_list = np.array(cluster_num_list)
np.save(save_path + 'cluster_num.npy', cluster_num_list)
for label_id in range(8):
    data1 = feature_wise_data[label_id]
    np.save(save_path+'%02d_centers.npy'%label_id, cluster_centers[label_id])
    save_path1 = save_path + '%02d/' % label_id
    if not os.path.exists(save_path1):
        os.mkdir(save_path1)
    for i, sample1 in enumerate(data1):
        cluster1 = pred_cluster[label_id][i]
        cluster1 = np.array([cluster1, ])
        video_id = sample1[0].split("/")[-1]
        start_index = int(np.ceil(sample1[1] / 2))
        np.save(save_path1 + video_id + "_%010d.npy" % start_index, cluster1)
