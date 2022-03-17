import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from kneed import KneeLocator

path1 = '3rd21st_CharadesEgo_vggsound_features/'
cls_wise_feature_list = []
name_list = []
for ii in range(157):
    path2 = path1 + str(ii) + '/'
    file_list = os.listdir(path2)
    name_list.append(file_list)
    feat_list = []
    for file in file_list:
        if file[5:8] != 'EGO':
            continue
        feat1 = np.load(path2+file)
        feat_list.append(feat1)
    feat_list = np.concatenate(feat_list, axis=0)
    cls_wise_feature_list.append(feat_list)

pred_cluster = []
cluster_centers = []
for ii in range(157):
    pred_cluster.append([])
cluster_num_list = []
for label_id in range(157):
    feature1 = cls_wise_feature_list[label_id]
    loss_list = []
    for cluster_num in range(1, 10):
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

save_path = 'audio_clusters/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
cluster_num_list = np.array(cluster_num_list)
np.save(save_path + 'cluster_num2.npy', cluster_num_list)
save_path2 = save_path + 'clusters_per_cls2/'
if not os.path.exists(save_path2):
    os.mkdir(save_path2)
for label_id in range(157):
    data1 = name_list[label_id]
    np.save(save_path+'%02d_centers2.npy'%label_id, cluster_centers[label_id])
    save_path1 = save_path + '%02d/' % label_id
    if not os.path.exists(save_path1):
        os.mkdir(save_path1)
    record = np.zeros((len(cluster_centers[label_id], )))
    for i, sample1 in enumerate(data1):
        cluster1 = pred_cluster[label_id][i]
        record[cluster1] += 1
        cluster1 = np.array([cluster1, ])
        np.save(save_path1 + sample1, cluster1)
    np.save(save_path2+str(label_id)+'.npy', record)