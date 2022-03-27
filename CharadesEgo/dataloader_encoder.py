from mmaction.datasets.pipelines import Compose
import cv2
import torch.utils.data
import csv
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np
import os
import random
import glob
from PIL import Image

class CharadesEgoTraining(torch.utils.data.Dataset):
    def __init__(self, split='train', source_domain='3rd', target_domain='1st', modality='rgb', cfg=None, use_audio=True, beta=0.99):
        self.base_path = '/local-ssd/yzhang9/data/CharadesEgo/'
        self.video_list = []

        with open(self.base_path+"CharadesEgo_v1_%s_only%s.csv"%(split, source_domain)) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i ==0:
                    continue
                if os.path.exists(self.base_path+"Labels/"+row[0]) and len(os.listdir(self.base_path+"Labels/"+row[0])) > 0:
                    self.video_list.append(row[0])


        self.source_domain = source_domain
        self.target_domain = target_domain
        self.split = split
        self.modality = modality
        num_per_cls = np.load("per_class_freq.npy") * len(self.video_list)
        effective_num = 1.0 - np.power(beta, num_per_cls)
        label_weights = (1.0 - beta) / effective_num
        label_weights = label_weights/np.sum(label_weights) * 157.0

        cluster_num = np.load('audio_clusters/cluster_num.npy')

        num_per_cluster = []
        effective_num = []
        weights = []
        for i in range(157):
            num_per_cluster.append([])
            effective_num.append([])
            weights.append([])
            sub_cluster_num = np.load('audio_clusters/clusters_per_cls/'+str(i)+'.npy')
            for jj in range(cluster_num[i]):
                num_per_cluster[i].append(int(sub_cluster_num[jj]))
                effective_num[i].append(1.0-np.power(beta, int(sub_cluster_num[jj])))
                weights[i].append((1.0-beta)/effective_num[i][jj])
        sum_weights = []
        for i in range(157):
            sum_weights.append(np.sum(weights[i]))
        #print(sum_weights)
        for i in range(157):
            for jj in range(cluster_num[i]):
                weights[i][jj] = weights[i][jj] / sum_weights[i] * cluster_num[i] * label_weights[i]
        print(weights, label_weights)
        video_list = []
        with open(self.base_path+"CharadesEgo_v1_%s_only%s.csv" % (split, target_domain)) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i == 0:
                    continue
                class_list1 = row[-4]
                class_list1 = class_list1.split(";")
                if len(class_list1[0]) > 0:
                    video_list.append(row[0])
        f.close()
        self.target_video_list = video_list
        self.num_target_videos = int(len(video_list)/2)
        self.target_video_list2 = self.target_video_list[self.num_target_videos:]
        self.target_video_list = self.target_video_list[:self.num_target_videos]

        self.cls_num = cluster_num
        self.weights = weights
        self.label_weights = label_weights

        # build the data pipeline
        if split == 'train':
            train_pipeline = cfg.data.train.pipeline
            self.train_pipeline = Compose(train_pipeline)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.val_pipeline = Compose(val_pipeline)

        self.cfg = cfg
        self.interval = 9

    def get_spec_piece(self, samples, start_time, end_time, duration, samplerate):
        start1 = start_time / duration * len(samples)
        end1 = end_time / duration * len(samples)
        start1 = int(np.round(start1))
        end1 = int(np.round(end1))
        samples = samples[start1:end1]

        resamples = samples[:160000]
        if len(resamples) == 0:
            resamples = np.zeros((160000))
        while len(resamples) < 160000:
            resamples = np.tile(resamples, 10)[:160000]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
        spectrogram = spectrogram + noise
        start1 = np.random.choice(256 - self.interval, (1,))[0]
        spectrogram[start1:(start1 + self.interval), :] = 0

        return spectrogram

    def __getitem__(self, index):
        #index = np.random.choice(len(self.video_list), (1,))[0]
        video_path = self.base_path +'CharadesEgo_v1_rgb/'+self.video_list[index]+"/"+self.video_list[index] +'-'
        label_path = self.base_path + "Labels/"+self.video_list[index]+"/"
        label_list = os.listdir(label_path)
        id1 = np.random.choice(len(label_list), (1,))[0]
        frame_list = []
        class_list = []
        with open(label_path + label_list[id1]) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                frame_list.append(row[0])
                class_list.append(row[1])
        f.close()

        start_frame = int(frame_list[0])
        end_frame = int(frame_list[-1])

        filename_tmpl = self.cfg.data.train.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.train.get('modality', 'RGB')
        start_index = self.cfg.data.train.get('start_index', start_frame)
        data = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data, frame_inds = self.train_pipeline(data)

        label1 = np.zeros((157))
        for i in frame_inds:
            tmp = class_list[i - start_frame].split(' ')
            for ttt in tmp:
                tt = ttt[1:]
                label1[int(tt)] += 1
        label1 = (label1 > 0).astype(np.float32)
        label_list = np.arange(157)[label1==1]
        label_weight1 = np.zeros((157,))
        for jjj in range(157):
            if jjj in list(label_list):
                file_list = glob.glob('audio_clusters/'+str(jjj)+'/'+self.video_list[index] +'*')
                for file_name in file_list:
                    start1 = int(file_name[-17:-11])
                    end1 = int(file_name[-10:-4])
                    if start1 <= frame_inds[0] and end1 > frame_inds[0]:
                        cluster_id = int(np.load(file_name)[0])
                        weight11 = self.weights[jjj][cluster_id]
                        label_weight1[jjj] = weight11
                        break
                if label_weight1[jjj] == 0:
                    label_weight1[jjj] = self.label_weights[jjj]
            else:
                label_weight1[jjj] = 0#self.label_weights[jjj]
        label_weight1 = np.sum(label_weight1) / len(label_list)


        #-------------------------------------------------------------------------------------------------------------
        audio_path = self.base_path + 'audio/' + self.video_list[index] + '.wav'
        start_time = frame_inds[0] / 24.0
        end_time = frame_inds[-1] / 24.0
        samples, samplerate = sf.read(audio_path)
        duration = len(samples) / samplerate
        spectrogram = self.get_spec_piece(samples, start_time, end_time, duration, samplerate)
        #-------------------------------------------------------------------------------------------------------------

        target_index = np.random.choice(self.num_target_videos, (1,))[0]

        video_path = self.base_path +'CharadesEgo_v1_rgb/'+self.target_video_list[target_index]+"/"+self.target_video_list[target_index] +'-'
        label_path = self.base_path + "Labels/"+self.target_video_list[target_index]+"/"
        label_list = os.listdir(label_path)
        id1 = np.random.choice(len(label_list), (1,))[0]
        frame_list = []
        class_list = []
        with open(label_path + label_list[id1]) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                frame_list.append(row[0])
                class_list.append(row[1])
        f.close()

        start_frame = int(frame_list[0])
        end_frame = int(frame_list[-1])

        filename_tmpl = self.cfg.data.train.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.train.get('modality', 'RGB')
        start_index = self.cfg.data.train.get('start_index', start_frame)
        target_data = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        target_data, target_frame_inds = self.train_pipeline(target_data)

        target_label1 = np.zeros((157))
        for i in target_frame_inds:
            tmp = class_list[i - start_frame].split(' ')
            for ttt in tmp:
                tt = ttt[1:]
                target_label1[int(tt)] += 1
        target_label1 = (target_label1 > 0).astype(np.float32)

        audio_path = self.base_path + 'audio/' + self.target_video_list[target_index] + '.wav'
        start_time = target_frame_inds[0] / 24.0
        end_time = target_frame_inds[-1] / 24.0
        samples, samplerate = sf.read(audio_path)
        duration = len(samples) / samplerate
        target_spectrogram = self.get_spec_piece(samples, start_time, end_time, duration, samplerate)

        #---------------------------------------------------------------------------------------------

        target_index2 = np.random.choice(len(self.target_video_list2), (1,))[0]

        video_path = self.base_path +'CharadesEgo_v1_rgb/'+self.target_video_list2[target_index2]+"/"+self.target_video_list2[target_index2] +'-'
        #label_path = self.base_path + "Labels/"+self.target_video_list2[target_index2]+"/"
        #label_list = os.listdir(label_path)
        #id1 = np.random.choice(len(label_list), (1,))[0]
        #frame_list = []
        #class_list = []
        #with open(label_path + label_list[id1]) as f:
        #    f_csv = csv.reader(f)
        #    for i, row in enumerate(f_csv):
        #        frame_list.append(row[0])
        #        class_list.append(row[1])
        #f.close()

        #start_frame = int(frame_list[0])
        #end_frame = int(frame_list[-1])
        start_frame = 1
        total_frames = len(os.listdir(self.base_path +'CharadesEgo_v1_rgb/'+self.target_video_list2[target_index2]+"/"))

        filename_tmpl = self.cfg.data.train.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.train.get('modality', 'RGB')
        start_index = self.cfg.data.train.get('start_index', start_frame)
        u_target_data = dict(
            frame_dir=video_path,
            total_frames=total_frames,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        u_target_data, u_target_frame_inds = self.train_pipeline(u_target_data)

        try:
            unlikely_label = np.load("unlikely_preds/"+self.target_video_list2[target_index2]+'.npy')
        except:
            unlikely_label = np.zeros((157,))

        # likely_label = np.load("likely_visual_preds10/" + self.target_video_list2[target_index2] + '.npy')
        # likely_label = likely_label *(1-unlikely_label)

        audio_path = self.base_path + 'audio/' + self.target_video_list2[target_index2] + '.wav'
        start_time = u_target_frame_inds[0] / 24.0
        end_time = u_target_frame_inds[-1] / 24.0
        samples, samplerate = sf.read(audio_path)
        duration = len(samples) / samplerate
        target_spectrogram2 = self.get_spec_piece(samples, start_time, end_time, duration, samplerate)

        return data, label1, spectrogram,label_weight1, target_data, target_label1, target_spectrogram, u_target_data, target_spectrogram2, unlikely_label

    def __len__(self):
        return len(self.video_list)


class CharadesEgoValidating(torch.utils.data.Dataset):
    def __init__(self, split='test', domain='3rd',  modality='rgb', cfg=None,):
        self.base_path = '/local-ssd/yzhang9/data/CharadesEgo/'
        self.video_list = []

        with open(self.base_path + "CharadesEgo_v1_%s_only%s.csv" % (split, domain)) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i == 0:
                    continue
                if os.path.exists(self.base_path + "Labels/" + row[0]) and len(
                        os.listdir(self.base_path + "Labels/" + row[0])) > 0:
                    self.video_list.append(row[0])

        len1 = int(len(self.video_list)/2)
        self.video_list = self.video_list[len1:]
        self.domain = domain
        self.split = split
        self.modality = modality


        # build the data pipeline
        if split == 'train':
            train_pipeline = cfg.data.train.pipeline
            self.train_pipeline = Compose(train_pipeline)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.val_pipeline = Compose(val_pipeline)

        self.cfg = cfg
        self.interval = 9

    def __getitem__(self, index):
        video_path = self.base_path + 'CharadesEgo_v1_rgb/' + self.video_list[index] + "/" + self.video_list[
            index] + '-'
        label_path = self.base_path + "Labels/" + self.video_list[index] + "/"
        label_list = os.listdir(label_path)
        id1 = np.random.choice(len(label_list), (1,))[0]
        frame_list = []
        class_list = []
        with open(label_path + label_list[id1]) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                frame_list.append(row[0])
                class_list.append(row[1])
        f.close()

        start_frame = int(frame_list[0])
        end_frame = int(frame_list[-1])

        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', start_frame)
        data = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data, frame_inds = self.val_pipeline(data)

        label1 = np.zeros((157))
        for i in frame_inds:
            tmp = class_list[i - start_frame].split(' ')
            for ttt in tmp:
                tt = ttt[1:]
                label1[int(tt)] += 1
        label1 = (label1 > 0).astype(np.float32)


        # -------------------------------------------------------------------------------------------------------------
        audio_path = self.base_path + 'audio/' + self.video_list[index] + '.wav'

        start_time = frame_inds[0] / 24.0
        end_time = frame_inds[-1] / 24.0

        samples, samplerate = sf.read(audio_path)

        duration = len(samples) / samplerate

        start1 = start_time / duration * len(samples)
        end1 = end_time / duration * len(samples)
        start1 = int(np.round(start1))
        end1 = int(np.round(end1))
        samples = samples[start1:end1]

        resamples = samples[:160000]
        if len(resamples) == 0:
            resamples = np.zeros((160000))
        while len(resamples) < 160000:
            resamples = np.tile(resamples, 10)[:160000]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        #noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
        #spectrogram = spectrogram + noise
        #start1 = np.random.choice(256 - self.interval, (1,))[0]
        #spectrogram[start1:(start1 + self.interval), :] = 0
        # -------------------------------------------------------------------------------------------------------------

        return data, label1, spectrogram,0,0,0,0,0,0,0

    def __len__(self):
        return len(self.video_list)


class CharadesEgoTesting(torch.utils.data.Dataset):
    def __init__(self, split='train', domain='1st',  modality='rgb', cfg=None,):
        self.base_path = '/local-ssd/yzhang9/data/CharadesEgo/'
        self.video_list = []

        with open(self.base_path + "CharadesEgo_v1_%s_only%s.csv" % (split, domain)) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i == 0:
                    continue
                if i == 0:
                    continue
                if os.path.exists(self.base_path + "Labels/" + row[0]) and len(
                        os.listdir(self.base_path + "Labels/" + row[0])) > 0:
                    self.video_list.append(row[0])

        self.domain = domain
        self.split = split
        self.modality = modality


        # build the data pipeline

        test_pipeline = cfg.data.test.pipeline
        self.val_pipeline = Compose(test_pipeline)

        self.cfg = cfg
        self.interval = 9

    def __getitem__(self, index):
        video_path = self.base_path + 'CharadesEgo_v1_rgb/' + self.video_list[index] + "/" + self.video_list[
            index] + '-'
        label_path = self.base_path + "Labels/" + self.video_list[index] + "/"
        label_list = os.listdir(label_path)
        id1 = np.random.choice(len(label_list), (1,))[0]
        frame_list = []
        class_list = []
        with open(label_path + label_list[id1]) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                frame_list.append(row[0])
                class_list.append(row[1])
        f.close()

        start_frame = int(frame_list[0])
        end_frame = int(frame_list[-1])

        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', start_frame)
        data = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data, frame_inds = self.val_pipeline(data)

        label1 = np.zeros((157))
        for i in frame_inds:
            tmp = class_list[i - start_frame].split(' ')
            for ttt in tmp:
                tt = ttt[1:]
                label1[int(tt)] += 1
        label1 = (label1 > 0).astype(np.float32)


        # -------------------------------------------------------------------------------------------------------------

        frame_inds = frame_inds.reshape((5, -1))

        audio_path = self.base_path + 'audio/' + self.video_list[index] + '.wav'
        samples, samplerate = sf.read(audio_path)

        spec_list = []
        for iii in range(5):
            start_time = frame_inds[iii][0] / 24.0
            end_time = frame_inds[iii][-1] / 24.0

            duration = len(samples) / samplerate
            spec1 = self.get_spec_piece(start_time, end_time, samples, samplerate, duration)
            spec_list.append(spec1)

        spec_list = np.stack(spec_list)
        # -------------------------------------------------------------------------------------------------------------

        return data, label1, spec_list, self.video_list[index]

    def get_spec_piece(self,start_time, end_time, samples, samplerate, duration):
        start1 = start_time / duration * len(samples)
        end1 = end_time / duration * len(samples)
        start1 = int(np.round(start1))
        end1 = int(np.round(end1))
        samples = samples[start1:end1]

        resamples = samples[:160000]
        if len(resamples) == 0:
            resamples = np.zeros((160000))
        while len(resamples) < 160000:
            resamples = np.tile(resamples, 10)[:160000]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        return spectrogram
    def __len__(self):
        return len(self.video_list)


