from mmaction.datasets.pipelines import Compose
import cv2
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np
import csv
import random
import pdb

class EPICDOMAINClusters(torch.utils.data.Dataset):
    def __init__(self, split='train', source_domain='D1', target_domain = 'D2', beta=0.999, cfg=None):
        self.base_path = '/home/xxx/data/EPIC_KITCHENS_UDA/'
        target_train_file = pd.read_pickle('/home/xxx/data/EPIC_KITCHENS_UDA/'+target_domain+"_train.pkl")
        source_train_file = pd.read_pickle('/home/xxx/data/EPIC_KITCHENS_UDA/'+source_domain+"_train.pkl")
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.split = split
        self.interval = 9

        cls_wise_data = []
        cluster_num = np.load('audio_clusters/'+source_domain+'/cluster_num.npy')
        new_label_map = []
        count = 0
        for ii in range(8):
            cls_wise_data.append([])
            new_label_map.append([])
            for jj in range(cluster_num[ii]):
                new_label_map[ii].append(count)
                count+=1
                cls_wise_data[ii].append([])

        # build the data pipeline
        if split == 'train':
            train_pipeline = cfg.data.train.pipeline
            self.train_pipeline = Compose(train_pipeline)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.val_pipeline = Compose(val_pipeline)

        data1 = []
        num_per_cls = np.zeros((8,))
        for _, line in source_train_file.iterrows():
            image = [source_domain + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                     line['stop_timestamp']]
            labels = line['verb_class']
            num_per_cls[int(labels)] += 1
            video_id = image[0].split("/")[-1]
            start_index = int(np.ceil(image[1] / 2))
            cluster_id = np.load("audio_clusters/"+source_domain+'/%02d/'%int(labels) + video_id + "_%010d.npy"%start_index)[0]
            data1.append([image[0], image[1], image[2], image[3], image[4], int(labels), new_label_map[int(labels)][cluster_id], cluster_id])
            cls_wise_data[int(labels)][cluster_id].append([image[0], image[1], image[2], image[3], image[4], int(labels)])

        target_data1 = []
        for _, line in target_train_file.iterrows():
            image = [target_domain + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                     line['stop_timestamp']]
            labels = line['verb_class']
            target_data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))


        effective_num = 1.0 - np.power(beta, num_per_cls)
        label_weights = (1.0 - beta) / effective_num
        label_weights = label_weights/np.sum(label_weights) * 8.0

        num_per_cluster = []
        effective_num = []
        weights = []
        for i in range(8):
            num_per_cluster.append([])
            effective_num.append([])
            weights.append([])
            for jj in range(cluster_num[i]):
                num_per_cluster[i].append(len(cls_wise_data[i][jj]))
                effective_num[i].append(1.0-np.power(beta, len(cls_wise_data[i][jj])))
                weights[i].append((1.0-beta)/effective_num[i][jj])
        sum_weights = []
        for i in range(8):
            sum_weights.append(np.sum(weights[i]))
        #print(sum_weights)
        for i in range(8):
            for jj in range(cluster_num[i]):
                weights[i][jj] = weights[i][jj] / sum_weights[i] * cluster_num[i] * label_weights[i]
        print(cluster_num)
        print(weights)
        print(label_weights)


        self.samples = data1
        self.target_samples = target_data1
        self.cls_num = int(np.sum(cluster_num))
        idx_list = list(range(len(self.target_samples)))
        random.shuffle(idx_list)
        self.target_idx_list = idx_list
        self.cfg = cfg
        self.weights = weights
        self.label_weights = label_weights

    def get_spectrogram(self, sample1):
        audio_path = self.base_path + 'AudioVGGSound/' + self.split + '/' + sample1[0] + '.wav'
        samples, samplerate = sf.read(audio_path)

        duration = len(samples) / samplerate

        fr_sec = sample1[3].split(':')
        hour1 = float(fr_sec[0])
        minu1 = float(fr_sec[1])
        sec1 = float(fr_sec[2])
        fr_sec = (hour1 * 60 + minu1) * 60 + sec1

        stop_sec = sample1[4].split(':')
        hour1 = float(stop_sec[0])
        minu1 = float(stop_sec[1])
        sec1 = float(stop_sec[2])
        stop_sec = (hour1 * 60 + minu1) * 60 + sec1

        start1 = fr_sec / duration * len(samples)
        end1 = stop_sec / duration * len(samples)
        start1 = int(np.round(start1))
        end1 = int(np.round(end1))
        samples = samples[start1:end1]

        resamples = samples[:160000]
        while len(resamples) < 160000:
            resamples = np.tile(resamples, 10)[:160000]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        if self.split == 'train':
            noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
            spectrogram = spectrogram + noise
            start1 = np.random.choice(256 - self.interval, (1,))[0]
            spectrogram[start1:(start1 + self.interval), :] = 0

        return spectrogram

    def __getitem__(self, index):
        video_path = self.base_path + 'frames_rgb_flow/rgb/' + self.split + '/' + self.samples[index][0]
        spectrogram = self.get_spectrogram(self.samples[index])
        index2 = self.target_idx_list.pop(0)
        self.target_idx_list.append(index2)
        target_video_path = self.base_path + 'frames_rgb_flow/rgb/' + self.split + '/' + self.target_samples[index2][0]
        target_spectrogram = self.get_spectrogram(self.target_samples[index2])
        if self.split == 'train':
            filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
            modality = self.cfg.data.train.get('modality', 'RGB')
            start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1] ))
            data = dict(
                frame_dir=video_path,
                total_frames=int(self.samples[index][2] - self.samples[index][1]),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data = self.train_pipeline(data)

            start_index = self.cfg.data.train.get('start_index', int(self.target_samples[index2][1] ))
            target_data = dict(
                frame_dir=target_video_path,
                total_frames=int(self.target_samples[index2][2] - self.target_samples[index2][1]),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            target_data = self.train_pipeline(target_data)

            start_index = int(np.ceil(self.target_samples[index2][1] / 2))
            video_id = self.target_samples[index2][0].split("/")[-1]
            audio_predict = np.load(
                "audio_preds/%s2%s/" % (self.source_domain, self.target_domain) + video_id + "_%010d.npy" % start_index)
            idx_list = np.argsort(audio_predict)
            idx_list = idx_list[:3]
            target_label = np.zeros((8,))
            for id in idx_list:
                target_label[id] = 1

        data['imgs'] = torch.cat((data['imgs'], target_data['imgs']), dim=0)

        label1 = self.samples[index][-3]
        #cluster_label = self.samples[index][-2]
        #print(label1, self.weights, self.samples[index][-1])
        weight1 = self.weights[label1][self.samples[index][-1]]
        #label_weight = self.label_weights[label1]
        spectrogram = np.stack((spectrogram, target_spectrogram))

        return data,spectrogram.astype(np.float32), label1, target_label, weight1,

    def __len__(self):
        return len(self.samples)

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', domain='D1', modality='rgb', cfg=None, use_audio=True):
        self.base_path = '/home/xxx/data/EPIC_KITCHENS_UDA/'
        train_file = pd.read_pickle('/home/xxx/data/EPIC_KITCHENS_UDA/'+domain+"_"+split+".pkl")
        self.domain = domain
        self.split = split
        self.modality = modality
        self.use_audio = use_audio
        self.interval = 9

        # build the data pipeline
        if split == 'train':
            train_pipeline = cfg.data.train.pipeline
            self.train_pipeline = Compose(train_pipeline)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.val_pipeline = Compose(val_pipeline)

        data1 = []
        class_dict = {}
        for _, line in train_file.iterrows():
            image = [domain + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                     line['stop_timestamp']]
            labels = line['verb_class']
            data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
            if line['verb'] not in list(class_dict.keys()):
                class_dict[line['verb']] = line['verb_class']

        self.class_dict = class_dict
        self.samples = data1
        self.cfg = cfg

    def __getitem__(self, index):
        video_path = self.base_path +'frames_rgb_flow/rgb/'+self.split + '/'+self.samples[index][0]
        if self.split == 'train':
            filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
            modality = self.cfg.data.train.get('modality', 'RGB')
            start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1]))
            data = dict(
                frame_dir=video_path,
                total_frames=int(self.samples[index][2] - self.samples[index][1]),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data = self.train_pipeline(data)
        else:
            filename_tmpl = self.cfg.data.val.get('filename_tmpl', 'frame_{:010}.jpg')
            modality = self.cfg.data.val.get('modality', 'RGB')
            start_index = self.cfg.data.val.get('start_index', int(self.samples[index][1]))
            data = dict(
                frame_dir=video_path,
                total_frames=int(self.samples[index][2] - self.samples[index][1]),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data = self.val_pipeline(data)
        label1 = self.samples[index][-1]

        if self.use_audio is True:
            audio_path = self.base_path + 'AudioVGGSound/' + self.split + '/' + self.samples[index][0] + '.wav'
            samples, samplerate = sf.read(audio_path)

            duration = len(samples) / samplerate

            fr_sec = self.samples[index][3].split(':')
            hour1 = float(fr_sec[0])
            minu1 = float(fr_sec[1])
            sec1 = float(fr_sec[2])
            fr_sec = (hour1 * 60 + minu1) * 60 + sec1

            stop_sec = self.samples[index][4].split(':')
            hour1 = float(stop_sec[0])
            minu1 = float(stop_sec[1])
            sec1 = float(stop_sec[2])
            stop_sec = (hour1 * 60 + minu1) * 60 + sec1

            start1 = fr_sec / duration * len(samples)
            end1 = stop_sec / duration * len(samples)
            start1 = int(np.round(start1))
            end1 = int(np.round(end1))
            samples = samples[start1:end1]

            resamples = samples[:160000]
            while len(resamples) < 160000:
                resamples = np.tile(resamples, 10)[:160000]

            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.
            frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
            spectrogram = np.log(spectrogram + 1e-7)

            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
            if self.split == 'train':
                noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
                spectrogram = spectrogram + noise
                # if np.random.uniform(0,1,(1,))[0]>0.5:
                #     spectrogram = spectrogram[:,::-1]
                start1 = np.random.choice(256 - self.interval, (1,))[0]
                spectrogram[start1:(start1 + self.interval), :] = 0

        if self.use_audio is True:
            return data, spectrogram.astype(np.float32), label1, 0, 0
        else:
            return data, label1, 0, 0, 0, 0

    def __len__(self):
        return len(self.samples)

class EPICDOMAINAudio(torch.utils.data.Dataset):
    def __init__(self, split='train', domain='D1', modality='rgb', use_audio=True):
        self.base_path = '/home/xxx/data/EPIC_KITCHENS_UDA/'
        train_file = pd.read_pickle('/home/xxx/data/EPIC_KITCHENS_UDA/'+domain+"_"+split+".pkl")
        self.domain = domain
        self.split = split
        self.modality = modality
        self.use_audio = use_audio
        self.interval = 9

        data1 = []
        class_dict = {}
        for _, line in train_file.iterrows():
            image = [domain + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                     line['stop_timestamp']]
            labels = line['verb_class']
            data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
            if line['verb'] not in list(class_dict.keys()):
                class_dict[line['verb']] = line['verb_class']

        self.class_dict = class_dict
        self.samples = data1

    def __getitem__(self, index):
        label1 = self.samples[index][-1]

        audio_path = self.base_path + 'AudioVGGSound/' + self.split + '/' + self.samples[index][0] + '.wav'
        samples, samplerate = sf.read(audio_path)

        duration = len(samples) / samplerate

        fr_sec = self.samples[index][3].split(':')
        hour1 = float(fr_sec[0])
        minu1 = float(fr_sec[1])
        sec1 = float(fr_sec[2])
        fr_sec = (hour1 * 60 + minu1) * 60 + sec1

        stop_sec = self.samples[index][4].split(':')
        hour1 = float(stop_sec[0])
        minu1 = float(stop_sec[1])
        sec1 = float(stop_sec[2])
        stop_sec = (hour1 * 60 + minu1) * 60 + sec1

        start1 = fr_sec / duration * len(samples)
        end1 = stop_sec / duration * len(samples)
        start1 = int(np.round(start1))
        end1 = int(np.round(end1))
        samples = samples[start1:end1]

        resamples = samples[:160000]
        while len(resamples) < 160000:
            resamples = np.tile(resamples, 10)[:160000]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        if self.split == 'train':
            noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
            spectrogram = spectrogram + noise
            # if np.random.uniform(0,1,(1,))[0]>0.5:
            #     spectrogram = spectrogram[:,::-1]
            start1 = np.random.choice(256 - self.interval, (1,))[0]
            spectrogram[start1:(start1 + self.interval), :] = 0

        return spectrogram.astype(np.float32), label1


    def __len__(self):
        return len(self.samples)

