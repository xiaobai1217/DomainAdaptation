from mmaction.datasets.pipelines import Compose
import cv2
import torch.utils.data

import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np
import csv
import random

class EPICDOMAINAdapter(torch.utils.data.Dataset):
    def __init__(self, split='train', source_domain='D1', target_domain='D2', modality='rgb', cfg=None, use_audio=True):
        self.base_path = '/home/xxx/data/EPIC_KITCHENS_UDA/'
        source_train_file = pd.read_pickle('/home/xxx/data/EPIC_KITCHENS_UDA/'+source_domain+"_"+split+".pkl")
        target_train_file = pd.read_pickle('/home/xxx/data/EPIC_KITCHENS_UDA/'+target_domain+"_"+split+".pkl")

        self.source_domain = source_domain
        self.target_domain = target_domain
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

        source_data1 = []
        class_dict = {}
        for _, line in source_train_file.iterrows():
            image = [source_domain + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                     line['stop_timestamp']]
            labels = line['verb_class']
            source_data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
            if line['verb'] not in list(class_dict.keys()):
                class_dict[line['verb']] = line['verb_class']

        target_data1 = []
        for _, line in target_train_file.iterrows():
            image = [target_domain + '/' + line['video_id'], line['start_frame'], line['stop_frame'],
                     line['start_timestamp'],
                     line['stop_timestamp']]
            # labels = line['verb_class']
            tmp = int(line['start_frame'])
            labels = np.load("pseudo_labels/%s2%s/" % (source_domain, target_domain) + line[
                'video_id'] + "_%010d.npy" % tmp)
            labels = np.argmax(labels)
            target_data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))

        self.source_samples = source_data1
        self.target_samples = target_data1
        idx_list = list(range(len(self.target_samples)))
        random.shuffle(idx_list)
        self.target_idx_list = idx_list
        self.class_dict = class_dict
        self.cfg = cfg

    def extract_spectrogram(self, sample1):
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
            # if np.random.uniform(0,1,(1,))[0]>0.5:
            #     spectrogram = spectrogram[:,::-1]
            start1 = np.random.choice(256 - self.interval, (1,))[0]
            spectrogram[start1:(start1 + self.interval), :] = 0
        return spectrogram

    def __getitem__(self, index):
        video_path = self.base_path +'frames_rgb_flow/rgb/'+self.split + '/'+self.source_samples[index][0]
        index2 = self.target_idx_list.pop(0)
        self.target_idx_list.append(index2)
        target_video_path = self.base_path +'frames_rgb_flow/rgb/'+self.split + '/'+self.target_samples[index2][0]

        filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        modality = self.cfg.data.train.get('modality', 'RGB')
        start_index = self.cfg.data.train.get('start_index', int(self.source_samples[index][1]))
        data = dict(
            frame_dir=video_path,
            total_frames=int(self.source_samples[index][2] - self.source_samples[index][1]),
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data = self.train_pipeline(data)

        start_index = self.cfg.data.train.get('start_index', int(self.target_samples[index2][1]))
        target_data = dict(
            frame_dir=target_video_path,
            total_frames=int(self.target_samples[index2][2] - self.target_samples[index2][1]),
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        target_data = self.train_pipeline(target_data)

        data['imgs'] = torch.cat((data['imgs'], target_data['imgs']), dim=0)
        label1 = self.source_samples[index][-1]
        target_label1 = self.target_samples[index2][-1]

        spectrogram = self.extract_spectrogram(self.source_samples[index])
        target_spectrogram = self.extract_spectrogram(self.target_samples[index2])

        spectrogram = np.stack((spectrogram, target_spectrogram))
        label1 = np.array([label1, target_label1])

        return data, spectrogram, label1

    def __len__(self):
        return len(self.source_samples)

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
        #print(data['imgs'].size())
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
            return data, spectrogram.astype(np.float32), label1
        else:
            return data, label1

    def __len__(self):
        return len(self.samples)
