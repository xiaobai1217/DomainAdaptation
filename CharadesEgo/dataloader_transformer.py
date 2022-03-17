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

def get_spectrogram_piece(samples, start_time, end_time, duration, samplerate, training=False):
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

    interval = 9
    if training is True:
        noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
        spectrogram = spectrogram + noise
        start1 = np.random.choice(256 - interval, (1,))[0]
        spectrogram[start1:(start1 + interval), :] = 0

    return spectrogram

class CharadesEgoProjectionTraining(torch.utils.data.Dataset):
    def __init__(self, split='train', source_domain='3rd', target_domain='1st', modality='rgb', cfg=None,):
        self.base_path = '/home/yzhang8/data/CharadesEgo/'
        self.video_list = []

        self.per_cls_freq = np.load("per_class_freq.npy")

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

        video_list = []
        target_class_list = []
        with open(self.base_path + "CharadesEgo_v1_%s_only%s.csv" % (split, target_domain)) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i == 0:
                    continue
                class_list1 = row[-4]
                class_list1 = class_list1.split(";")
                if len(class_list1[0]) > 0:
                    tmp = []
                    for item in class_list1:
                        tmp.append(int((item.split(" "))[0][1:]))

                    labels = np.zeros((157,))
                    for ttt in tmp:
                        labels[ttt] = 1

                    target_class_list.append(labels)
                    video_list.append(row[0])
        f.close()
        self.target_video_list = video_list
        self.target_class_list = np.stack(target_class_list)
        self.num_target_videos = int(len(video_list)/2)
        self.target_video_list = self.target_video_list[:self.num_target_videos]
        self.target_class_list = self.target_class_list[:self.num_target_videos]
        # build the data pipeline
        if split == 'train':
            train_pipeline = cfg.data.train.pipeline
            self.train_pipeline = Compose(train_pipeline)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.val_pipeline = Compose(val_pipeline)

        self.cfg = cfg

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

        #-------------------------------------------------------------------------------------------------------------
        audio_path = self.base_path + 'audio/' + self.video_list[index] + '.wav'
        start_time = frame_inds[0] / 24.0
        end_time = frame_inds[-1] / 24.0
        samples, samplerate = sf.read(audio_path)
        duration = len(samples) / samplerate
        spectrogram = get_spectrogram_piece(samples, start_time, end_time, duration, samplerate)
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
        target_spectrogram = get_spectrogram_piece(samples, start_time, end_time, duration, samplerate)

        #----------------------------------------------get judge sample---------------------------------------------------------------
        if np.random.uniform(0,1,(1,))[0] > 0.5:
            common_labels = np.sum(np.expand_dims(label1, axis=0) * self.target_class_list, axis=1)
            #print(self.num_target_videos, common_labels.shape)
            potential_idxs = np.arange(self.num_target_videos)[common_labels > 0]
            idx1 = np.random.choice(potential_idxs, (1,))[0]

            video_path = self.base_path + 'CharadesEgo_v1_rgb/' + self.target_video_list[idx1] + "/" + \
                         self.target_video_list[idx1] + '-'
            label_path = self.base_path + "Labels/" + self.target_video_list[idx1] + "/"
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
            another_target_data = dict(
                frame_dir=video_path,
                total_frames=end_frame - start_frame,
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            another_target_data, another_target_frame_inds = self.train_pipeline(another_target_data)

            another_target_label1 = np.zeros((157))
            for i in another_target_frame_inds:
                tmp = class_list[i - start_frame].split(' ')
                for ttt in tmp:
                    tt = ttt[1:]
                    another_target_label1[int(tt)] += 1
            another_target_label1 = (another_target_label1 > 0).astype(np.float32)

            audio_path = self.base_path + 'audio/' + self.target_video_list[idx1] + '.wav'
            start_time = another_target_frame_inds[0] / 24.0
            end_time = another_target_frame_inds[-1] / 24.0
            samples, samplerate = sf.read(audio_path)
            duration = len(samples) / samplerate
            another_target_spectrogram = get_spectrogram_piece(samples, start_time, end_time, duration, samplerate)

            judge_label = np.sum(another_target_label1 * label1) > 0
            return data, label1, spectrogram, target_data, target_label1, target_spectrogram, another_target_data, judge_label, another_target_spectrogram
        else:
            judge_label = np.sum(target_label1 * label1) > 0
            return data, label1, spectrogram, target_data, target_label1, target_spectrogram,target_data, judge_label, target_spectrogram

    def __len__(self):
        return len(self.video_list)


class CharadesEgoProjectionValidating(torch.utils.data.Dataset):
    def __init__(self, split='test', domain='3rd',  modality='rgb', cfg=None,):
        self.base_path = '/home/yzhang8/data/CharadesEgo/'
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

        spectrogram = get_spectrogram_piece(samples,start_time,end_time,duration,samplerate,training=False)

        # -------------------------------------------------------------------------------------------------------------

        return data, label1, spectrogram,0,0,0,0,0,0

    def __len__(self):
        return len(self.video_list)


class CharadesEgoReweightingTesting(torch.utils.data.Dataset):
    def __init__(self, split='train', domain='1st',  modality='rgb', cfg=None,):
        self.base_path = '/home/yzhang8/data/CharadesEgo/'
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


