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
import os

class CharadesEgoAudio(torch.utils.data.Dataset):
    def __init__(self, split='train', domain='1st', cfg=None):
        self.base_path = "/home/yzhang8/data/CharadesEgo/"
        self.split = split
        self.domain = domain
        self.video_list = []
        with open(self.base_path+"CharadesEgo_v1_%s_only%s.csv"%(split, domain)) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i ==0:
                    continue
                if os.path.exists(self.base_path+"Labels/"+row[0]) and len(os.listdir(self.base_path+"Labels/"+row[0])) > 0:
                    self.video_list.append(row[0])

        if split == 'train':
            len1 = int(0.75 * len(self.video_list))
            self.video_list = self.video_list[:len1]

        self.interval = 9

    def __getitem__(self, index):
        audio_path = self.base_path + 'audio/' +"/" + self.video_list[index] + '.wav'
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

        if end_frame == start_frame:
            index = np.random.choice(self.__len__(), (1,))[0]
            audio_path = self.base_path + 'audio/' + "/" + self.video_list[index] + '.wav'
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

        #visual length 64*8
        if end_frame - start_frame - 64*8 > 0:
            start1 = np.random.choice(end_frame -start_frame - 64*8, (1,))[0]
            start_frame1 = start_frame + start1
            end_frame1 = start_frame1 + 64*8
        else:
            start_frame1 = start_frame
            end_frame1 = end_frame

        start_time = start_frame1 / 24.0
        end_time = end_frame1 / 24.0

        samples, samplerate = sf.read(audio_path)

        duration = len(samples) / samplerate

        start1 = start_time / duration * len(samples)
        end1 = end_time / duration * len(samples)
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

        label1 = np.zeros((157))
        for i in range(int(start_frame1), int(end_frame1)):
            tmp = class_list[i - start_frame].split(' ')
            for ttt in tmp:
                tt = ttt[1:]
                label1[int(tt)] += 1
        label1 = (label1 > 0).astype(np.float32)

        return spectrogram.astype(np.float32), label1

    def __len__(self):
        return len(self.video_list)