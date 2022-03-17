from mmaction.apis import init_recognizer, inference_recognizer
import torch
from data_loader_epic import EPICDOMAIN
import argparse
import tqdm
import os
import numpy as np
import math
import csv
import collections
import torch.nn as nn
from torch.optim import lr_scheduler
import pandas as pd
from mmaction.datasets.pipelines import Compose
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
import soundfile as sf
from scipy import signal

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--source_domain', type=str, help='input a str', default='D1')
parser.add_argument('--target_domain', type=str, help='input a str', default='D2')
args = parser.parse_args()


# assign the desired device.
device = 'cuda:0'  # or 'cpu'
device = torch.device(device)

audio_args = get_arguments()
audio_model = AVENet(audio_args)
checkpoint = torch.load("vggsound_avgpool.pth.tar")
audio_model.load_state_dict(checkpoint['model_state_dict'])
audio_model = audio_model.cuda()
audio_model.eval()

audio_cls_model = AudioAttGenModule()
audio_cls_model.fc = nn.Linear(512, 8)
checkpoint = torch.load("checkpoints/best_%s2%s_audio.pt"%(args.source_domain, args.target_domain))
audio_cls_model.load_state_dict(checkpoint['audio_state_dict'])
audio_cls_model = audio_cls_model.cuda()
audio_cls_model.eval()

base_path = '/home/yzhang8/data/EPIC_KITCHENS_UDA/'
test_file = pd.read_pickle('/home/yzhang8/data/EPIC_KITCHENS_UDA/' + args.target_domain + "_train.pkl")


data1 = []
class_dict = {}
for _, line in test_file.iterrows():
    image = [args.target_domain + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
             line['stop_timestamp']]
    labels = line['verb_class']
    # one_hot = np.zeros(8)
    # one_hot[labels] = 1.0
    data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
    if line['verb'] not in list(class_dict.keys()):
        class_dict[line['verb']] = line['verb_class']

acc = 0
save_path = args.source_domain + '2' + args.target_domain + '_audio_train/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, sample1 in enumerate(data1):
    label1 = sample1[-1]
    start_index = int(np.ceil(sample1[1] / 2))

    audio_path = base_path + 'AudioVGGSound/train/' +sample1[0] + '.wav'
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
    spectrogram = torch.Tensor(spectrogram).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()

    predict_list = []
    with torch.no_grad():
        _, audio_feat, _ = audio_model(spectrogram)
        audio_predict = audio_cls_model(audio_feat.detach())

        predict1 = torch.softmax(audio_predict, dim=1)

    predict1 = predict1[0].detach().cpu().numpy()

    if np.argmax(predict1) == label1:
        acc += 1
    print(i, acc / (i+1), len(data1))
    video_id = sample1[0].split("/")[-1]
    np.save(save_path+video_id + "_%010d.npy"%start_index, predict1)

print(acc/len(data1))

