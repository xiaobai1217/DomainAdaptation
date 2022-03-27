import torch
import argparse
import tqdm
import os
import numpy as np
import math
import csv
import collections
import torch.nn as nn
from util1 import AveragePrecisionMeter
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from scipy import signal
import soundfile as sf


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--source_domain', type=str, help='input a str', default='3rd')
parser.add_argument('--target_domain', type=str, help='input a str', default='1st')
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

audio_att_model = AudioAttGenModule()
audio_att_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
audio_att_model.fc = nn.Linear(512, 157)
audio_att_model = audio_att_model.cuda()
checkpoint = torch.load("checkpoints/best_%s2%s_audio.pt"%(args.source_domain, args.target_domain))
audio_att_model.load_state_dict(checkpoint['audio_state_dict'])
audio_att_model.eval()

base_path = '/local-ssd/yzhang9/data/CharadesEgo/'

def get_spec_piece(samples, start_time, end_time, duration):
    start1 = start_time / duration * len(samples)
    end1 = end_time / duration * len(samples)
    start1 = int(np.round(start1))
    end1 = int(np.round(end1))
    samples = samples[start1:end1]

    resamples = samples[:160000]
    if len(resamples) == 0:
        resamples = np.zeros((160000,))
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

video_list = []
class_list = []
with open(base_path + "CharadesEgo_v1_%s_only%s.csv" % ('train', '1st')) as f:
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

acc = 0
save_path = args.source_domain + '2' + args.target_domain + '_CharadesEgo_audio_on_train/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, sample1 in enumerate(video_list):
    print(i, len(video_list))
    #if i < 2426:
    #    continue 
    audio_path = base_path + 'audio/' + sample1 + '.wav'

    label1 = class_list[i]
    frame_num = len(os.listdir(base_path + 'CharadesEgo_v1_rgb/' + sample1))

    if frame_num - 64 * 8 > 0:
        start_frame1 = np.arange(frame_num-64*8-1)
        end_frame1 = start_frame1 + 64 * 8
    else:
        start_frame1 = np.array([0,])
        end_frame1 = np.array([frame_num-1,])

    start_time = start_frame1 / 24.0
    end_time = end_frame1 / 24.0

    samples, samplerate = sf.read(audio_path)
    duration = len(samples) / samplerate


    spec_list = []
    for ii in range(len(start_time)):
        spec1 = get_spec_piece(samples, start_time[ii], end_time[ii], duration)
        spec_list.append(spec1)
    if len(spec_list) > 0:
        spec_list = np.stack(spec_list)
    else:
        continue
    spec_list = torch.Tensor(spec_list).type(torch.FloatTensor).cuda().unsqueeze(1)

    predict_list = []
    for ii in range(0,spec_list.size(0), 32):
        with torch.no_grad():
            #print(ii)
            #print(spec_list[ii*32:(ii+1)*32,:,:,:].size())
            _, audio_feat,_ = audio_model(spec_list[ii:ii+32,:,:,:])
            audio_predict = audio_att_model(audio_feat.detach())
            predict1 = torch.sigmoid(audio_predict)
            predict_list.append(predict1)
    predict1 = torch.cat(predict_list, dim=0)
    predict1 = torch.mean(predict1, dim=0)

    labels = np.zeros((1,157))
    for tmp in label1:
        labels[0,tmp] = 1

    predict = predict1.detach().unsqueeze(0).cpu().numpy()
    
    ap_meter.add(predict, labels)

    np.save(save_path+sample1, predict[0])

map = 100 * ap_meter.value().mean()
print(map)



