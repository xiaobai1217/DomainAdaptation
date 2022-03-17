import torch
import argparse
import tqdm
import os
import numpy as np
import math
import csv
from VGGSound.model import AVENet
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

base_path = '/home/yzhang8/data/CharadesEgo/'

def get_spec_piece(samples, start_time, end_time, duration):
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
    return spectrogram

video_list = []
with open(base_path + "CharadesEgo_v1_%s_only%s.csv" % ('train', '1st')) as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        if i == 0:
            continue
        if os.path.exists(base_path + "Labels/" + row[0]) and len(os.listdir(base_path + "Labels/" + row[0])) > 0:
            video_list.append(row[0])
f.close()

len1 = len(video_list)
len1 = int(len1 / 2)
video_list = video_list[:len1]


acc = 0
save_path = args.source_domain + '2' + args.target_domain + '_CharadesEgo_vggsound_features/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, sample1 in enumerate(video_list):
    print(i, len(video_list))
    #video_path = base_path + 'CharadesEgo_v1_rgb/' + video_list[i] + "/" + video_list[i] + '-'
    label_path = base_path + "Labels/" + video_list[i] + "/"
    label_list = os.listdir(label_path)
    for id1 in range(len(label_list)):
        frame_list = []
        class_list = []
        all_labels = []
        with open(label_path + label_list[id1]) as f:
            f_csv = csv.reader(f)
            for jjj, row in enumerate(f_csv):
                frame_list.append(row[0])
                tmp = row[1].split(' ')
                tmp2 = []
                for tmpp in tmp:
                    all_labels.append(int(tmpp[1:]))
                    tmp2.append(int(tmpp[1:]))
                class_list.append(tmp2)
        f.close()

        all_labels = np.unique(np.array(all_labels))
        start_frame = int(frame_list[0])
        end_frame = int(frame_list[-1])

        for label1 in list(all_labels):
            id11 = 0
            start_list1 = []
            end_list1 = []
            flag = 0
            while id11 < (end_frame-start_frame):
                if label1 in class_list[id11] and flag==0:
                    start_list1.append(id11+start_frame)
                    flag = 1
                if flag ==1 and label1 not in class_list[id11]:
                    end_list1.append(id11-1+start_frame)
                    flag=0
                id11 += 1
            if flag == 1:
                end_list1.append(id11-1+start_frame)


            for start_id, start1 in enumerate(start_list1):
                end1 = end_list1[start_id]

                start_time = start1 / 24.0
                end_time = end1 / 24.0
                audio_path = base_path + 'audio/' + video_list[i] + '.wav'
                samples, samplerate = sf.read(audio_path)

                duration = len(samples) / samplerate

                start1 = start_time / duration * len(samples)
                end1 = end_time / duration * len(samples)
                start1 = int(np.round(start1))
                end1 = int(np.round(end1))
                samples = samples[start1:end1]

                resamples = samples[:160000]
                if len(resamples) == 0:
                    continue
                while len(resamples) < 160000:
                    resamples = np.tile(resamples, 10)[:160000]

                resamples[resamples > 1.] = 1.
                resamples[resamples < -1.] = -1.
                frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
                spectrogram = np.log(spectrogram + 1e-7)

                mean = np.mean(spectrogram)
                std = np.std(spectrogram)
                spectrogram = np.divide(spectrogram - mean, std + 1e-9)

                spectrogram = torch.Tensor(spectrogram).type(torch.FloatTensor).cuda().unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    _, _, audio_feat = audio_model(spectrogram)

                audio_feat = audio_feat.detach().cpu().numpy()
                save_path1 = save_path + str(label1) + '/'
                if not os.path.exists(save_path1):
                    os.mkdir(save_path1)

                np.save(save_path1 + video_list[i] + '_%06d_%06d.npy'%(start_list1[start_id], end_list1[start_id]), audio_feat)





