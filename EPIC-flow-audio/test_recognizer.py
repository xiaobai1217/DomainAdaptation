from mmaction.apis import init_recognizer, inference_recognizer
import torch
import torch.nn.functional as F
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
from vit_cls import ViTCls
from vit import ViT
from train_recognizer import Recognizer
from scipy import signal

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--source_domain', type=str, help='input a str', default='D1')
parser.add_argument('--target_domain', type=str, help='input a str', default='D2')
parser.add_argument('--depth', type=int, help='input a str', default=2)
parser.add_argument('--dropout', type=float, help='input a str', default=0.15)
parser.add_argument('--emb_dropout', type=float, help='input a str', default=0.1)
args = parser.parse_args()

config_file = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow2.py'
checkpoint_file = '/home/yzhang8/data/mmaction2_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

# assign the desired device.
device = 'cuda:0'  # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
model.cls_head.fc_cls = nn.Linear(2048, 8).cuda()
cfg = model.cfg
model = torch.nn.DataParallel(model)
checkpoint = torch.load("checkpoints/best_%s2%s_1stStage.pt" % (args.source_domain, args.target_domain))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

attention_model = ViT(dim=256, depth=8, heads=8, mlp_dim=512, dropout=0.15, emb_dropout=0.1, dim_head=64)
attention_model = attention_model.cuda()
attention_model = torch.nn.DataParallel(attention_model)
attention_model.load_state_dict(checkpoint['audio_state_dict'])
attention_model.eval()

audio_args = get_arguments()
audio_model = AVENet(audio_args)
checkpoint = torch.load("vggsound_avgpool.pth.tar")
audio_model.load_state_dict(checkpoint['model_state_dict'])
audio_model = audio_model.cuda()
audio_model = torch.nn.DataParallel(audio_model)
audio_model.eval()

audio_cls_model = AudioAttGenModule()
audio_cls_model.fc = nn.Linear(512, 8)
audio_cls_model = audio_cls_model.cuda()
checkpoint = torch.load("checkpoints/best_%s2%s_audio.pt")
audio_cls_model.load_state_dict(checkpoint['audio_state_dict'])
audio_cls_model = torch.nn.DataParallel(audio_cls_model)
audio_cls_model.eval()

recognizer = Recognizer()
recognizer = recognizer.cuda()
recognizer = torch.nn.DataParallel(recognizer)
checkpoint = torch.load("checkpoints/best_%s2%s_slow_flow_transformer_cls.pt" % (args.source_domain, args.target_domain,))
recognizer.load_state_dict(checkpoint['adapter_state_dict'])
recognizer.eval()

base_path = '/home/yzhang8/data/EPIC_KITCHENS_UDA/'
test_file = pd.read_pickle('/home/yzhang8/data/EPIC_KITCHENS_UDA/' + args.target_domain + "_test.pkl")
test_pipeline = cfg.data.test.pipeline
test_pipeline = Compose(test_pipeline)

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
save_path = 'preds/'+args.source_domain + '2' + args.target_domain + '_transformer4cls/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, sample1 in enumerate(data1):
    label1 = sample1[-1]
    video_path = base_path + 'frames_rgb_flow/flow/test/' + sample1[0]
    filename_tmpl = cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
    modality = cfg.data.train.get('modality', 'Flow')
    start_index = cfg.data.val.get('start_index', int(np.ceil(sample1[1] / 2)))
    data = dict(
        frame_dir=video_path,
        total_frames=int((sample1[2] - sample1[1]) / 2),
        # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
        label=-1,
        start_index=start_index,
        filename_tmpl=filename_tmpl,
        modality=modality)
    data = test_pipeline(data)
    clip = data['imgs'].cuda()

    audio_path = base_path + 'AudioVGGSound/test/' +sample1[0] + '.wav'
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


    with torch.no_grad():
        v_feat = model.module.backbone.get_feature(clip)  #
        _, audio_feat4att, _ = audio_model(spectrogram)  #
        _,audio_feat = audio_cls_model(audio_feat4att)

        channel_att = attention_model(audio_feat4att.detach())
        channel_att = channel_att.unsqueeze(1).repeat(1, 5, 1)
        channel_att = channel_att.view(5, 1024)
        adapted_v_feat = torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * v_feat.detach()
        v_feat = model.module.backbone.get_predict(adapted_v_feat.detach())  # batch_size*5, 2048, 8, 7, 7

        v_feat = F.adaptive_max_pool3d(v_feat, (1, 1, 1))
        v_feat = v_feat.flatten(1)
        v_feat = v_feat.view(1, 5, 2048)
        predict1, audio_att1, audio_att2 = recognizer(visual_feat=v_feat.detach(), audio_feat=audio_feat.detach(), target=True)

        predict1 = torch.softmax(predict1, dim=1)

    predict1 = torch.mean(predict1, dim=0).detach().cpu().numpy()

    if np.argmax(predict1) == label1:
        acc += 1
    print(i, acc / (i+1), len(data1))
    video_id = sample1[0].split("/")[-1]
    np.save(save_path+video_id + "_%010d.npy"%start_index, predict1)

print(acc/len(data1))

