from mmaction.apis import init_recognizer, inference_recognizer
import torch
#from data_loader_epic import EPICDOMAIN
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
from util1 import AveragePrecisionMeter
from vit import ViT
from vit_cls import ViTCls
from scipy import signal
import soundfile as sf
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
import torch.nn.functional as F
from dataloader_recognizer import get_spectrogram_piece
from train_recognizer import Recognizer

class CharadesEgoTesting(torch.utils.data.Dataset):
    def __init__(self, split='test', domain='3rd',  modality='rgb', cfg=None,):
        self.base_path = '/local-ssd/yzhang9/data/CharadesEgo/'

        self.video_list = []
        self.class_list = []
        with open(self.base_path + "CharadesEgo_v1_%s_only%s.csv" % ('test', '1st')) as f:
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
                    self.class_list.append(tmp)
                    self.video_list.append(row[0])
        f.close()

        self.domain = domain
        self.split = split
        self.modality = modality

        # build the data pipeline
        test_pipeline = cfg.data.test.pipeline
        self.test_pipeline = Compose(test_pipeline)

        self.cfg = cfg
        self.interval = 9

    def __getitem__(self, index):
        video_path = self.base_path + 'CharadesEgo_v1_rgb/' + self.video_list[index] + "/" + self.video_list[
            index] + '-'
        total_frames = len(os.listdir(self.base_path + 'CharadesEgo_v1_rgb/' + self.video_list[index]))

        label1 = self.class_list[index]
        labels = np.zeros((157,))
        for tmp in label1:
            labels[tmp] = 1

        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', 1)
        data = dict(
            frame_dir=video_path,
            total_frames=total_frames,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data, frame_inds = self.test_pipeline(data)
        data = data['imgs']
        num_clips = data.size()[0]
        frame_inds = frame_inds.reshape((num_clips, -1))

        # -------------------------------------------------------------------------------------------------------------
        audio_path = self.base_path + 'audio/' + self.video_list[index] + '.wav'
        samples, samplerate = sf.read(audio_path)
        duration = len(samples) / samplerate
        spec_list = []
        for ii in range(num_clips):
            start_time = frame_inds[ii,0] / 24.0
            end_time = frame_inds[ii,-1] / 24.0
            spectrogram = get_spectrogram_piece(samples, start_time, end_time, duration, samplerate, training=False)
            spec_list.append(spectrogram)
        spectrograms = np.stack(spec_list)
        # -------------------------------------------------------------------------------------------------------------

        return data, labels, spectrograms.astype(np.float32), self.video_list[index]

    def __len__(self):
        return len(self.video_list)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--source_domain', type=str, help='input a str', default='3rd')
parser.add_argument('--target_domain', type=str, help='input a str', default='1st')
parser.add_argument('--depth', type=int, help='input a str', default=2)
parser.add_argument('--dropout', type=float, help='input a str', default=0.15)
parser.add_argument('--emb_dropout', type=float, help='input a str', default=0.1)

args = parser.parse_args()

config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
#checkpoint_file = '/var/scratch/yzhang9/data/mmaction2_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'
checkpoint_file = '/var/scratch/yzhang9/data/mmaction2_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

# assign the desired device.
device = 'cuda:0'  # or 'cpu'
device = torch.device(device)

# build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
model.cls_head.fc_cls = nn.Linear(2304, 157).cuda()
cfg = model.cfg
model = torch.nn.DataParallel(model)
checkpoint = torch.load("checkpoints/best_%s2%s_CharadesEgo_encoder_finetuned.pt"%(args.source_domain, args.target_domain,))
model.load_state_dict(checkpoint['state_dict'])
model.eval()


attention_model = ViT(dim=256, depth=8, heads=8, mlp_dim=512, dropout=0.15, emb_dropout=0.1, dim_head=64)
attention_model = attention_model.cuda()
attention_model = torch.nn.DataParallel(attention_model)
attention_model.load_state_dict(checkpoint['att_state_dict'])
attention_model.eval()

audio_args = get_arguments()
audio_model = AVENet(audio_args)
checkpoint = torch.load("vggsound_avgpool.pth.tar")
audio_model.load_state_dict(checkpoint['model_state_dict'])
audio_model = audio_model.cuda()
audio_model = torch.nn.DataParallel(audio_model)
audio_model.eval()

audio_cls_model = AudioAttGenModule()
audio_cls_model.fc = nn.Linear(512, 157)
audio_cls_model = audio_cls_model.cuda()
checkpoint = torch.load("checkpoints/best_%s2%s_audio.pt" % (args.source_domain, args.target_domain))
audio_cls_model.load_state_dict(checkpoint['audio_state_dict'])
audio_cls_model = torch.nn.DataParallel(audio_cls_model)
audio_cls_model.eval()

recognizer = Recognizer()
recognizer = recognizer.cuda()
recognizer = torch.nn.DataParallel(recognizer)
checkpoint = torch.load("checkpoints/best_3rd21st_CharadesEgo_recognizer_fintuned.pt")
recognizer.load_state_dict(checkpoint['state_dict'])
recognizer.eval()

test_dataset = CharadesEgoTesting(split='test', domain=args.target_domain, modality='rgb', cfg=cfg, )
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=True,
                                                  pin_memory=(device.type == "cuda"), drop_last=True)

#base_path = '/var/scratch/yzhang9/data/CharadesEgo/'
base_path = '/local-ssd/yzhang9/data/CharadesEgo/'
test_pipeline = cfg.data.test.pipeline
test_pipeline = Compose(test_pipeline)


ap_meter = AveragePrecisionMeter(False)

ap_meter.reset()

acc = 0
save_path = args.source_domain + '2' + args.target_domain + '_CharadesEgo_transformer/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
for (i, (clip, labels, spectrogram, sample1)) in enumerate(test_dataloader):
    print(i, len(test_dataloader))
    clip = clip.cuda()
    spectrogram = spectrogram.squeeze(0).unsqueeze(1).cuda()

    predict_list = []
    with torch.no_grad():
        _,audio_feat,_ = audio_model(spectrogram)
        _,audiofeat4trans = audio_cls_model(audio_feat)

        channel_att, channel_att2 = attention_model(audio_feat.detach())

        for j in range(0,clip.size(0)):
            x_slow, x_fast = model.module.backbone.get_feature(clip[j])
            feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

            adapted_v_feat = [torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[0],
                              torch.sigmoid(channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[1]]

            slow_feat, fast_feat = model.module.backbone.get_predict(adapted_v_feat)

            slow_feat = F.adaptive_max_pool3d(slow_feat.detach(), (16, 1, 1)).squeeze(3).squeeze(3)
            fast_feat = F.adaptive_max_pool3d(fast_feat.detach(), (64, 1, 1)).squeeze(3).squeeze(3)
            slow_feat = slow_feat.transpose(1, 2).contiguous().detach()
            fast_feat = fast_feat.transpose(1, 2).contiguous().detach()
            predict1,_,_ = recognizer(slow_feat, fast_feat, audiofeat4trans.detach(), target=True)

            predict1 = torch.sigmoid(predict1)
            predict_list.append(predict1.detach())
    predict1 = torch.cat(predict_list, dim=0)
    predict1,_ = torch.max(predict1, dim=0)
    predict1 = predict1.detach().unsqueeze(0).cpu().numpy()
    ap_meter.add(predict1, labels.numpy())


    np.save(save_path+sample1[0], predict1[0])

map = 100 * ap_meter.value().mean()
print(map)



