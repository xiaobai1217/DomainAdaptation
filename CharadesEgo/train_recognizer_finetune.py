from mmaction.apis import init_recognizer, inference_recognizer
import torch
from dataloader_recognizer import CharadesEgoTraining, CharadesEgoValidating
import argparse
import tqdm
import os
import numpy as np
import math
import csv
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from util1 import AveragePrecisionMeter
import pdb
from vit import ViT
from vit_cls import ViTCls
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from train_recognizer import Recognizer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_one_step(model, audio_model,audio_cls_model, att_model, recognizer, clip, spectrogram, labels, target_clip, target_labels, target_spectrogram):
    target_clip = target_clip['imgs'].squeeze(1).cuda()
    target_labels = target_labels.cuda()
    b,c,f,h,w = clip.size()

    target_spectrogram = target_spectrogram.type(torch.FloatTensor).cuda().unsqueeze(1)

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        _, audio_feat, _ = audio_model(spectrogram)  # 16,256,17,63
        _,audiofeat4trans = audio_cls_model(audio_feat)
        feat = (x_slow.detach(), x_fast.detach()) #slow 16,1280,16,14,14, fast 16,128,64,14,14

        target_x_slow, target_x_fast = model.module.backbone.get_feature(target_clip)
        _, target_audio_feat, _ = audio_model(target_spectrogram)  # 16,256,17,63
        _,target_audiofeat4trans = audio_cls_model(target_audio_feat)

        target_feat = (target_x_slow.detach(), target_x_fast.detach()) #slow 16,1280,16,14,14, fast 16,128,64,14,14

        channel_att, channel_att2 = att_model(audio_feat.detach())
        adapted_v_feat = [torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[0], torch.sigmoid(channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[1]]
        slow_feat, fast_feat = model.module.backbone.get_predict(adapted_v_feat)

        target_channel_att, target_channel_att2 = att_model(target_audio_feat.detach())
        adapted_v_feat = [torch.sigmoid(target_channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * target_feat[0],
                          torch.sigmoid(target_channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * target_feat[1]]
        target_slow_feat, target_fast_feat = model.module.backbone.get_predict(adapted_v_feat)

        slow_feat = F.adaptive_max_pool3d(slow_feat.detach(), (16,1,1)).squeeze(3).squeeze(3)
        fast_feat = F.adaptive_max_pool3d(fast_feat.detach(), (64,1,1)).squeeze(3).squeeze(3)
        slow_feat1 = slow_feat.transpose(1,2).contiguous().detach().clone()
        fast_feat1 = fast_feat.transpose(1,2).contiguous().detach().clone()

        target_slow_feat = F.adaptive_max_pool3d(target_slow_feat.detach(), (16,1,1)).squeeze(3).squeeze(3)
        target_fast_feat = F.adaptive_max_pool3d(target_fast_feat.detach(), (64,1,1)).squeeze(3).squeeze(3)
        target_slow_feat1 = target_slow_feat.transpose(1,2).contiguous().detach().clone()
        target_fast_feat1 = target_fast_feat.transpose(1,2).contiguous().detach().clone()

    #2048,16,7,7; 256,64,7,7


    predict1, audio_att1, audio_att2= recognizer(slow_feat1, fast_feat1, audiofeat4trans.detach().clone(), target=False)


    target_predict1, _, _  = recognizer(target_slow_feat1, target_fast_feat1, target_audiofeat4trans.detach().clone(), target=True)

    labels2 = labels.unsqueeze(1).repeat(1,80,1).view(b*80,157).clone()
    #print(audio_att1.size(), audio_att2.size(), labels.size(), labels2.size())
    loss = (nn.BCELoss()(F.sigmoid(predict1), labels) + nn.BCELoss()(F.sigmoid(target_predict1), target_labels) )/2.0+ (criterion(F.sigmoid(audio_att1), labels) + criterion(F.sigmoid(audio_att2), labels2)*0.2)*0.1

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss, predict1


def validate_one_step(model, audio_model, audio_cls_model, att_model, adapter, clip, spectrogram, labels):
    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        _, audio_feat, _ = audio_model(spectrogram)  # 16,256,17,63
        _,audiofeat4trans = audio_cls_model(audio_feat)

        feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        channel_att, channel_att2 = att_model(audio_feat.detach())
        adapted_v_feat = [torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[0],
                          torch.sigmoid(channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[1]]

        slow_feat, fast_feat = model.module.backbone.get_predict(adapted_v_feat)
        slow_feat = F.adaptive_max_pool3d(slow_feat.detach(), (16, 1, 1)).squeeze(3).squeeze(3)
        fast_feat = F.adaptive_max_pool3d(fast_feat.detach(), (64, 1, 1)).squeeze(3).squeeze(3)
        slow_feat = slow_feat.transpose(1,2).contiguous().detach()
        fast_feat = fast_feat.transpose(1,2).contiguous().detach()
        predict1,_,_ = recognizer(slow_feat, fast_feat, audiofeat4trans.detach(), target=True)

    loss = criterion(F.sigmoid(predict1), labels)
    return loss, predict1
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, help='input a str', default='3rd')
    parser.add_argument('--target_domain', type=str, help='input a str', default='1st')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--emb_dropout', type=float, default=0.1)

    args = parser.parse_args()

    # config_file = 'configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py'
    # # download the checkpoint from model zoo and put it in `checkpoints/`
    # checkpoint_file = 'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = '/var/scratch/yzhang9/data/mmaction2_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

     # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, checkpoint_file, device=device,use_frames=True)
    model.cls_head.fc_cls = nn.Linear(2304, 157).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load("checkpoints/best_%s2%s_CharadesEgo_encoder_finetuned.pt"%(args.source_domain, args.target_domain))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    attention_model = ViT(dim=256, depth=8, heads=8,mlp_dim=512,dropout=0.15, emb_dropout=0.1, dim_head=64)
    attention_model = attention_model.cuda()
    attention_model = torch.nn.DataParallel(attention_model)
    attention_model.load_state_dict(checkpoint['att_state_dict'])
    attention_model.eval()

    ap_meter = AveragePrecisionMeter(False)

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
    checkpoint = torch.load("checkpoints/best_%s2%s_audio.pt"%(args.source_domain, args.target_domain))
    audio_cls_model.load_state_dict(checkpoint['audio_state_dict'])
    audio_cls_model = torch.nn.DataParallel(audio_cls_model)
    audio_cls_model.eval()


    recognizer = Recognizer()
    recognizer = recognizer.cuda()
    recognizer = torch.nn.DataParallel(recognizer)
    checkpoint = torch.load("checkpoints/best_3rd21st_CharadesEgo_recognizer.pt")
    recognizer.load_state_dict(checkpoint['state_dict'])
    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    log_path = base_path + "log%s2%s_recognizer_finetune.csv"%(args.source_domain, args.target_domain,)
    cmd = ['rm -rf ', log_path]
    os.system(' '.join(cmd))

    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    batch_size = 16
    lr = args.lr

    optim = torch.optim.Adam(recognizer.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optim, step_size=60, gamma=0.1)
    train_dataset = CharadesEgoTraining(split='train', source_domain=args.source_domain, target_domain=args.target_domain,modality='rgb', cfg=cfg, )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=3, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    validate_dataset = CharadesEgoValidating(split='test', domain=args.target_domain, modality='rgb', cfg=cfg,)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, num_workers=3, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    dataloaders = {'train': train_dataloader, 'val': validate_dataloader}
    BestLoss = float("inf")
    BestEpoch = 0
    BestMAP = 0
    with open(log_path, "a") as f:
        for epoch_i in range(30):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                max_record=0
                count = 0
                total_loss = 0
                print(split)
                ap_meter.reset()

                recognizer.train(split=='train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, labels, spectrogram, target_clip, target_labels, target_spectrogram)) in enumerate(dataloaders[split]):
                        clip = clip['imgs'].cuda().squeeze(1)
                        labels = labels.cuda()
                        spectrogram = spectrogram.type(torch.FloatTensor).cuda().unsqueeze(1)

                        if split=='train':
                            loss, predict1 = train_one_step(model, audio_model, audio_cls_model, attention_model, recognizer, clip, spectrogram, labels, target_clip, target_labels, target_spectrogram)
                        else:
                            loss, predict1 = validate_one_step(model,audio_model, audio_cls_model, attention_model, recognizer, clip, spectrogram, labels)

                        ap_meter.add(predict1.data, labels)

                        total_loss += loss.item() * batch_size

                        max_record += torch.max(predict1.detach()).item() * clip.size()[0]

                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f},{:.4f}".format(total_loss / float(count),
                                                                                                  loss.item(),max_record/float(count)))
                        pbar.update()
                    map = 100 * ap_meter.value().mean()
                    print(map)
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), map))
                    f.flush()
            scheduler.step()


            if map > BestMAP:
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestMAP =map
                save = {
                    'epoch': epoch_i,
                    'state_dict': recognizer.state_dict(),
                    #'att_state_dict': attention_model.state_dict(),
                    'best_loss': BestLoss,
                    'best_map': BestMAP
                }

                torch.save(save,
                           base_path + "best_%s2%s_CharadesEgo_recognizer_fintuned.pt" % (args.source_domain, args.target_domain))

        f.write("BestEpoch,{},BestLoss,{},BestMAP,{} \n".format(BestEpoch, BestLoss, BestMAP))
        f.flush()

    f.close()
