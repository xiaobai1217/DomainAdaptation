from mmaction.apis import init_recognizer, inference_recognizer
import torch
from dataloader_encoder import CharadesEgoAligningTraining, CharadesEgoAligningValidating
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
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# loss, predict1 = train_one_step(model, audio_model, attention_model, clip, spectrogram, labels, target_clip, target_spectrogram, likely_label, unlikely_label)
def train_one_step(model, audio_model, att_model, clip, spectrogram, labels, label_weights, target_clip, target_labels, target_spectrogram, unlabeled_target, unlabeled_spectrogram,
                  unlikely_label):
    label_weights = label_weights.cuda().unsqueeze(1) / torch.sum(label_weights) * label_weights.size()[0]
    target_clip = target_clip['imgs'].squeeze(1).cuda()
    target_labels = target_labels.cuda()
    target_spectrogram = target_spectrogram.type(torch.FloatTensor).cuda().unsqueeze(1)
    unlabeled_target = unlabeled_target['imgs'].squeeze(1).cuda()
    unlabeled_spectrogram = unlabeled_spectrogram.type(torch.FloatTensor).cuda().unsqueeze(1)
    unlikely_label = unlikely_label.cuda()


    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        _, audio_feat, _ = audio_model(spectrogram)  # 16,256,17,63
        feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        l_target_x_slow, l_target_x_fast = model.module.backbone.get_feature(target_clip)
        _, l_target_audio_feat, _ = audio_model(target_spectrogram)  # 16,256,17,63
        l_target_feat = (
        l_target_x_slow.detach(), l_target_x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        target_x_slow, target_x_fast = model.module.backbone.get_feature(unlabeled_target)
        _, target_audio_feat, _ = audio_model(unlabeled_spectrogram)  # 16,256,17,63
        target_feat = (target_x_slow.detach(), target_x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

    channel_att, channel_att2 = att_model(audio_feat.detach())
    adapted_v_feat = [torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[0],
                      torch.sigmoid(channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[1]]
    predict1 = model.module.backbone.get_predict(adapted_v_feat)
    predict1 = model.module.cls_head(predict1)

    l_target_channel_att, l_target_channel_att2 = att_model(l_target_audio_feat.detach())
    adapted_v_feat = [torch.sigmoid(l_target_channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * l_target_feat[0],
                      torch.sigmoid(l_target_channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * l_target_feat[1]]
    l_target_predict1 = model.module.backbone.get_predict(adapted_v_feat)
    l_target_predict1 = model.module.cls_head(l_target_predict1)

    target_channel_att, target_channel_att2 = att_model(target_audio_feat.detach())
    adapted_v_feat = [torch.sigmoid(target_channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * target_feat[0],
                      torch.sigmoid(target_channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * target_feat[1]]
    target_predict1 = model.module.backbone.get_predict(adapted_v_feat)
    target_predict1 = model.module.cls_head(target_predict1)
    target_predict1 = F.sigmoid(target_predict1)

    loss = F.binary_cross_entropy_with_logits(input=predict1, target=labels, weight=label_weights) + nn.BCELoss()(F.sigmoid(l_target_predict1), target_labels)
    # loss = nn.BCELoss()(F.sigmoid(l_target_predict1), l_target_labels)
    target_loss = torch.mean(
        torch.sum(-torch.log(1 - target_predict1 + 1e-7) * unlikely_label, dim=1) / torch.sum(unlikely_label + 1,
                                                                                              dim=1))
    #target_loss2 = torch.mean(
    #    torch.sum(-torch.log(target_predict1 + 1e-7) * likely_label) / torch.sum(likely_label + 1, dim=1))

    loss = loss + target_loss * 0.01

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss, predict1


def validate_one_step(model, audio_model, att_model, clip, spectrogram, labels):

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        _, audio_feat, _ = audio_model(spectrogram)  # 16,256,17,63
        feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        channel_att, channel_att2 = att_model(audio_feat.detach())
        adapted_v_feat = [torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[0],
                          torch.sigmoid(channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[1]]

        predict1 = model.module.backbone.get_predict(adapted_v_feat)
        predict1 = model.module.cls_head(predict1)

    loss = criterion(F.sigmoid(predict1), labels)
    return loss, predict1


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, help='input a str', default='3rd')
    parser.add_argument('--target_domain', type=str, help='input a str', default='1st')
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--beta', type=float, default=0.999)
    args = parser.parse_args()

    # config_file = 'configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py'
    # # download the checkpoint from model zoo and put it in `checkpoints/`
    # checkpoint_file = 'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = '/home/yzhang8/data/mmaction2_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    # assign the desired device.
    device = 'cuda:0'  # or 'cpu'
    device = torch.device(device)

    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(2304, 157).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)
    #checkpoint = torch.load(
    #    "checkpoints/best_%s2%s_CharadesEgo_reweighting_att10.pt" % (args.source_domain, args.target_domain,))
    #model.load_state_dict(checkpoint['state_dict'])
    ap_meter = AveragePrecisionMeter(False)

    attention_model = ViT(dim=256, depth=8, heads=8, mlp_dim=512, dropout=0.15, emb_dropout=0.1, dim_head=64)
    attention_model = attention_model.cuda()
    attention_model = torch.nn.DataParallel(attention_model)
    #attention_model.load_state_dict(checkpoint['att_state_dict'])

    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    checkpoint = torch.load("vggsound_avgpool.pth.tar")
    audio_model.load_state_dict(checkpoint['model_state_dict'])
    audio_model = audio_model.cuda()
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.eval()

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    log_path = base_path + "log%s2%s_encoder.csv" % (args.source_domain, args.target_domain)
    cmd = ['rm -rf ', log_path]
    os.system(' '.join(cmd))

    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    batch_size = 8
    lr = args.lr  # D1->D3, D1->D2 2e-3, others, 1e-2
    # optim = torch.optim.Adam([{'params': model.backbone.layer3.parameters(), 'lr': 1e-4}, {'params': list(model.backbone.layer4.parameters())+list(model.cls_head.parameters())}], lr=lr, weight_decay=1e-4)
    # optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    optim = torch.optim.Adam(list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) + list(
        attention_model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optim, step_size=60, gamma=0.1)
    train_dataset = CharadesEgoAligningTraining(split='train', source_domain=args.source_domain,
                                                target_domain=args.target_domain, modality='rgb', cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    validate_dataset = CharadesEgoAligningValidating(split='test', domain=args.target_domain, modality='rgb', cfg=cfg, )
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, num_workers=4,
                                                      shuffle=True,
                                                      pin_memory=(device.type == "cuda"), drop_last=True)
    dataloaders = {'train': train_dataloader, 'val': validate_dataloader}
    BestLoss = float("inf")
    BestEpoch = 0
    BestMAP = 0
    with open(log_path, "a") as f:
        for epoch_i in range(20):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                max_record = 0
                count = 0
                total_loss = 0
                print(split)
                ap_meter.reset()

                model.train(split == 'train')
                attention_model.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    #  data, label1, spectrogram,label_weight1, target_data, target_label1, target_spectrogram, u_target_data, target_spectrogram2, unlikely_label
                    for (i, (clip, labels, spectrogram, label_weights, target_clip, target_labels, target_spectrogram, unlabeled_target, unlabeled_spectrogram,
                             unlikely_label)) in enumerate(dataloaders[split]):
                        clip = clip['imgs'].squeeze(1).cuda()
                        labels = labels.cuda()
                        spectrogram = spectrogram.type(torch.FloatTensor).cuda().unsqueeze(1)

                        if split == 'train':
                            loss, predict1 = train_one_step(model, audio_model, attention_model, clip, spectrogram,
                                                            labels, label_weights, target_clip, target_labels, target_spectrogram, unlabeled_target, unlabeled_spectrogram,
                                                            unlikely_label)
                        else:

                            loss, predict1 = validate_one_step(model, audio_model, attention_model, clip, spectrogram,
                                                               labels)
                            #clip = clip['imgs']
                        ap_meter.add(predict1.data, labels)

                        total_loss += loss.item() * batch_size

                        max_record += torch.max(predict1.detach()).item() * clip.size()[0]

                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f},{:.4f}".format(total_loss / float(count),
                                                                                       loss.item(),
                                                                                       max_record / float(count)))
                        pbar.update()
                    map = 100 * ap_meter.value().mean()
                    print(map)
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), map))
                    f.flush()
            scheduler.step()

            if map > BestMAP:
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestMAP = map
                save = {
                    'epoch': epoch_i,
                    'state_dict': model.state_dict(),
                    'att_state_dict': attention_model.state_dict(),
                    'best_loss': BestLoss,
                    'best_map': BestMAP
                }

                torch.save(save,
                           base_path + "best_%s2%s_CharadesEgo_encoder.pt" % (
                           args.source_domain, args.target_domain))

        f.write("BestEpoch,{},BestLoss,{},BestMAP,{} \n".format(BestEpoch, BestLoss, BestMAP))
        f.flush()

    f.close()
