from mmaction.apis import init_recognizer, inference_recognizer
import torch
from dataloader_recognizer import CharadesEgoProjectionTraining, CharadesEgoProjectionValidating
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
from activity_sound_vit import ViTCls
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ActivitySoundTransformer(nn.Module):
    def __init__(self):
        super(ActivitySoundTransformer, self).__init__()
        self.delegate_vectors = nn.Parameter(torch.randn(8, 512))
        self.audio_fc = nn.Sequential(nn.Linear(512, 512),
                                      nn.BatchNorm1d(512),
                                      nn.ReLU(),
                                      nn.Linear(512, 8))
        self.visual_fc = nn.Sequential(nn.Linear(512, 64),
                                       nn.LayerNorm(64),
                                       nn.ReLU())
        self.audio_visual_fc = nn.Sequential(nn.Linear(512+64, 512),
                                             nn.BatchNorm1d(512),
                                             nn.ReLU(),
                                             nn.Linear(512, 8))

        self.to_patch_embedding2 = nn.Linear(2048,dim)
        self.to_patch_embedding3 = nn.Linear(256,dim)

        self.transformer = ViTCls(dim=256, depth=2, heads=8, mlp_dim = 512, num_classes=157, dropout=0.15, emb_dropout=0.1)

    def forward(self, slow_feat, fast_feat, audio_feat, target=True):
        v_feat = self.to_patch_embedding2(slow_x)
        v_feat2 = self.to_patch_embedding3(fast_x)
        visual_feat = torch.cat((v_feat, v_feat2), dim=1)
        #----------------------------------------------
        visual_feat2 = self.visual_fc(visual_feat)
        b,n,c = visual_feat.size()
        visual_feat2 = visual_feat2.view(b*n, 64)
        b1,_ = audio_feat.size()
        audio_att1 = self.audio_fc(audio_feat)
        audio_att = torch.softmax(audio_att1, dim=1)
        delegate_vectors = self.delegate_vectors.unsqueeze(0).repeat(b*n,1,1)
        #print(audio_att.size(), delegate_vectors.size())
        audio_att = audio_att.unsqueeze(1).repeat(1,5,1).view(5*b1,-1)
        new_audio_vector = einsum('b d, b d i -> b i', audio_att, delegate_vectors)
        vector1 = torch.cat((new_audio_vector, visual_feat2), dim=1)
        audio_att22 = self.audio_visual_fc(vector1)
        audio_att2 = torch.softmax(audio_att22, dim=1)
        new_audio_vector2 = einsum('b d, b d i -> b i', audio_att2, delegate_vectors)
        new_audio_vector2 = new_audio_vector2.view(b,n,-1)
        #print(visual_feat.size(), new_audio_vector2.size())
        pred = self.transformer(visual_feat, new_audio_vector2, target=target)

        return pred, audio_att1, audio_att22


def train_one_step(model, audio_model, att_model, adapter, clip, spectrogram, labels, target_clip, target_labels, target_spectrogram,a_target_clip, judge_labels, a_target_spectrogram):
    target_clip = target_clip['imgs'].squeeze(1).cuda()
    target_labels = target_labels.cuda()
    target_spectrogram = target_spectrogram.type(torch.FloatTensor).cuda().unsqueeze(1)

    a_target_clip = a_target_clip['imgs'].squeeze(1).cuda()
    judge_labels = judge_labels.type(torch.FloatTensor).cuda()
    a_target_spectrogram = a_target_spectrogram.type(torch.FloatTensor).cuda().unsqueeze(1)

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        _, audio_feat, audiofeat4trans = audio_model(spectrogram)  # 16,256,17,63
        feat = (x_slow.detach(), x_fast.detach()) #slow 16,1280,16,14,14, fast 16,128,64,14,14

        target_x_slow, target_x_fast = model.module.backbone.get_feature(target_clip)
        _, target_audio_feat, target_audiofeat4trans = audio_model(target_spectrogram)  # 16,256,17,63
        target_feat = (target_x_slow.detach(), target_x_fast.detach()) #slow 16,1280,16,14,14, fast 16,128,64,14,14

        a_target_x_slow, a_target_x_fast = model.module.backbone.get_feature(a_target_clip)
        _, a_target_audio_feat, a_target_audiofeat4trans = audio_model(a_target_spectrogram)  # 16,256,17,63
        a_target_feat = (a_target_x_slow.detach(), a_target_x_fast.detach()) #slow 16,1280,16,14,14, fast 16,128,64,14,14

        channel_att, channel_att2 = att_model(audio_feat.detach())
        adapted_v_feat = [torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[0], torch.sigmoid(channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[1]]
        slow_feat, fast_feat = model.module.backbone.get_predict(adapted_v_feat)

        target_channel_att, target_channel_att2 = att_model(target_audio_feat.detach())
        adapted_v_feat = [torch.sigmoid(target_channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * target_feat[0],
                          torch.sigmoid(target_channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * target_feat[1]]
        target_slow_feat, target_fast_feat = model.module.backbone.get_predict(adapted_v_feat)
        #target_predict1 = model.module.cls_head(target_predict1)

        a_target_channel_att, a_target_channel_att2 = att_model(a_target_audio_feat.detach())
        a_adapted_v_feat = [torch.sigmoid(a_target_channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * a_target_feat[0],
                          torch.sigmoid(a_target_channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * a_target_feat[1]]
        a_target_slow_feat, a_target_fast_feat = model.module.backbone.get_predict(a_adapted_v_feat)

    #2048,16,7,7; 256,64,7,7
    slow_feat = F.adaptive_max_pool3d(slow_feat.detach(), (16,1,1)).squeeze(3).squeeze(3)
    fast_feat = F.adaptive_max_pool3d(fast_feat.detach(), (64,1,1)).squeeze(3).squeeze(3)
    slow_feat = slow_feat.transpose(1,2).contiguous().detach()
    fast_feat = fast_feat.transpose(1,2).contiguous().detach() 
    predict1,audio_att1, audio_att2 = adapter(slow_feat, fast_feat, audiofeat4trans.unsqueeze(1).detach(), target=False)

    target_slow_feat = F.adaptive_max_pool3d(target_slow_feat.detach(), (16,1,1)).squeeze(3).squeeze(3)
    target_fast_feat = F.adaptive_max_pool3d(target_fast_feat.detach(), (64,1,1)).squeeze(3).squeeze(3)
    target_slow_feat = target_slow_feat.transpose(1,2).contiguous().detach()
    target_fast_feat = target_fast_feat.transpose(1,2).contiguous().detach()
    target_predict1,_,_ = adapter(target_slow_feat, target_fast_feat, target_audiofeat4trans.unsqueeze(1).detach(), target=True)

    loss = (nn.BCELoss()(F.sigmoid(predict1), labels) + nn.BCELoss()(F.sigmoid(target_predict1), target_labels) )/2.0+ (criterion(audio_att1, labels) + criterion(audio_att2, labels2)*0.2)*0.1

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss, predict1


def validate_one_step(model, audio_model, att_model, adapter, clip, spectrogram, labels):
    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        _, audio_feat, audiofeat4trans = audio_model(spectrogram)  # 16,256,17,63
        feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        channel_att, channel_att2 = att_model(audio_feat.detach())
        adapted_v_feat = [torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[0],
                          torch.sigmoid(channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * feat[1]]

        slow_feat, fast_feat = model.module.backbone.get_predict(adapted_v_feat)
        #predict1 = model.module.cls_head(predict1)

        slow_feat = F.adaptive_max_pool3d(slow_feat.detach(), (16, 1, 1)).squeeze(3).squeeze(3)
        fast_feat = F.adaptive_max_pool3d(fast_feat.detach(), (64, 1, 1)).squeeze(3).squeeze(3)
        slow_feat = slow_feat.transpose(1,2).contiguous().detach()
        fast_feat = fast_feat.transpose(1,2).contiguous().detach()
        predict1,_,_ = adapter(slow_feat, fast_feat, audiofeat4trans.unsqueeze(1).detach(), target=True)

    loss = criterion(F.sigmoid(predict1), labels)
    return loss, predict1

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, help='input a str', default='3rd')
    parser.add_argument('--target_domain', type=str, help='input a str', default='1st')
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--emb_dropout', type=float, default=0.1)

    args = parser.parse_args()

    # config_file = 'configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py'
    # # download the checkpoint from model zoo and put it in `checkpoints/`
    # checkpoint_file = 'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = '/home/yzhang8/data/mmaction2_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

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


    adapter = ActivitySoundTransformer()
    adapter = adapter.cuda()
    adapter = torch.nn.DataParallel(adapter)

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    log_path = base_path + "log%s2%s_transformer.csv"%(args.source_domain, args.target_domain)
    cmd = ['rm -rf ', log_path]
    os.system(' '.join(cmd))

    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    batch_size = 16
    lr = args.lr

    optim = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optim, step_size=60, gamma=0.1)
    train_dataset = CharadesEgoProjectionTraining(split='train', source_domain=args.source_domain, target_domain=args.target_domain,modality='rgb', cfg=cfg, )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=3, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    validate_dataset = CharadesEgoProjectionValidating(split='test', domain=args.target_domain, modality='rgb', cfg=cfg,)
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

                adapter.train(split=='train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, labels, spectrogram, target_clip, target_labels, target_spectrogram,a_target_clip, judge_labels, a_target_spectrogram)) in enumerate(dataloaders[split]):
                        clip = clip['imgs'].cuda().squeeze(1)
                        labels = labels.cuda()
                        #label_weights = label_weights.cuda().unsqueeze(1) / torch.sum(label_weights) * label_weights.size()[0]
                        spectrogram = spectrogram.type(torch.FloatTensor).cuda().unsqueeze(1)

                        if split=='train':
                            loss, predict1 = train_one_step(model, audio_model, attention_model, adapter, clip, spectrogram, labels, target_clip, target_labels, target_spectrogram, a_target_clip, judge_labels, a_target_spectrogram)
                        else:
                            loss, predict1 = validate_one_step(model,audio_model, attention_model, adapter, clip, spectrogram, labels)

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
                    'state_dict': adapter.state_dict(),
                    #'att_state_dict': attention_model.state_dict(),
                    'best_loss': BestLoss,
                    'best_map': BestMAP
                }

                torch.save(save,
                           base_path + "best_%s2%s_CharadesEgo_transformer.pt" % (args.source_domain, args.target_domain))

        f.write("BestEpoch,{},BestLoss,{},BestMAP,{} \n".format(BestEpoch, BestLoss, BestMAP))
        f.flush()

    f.close()
