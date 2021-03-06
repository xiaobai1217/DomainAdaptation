from mmaction.apis import init_recognizer, inference_recognizer
import torch
from dataloader_recognizer import EPICDOMAINRecognizer, EPICDOMAIN
import argparse
import tqdm
import os
import numpy as np
import math
import csv
import collections
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
import random
from vit_cls import ViTCls
from vit import ViT
import pdb
from config import config_func
from torch import nn, einsum

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.delegate_vectors = nn.Parameter(torch.randn(8, 512))
        self.audio_fc = nn.Sequential(nn.Linear(512, 512),
                                      nn.BatchNorm1d(512),
                                      nn.ReLU(),
                                      nn.Linear(512, 8))
        self.visual_fc = nn.Sequential(nn.Linear(2048, 64),
                                       nn.LayerNorm(64),
                                       nn.ReLU())
        self.audio_visual_fc = nn.Sequential(nn.Linear(512+64, 512),
                                             nn.BatchNorm1d(512),
                                             nn.ReLU(),
                                             nn.Linear(512, 8))

        self.transformer = ViTCls(dim=256, depth=3, heads=8, mlp_dim = 512, num_classes=8, dropout=0.15, emb_dropout=0.1)

    def forward(self, visual_feat,audio_feat, target=True):
        #b,c,f,h,w = visual_feat.size()
        #visual_feat2 = F.adaptive_avg_pool3d(visual_feat, (1,1,1))
        #visual_feat2 = visual_feat2.flatten(1)
        #visual_feat = visual_feat.view(b,c,f*h*w).transpose(1,2) #b,n,c

        #audio_feat = F.adaptive_avg_pool2d(audio_feat, (1,1))
        #audio_feat = audio_feat.flatten(1)
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

def train_one_step(model,attention_model, recognizer, clip, spectrogram, labels, ):
    #print(clip['imgs'].size())
    judge_label = (labels[:,1] == labels[:,0]).type(torch.FloatTensor).cuda()
    target_clip = clip['imgs'][:, 5:].cuda()
    b,n,c,f,h,w = target_clip.size()
    target_clip = target_clip.view(b*n, c, f, h, w)
    clip = clip['imgs'][:, :5].cuda()
    clip = clip.view(b*n, c, f, h, w)
    spectrogram = spectrogram.type(torch.FloatTensor)
    target_spectrogram = spectrogram[:, 1:2].cuda()
    spectrogram = spectrogram[:, :1].cuda()
    target_labels = labels[:,1].cuda()
    labels = labels[:,0].cuda()


    with torch.no_grad():
        v_feat = model.module.backbone.get_feature(clip)  # 16*5,1024,8,14,14
        _, audio_feat4att, _ = audio_model(spectrogram)  # 16,256,17,63
        _,audio_feat = audio_cls_model(audio_feat4att)

        target_v_feat = model.module.backbone.get_feature(target_clip)
        _, target_audio_feat4att, _ = audio_model(target_spectrogram)
        _,target_audio_feat = audio_cls_model(target_audio_feat4att)

        channel_att = attention_model(audio_feat4att.detach())
        channel_att = channel_att.unsqueeze(1).repeat(1,5,1)
        channel_att = channel_att.view(b*n, 1024)

        adapted_v_feat = torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * v_feat.detach()
        v_feat = model.module.backbone.get_predict(adapted_v_feat.detach()) #batch_size*5, 2048, 8, 7, 7
        target_channel_att = attention_model(target_audio_feat4att.detach())
        target_channel_att = target_channel_att.unsqueeze(1).repeat(1,5,1)
        target_channel_att = target_channel_att.view(b*n, 1024)
        adapted_target_v_feat = torch.sigmoid(target_channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * target_v_feat.detach()
        target_v_feat = model.module.backbone.get_predict(adapted_target_v_feat.detach())

        v_feat = F.adaptive_max_pool3d(v_feat, (1,1,1))
        v_feat = v_feat.flatten(1)
        target_v_feat = F.adaptive_max_pool3d(target_v_feat, (1,1,1))
        target_v_feat = target_v_feat.flatten(1)
        v_feat = v_feat.view(b, n, 2048)
        target_v_feat = target_v_feat.view(b, n, 2048)

    predict1, audio_att1, audio_att2 = recognizer(v_feat.detach(), audio_feat.detach(), target=False)
    target_predict1, _, _ = recognizer(target_v_feat.detach(), target_audio_feat.detach(), target=True)

    loss = criterion(predict1, labels)

    # idx = np.random.randint(0,5,(b,))[0]
    # loss3 = criterion(predict11[:,idx,:], labels) + criterion(target_predict11[:, idx, :], labels)
    labels2 = labels.unsqueeze(1).repeat(1,5).view(b*5)
    loss = (loss + criterion(target_predict1, target_labels))/2.0 + (criterion(audio_att1, labels) + criterion(audio_att2, labels2)*0.2)*0.1

    optim.zero_grad()
    loss.backward()
    #nn.utils.clip_grad_norm_(recognizer.parameters(), max_norm=1.0, norm_type=2)
    optim.step()
    return predict1, loss

def validate_one_step(model,attention_model, recognizer, clip, spectrogram, labels):
    clip = clip['imgs'].cuda()
    b,n,c,f,h,w = clip.size()
    clip = clip.view(b*n, c, f, h, w)
    spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
    labels = labels.cuda()

    with torch.no_grad():
        v_feat = model.module.backbone.get_feature(clip)  # 16,1024,8,14,14
        _, audio_feat4att, _ = audio_model(spectrogram)  # 16,256,17,63
        _,audio_feat = audio_cls_model(audio_feat4att)

        channel_att = attention_model(audio_feat4att.detach())
        channel_att = channel_att.unsqueeze(1).repeat(1,5,1)
        channel_att = channel_att.view(b*n, 1024)
        adapted_v_feat = torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * v_feat.detach()
        v_feat = model.module.backbone.get_predict(adapted_v_feat.detach()) #batch_size*5, 2048, 8, 7, 7

        v_feat = F.adaptive_max_pool3d(v_feat, (1,1,1))
        v_feat = v_feat.flatten(1)
        v_feat = v_feat.view(b, n, 2048)
        predict1,_,_ = recognizer(v_feat, audio_feat, target=True)

    loss = criterion(predict1, labels)

    return predict1, loss

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, help='input a str', default='D1')
    parser.add_argument('--target_domain', type=str, help='input a str', default='D2')
    parser.add_argument('--lr', type=float, help='input a str', default=7e-4)
    parser.add_argument('--depth', type=int, help='input a str', default=2)
    args = parser.parse_args()
    # config_file = 'configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py'
    # # download the checkpoint from model zoo and put it in `checkpoints/`
    # checkpoint_file = 'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'
    config_file = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow2.py'
    checkpoint_file = '/home/yzhang8/data/mmaction2_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'
    opts = config_func(args.source_domain, args.target_domain)
    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    # build the model from a config file and a checkpoint file
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

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    log_path = base_path + "log%s2%s_flow_transformer_cls.csv"%(args.source_domain, args.target_domain)
    cmd = ['rm -rf ', log_path]
    os.system(' '.join(cmd))

    batch_size = 8
    lr = args.lr#5e-4
    criterion = nn.CrossEntropyLoss()#nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()

    optim = torch.optim.Adam(list(recognizer.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)
    train_dataset = EPICDOMAINAdapter(split='train', source_domain=args.source_domain, target_domain=args.target_domain,modality='rgb', cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=3, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    validate_dataset = EPICDOMAIN(split='test', domain=args.target_domain, modality='rgb', cfg=cfg, use_audio=True)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, num_workers=3, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    dataloaders = {'train': train_dataloader, 'val': validate_dataloader}
    BestLoss = float("inf")
    BestEpoch = 0
    BestAcc = 0
    iter = 0
    print(opts.iter_num_trans)
    with open(log_path, "a") as f:
        for epoch_i in range(24):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                recognizer.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, spectrogram, labels)) in enumerate(dataloaders[split]):
                        # clip, 16,10,2,8,224,224
                        if split=='train':
                            predict1, loss = train_one_step(model, attention_model, recognizer, clip, spectrogram, labels,)
                        else:
                            predict1, loss = validate_one_step(model, attention_model, recognizer, clip, spectrogram,
                                                               labels, )
                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)

                        if split=='train':
                            labels = labels[:,0]
                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                  loss.item(),
                                                                                                  acc / float(count)))
                        pbar.update()

                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()
        BestAcc = acc / float(count)
        BestEpoch = epoch_i
        save = {
            # 'state_dict': model.state_dict(),
            'adapter_state_dict': recognizer.state_dict(),
            'best_acc': BestAcc
        }

        torch.save(save,
                   base_path + "best_%s2%s_slow_flow_transformer_cls.pt" % (args.source_domain, args.target_domain))

        f.write("{},{}\n".format(BestEpoch, BestAcc))
        f.flush()

        f.close()