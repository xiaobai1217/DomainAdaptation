from mmaction.apis import init_recognizer, inference_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import math
import csv
import collections
import torch.nn as nn
from torch.optim import lr_scheduler
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from vit import ViT
from dataloader_audio_adaptive_encoder import EPICDOMAINClusters, EPICDOMAIN
from config import config_func
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def train_one_step(model, attention_model, clip, spectrogram, labels, target_label, weights):
    target_clip = clip['imgs'][:, 1].cuda()
    clip = clip['imgs'][:, 0].cuda()
    spectrogram = spectrogram.type(torch.FloatTensor)
    target_spectrogram = spectrogram[:, 1:2].cuda()
    spectrogram = spectrogram[:, :1].cuda()
    weights = weights.type(torch.FloatTensor).cuda()
    target_label = target_label.cuda()
    labels = labels.cuda()

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)  # 16,1024,8,14,14
        _, audio_feat, _ = audio_model(spectrogram)  # 16,256,17,63
        target_slow, target_fast = model.module.backbone.get_feature(target_clip)
        _, target_audio_feat, _ = audio_model(target_spectrogram)
        v_feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14
        target_v_feat = (target_slow.detach(), target_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

    channel_att, channel_att2 = attention_model(audio_feat.detach())
    adapted_v_feat = [torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * v_feat[0],
                      torch.sigmoid(channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * v_feat[1]]
    v_feat = model.module.backbone.get_predict(adapted_v_feat)
    predict1 = model.module.cls_head(v_feat)

    target_channel_att, target_channel_att2 = attention_model(target_audio_feat.detach())
    adapted_target_feat = [torch.sigmoid(target_channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * target_v_feat[0],
                      torch.sigmoid(target_channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * target_v_feat[1]]

    adapted_target_feat = model.module.backbone.get_predict(adapted_target_feat)
    target_predict1 = model.module.cls_head(adapted_target_feat)


    loss = torch.mean(criterion(predict1, labels) * weights)
    target_predict1 = torch.softmax(target_predict1, dim=1)
    loss2 = torch.mean(torch.sum(-target_label * torch.log(1 - target_predict1 + 1e-7), dim=1))
    loss = loss2 * 0.01 + loss

    optim.zero_grad()
    loss.backward()
    #nn.utils.clip_grad_norm_(attention_model.parameters(), max_norm=1.0, norm_type=2)
    #nn.utils.clip_grad_norm_(model.module.backbone.layer4.parameters(), max_norm=1.0, norm_type=2)
    #nn.utils.clip_grad_norm_(model.module.cls_head.parameters(), max_norm=1.0, norm_type=2)
    optim.step()
    return predict1, loss

def validate_one_step(model, attention_model, clip, spectrogram,labels):
    clip = clip['imgs'].cuda().squeeze(1)
    spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
    labels = labels.cuda()

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)  # 16,1024,8,14,14
        _, audio_feat, _ = audio_model(spectrogram)  # 16,256,17,63
        v_feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        channel_att, channel_att2 = attention_model(audio_feat.detach())
        adapted_v_feat = [torch.sigmoid(channel_att.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * v_feat[0],
                          torch.sigmoid(channel_att2.unsqueeze(2).unsqueeze(2).unsqueeze(2)) * v_feat[1]]
        v_feat = model.module.backbone.get_predict(adapted_v_feat)
        predict1 = model.module.cls_head(v_feat)

    loss = torch.mean(criterion(predict1, labels))

    return predict1, loss


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, help='input a str', default='D1')
    parser.add_argument('--target_domain', type=str, help='input a str', default='D2')
    args = parser.parse_args()
    opts = config_func(args.source_domain, args.target_domain)

    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = '/home/xxx/data/mmaction2_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

     # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, checkpoint_file, device=device,use_frames=True)
    model.cls_head.fc_cls = nn.Linear(2304, 8).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    checkpoint = torch.load("vggsound_avgpool.pth.tar")
    audio_model.load_state_dict(checkpoint['model_state_dict'])
    audio_model = audio_model.cuda()
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.eval()

    attention_model = ViT(dim=256, depth=8, heads=8,mlp_dim=512,dropout=0.15, emb_dropout=0.1, dim_head=64)
    attention_model = attention_model.cuda()
    attention_model = torch.nn.DataParallel(attention_model)
    # audio_att_model = AudioAttGenModule()
    # audio_att_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # audio_att_model.fc = nn.Linear(512, 1024)
    # audio_att_model = audio_att_model.cuda()
    # audio_att_model = torch.nn.DataParallel(audio_att_model)


    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    log_path = base_path + "log%s2%s_1stStage.csv"%(args.source_domain, args.target_domain)
    cmd = ['rm -rf ', log_path]
    os.system(' '.join(cmd))

    criterion = nn.CrossEntropyLoss(reduce=False)#nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()
    batch_size = 16

    optim = torch.optim.Adam(list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) +list(model.module.cls_head.parameters()) + list(attention_model.parameters()), lr=opts.lr_1stStage, weight_decay=1e-4)
    # scheduler = lr_scheduler.StepLR(optim, step_size=60, gamma=0.1)
    train_dataset = EPICDOMAINClusters(split='train', source_domain=args.source_domain, target_domain=args.target_domain,cfg=cfg, beta=0.999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)

    test_dataset = EPICDOMAIN(split='test', domain=args.target_domain, modality='flow', cfg=cfg, use_audio=True,)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    dataloaders = {'train': train_dataloader, 'test': test_dataloader }
    BestLoss = float("inf")
    BestEpoch = 0
    BestAcc = 0
    with open(log_path, "a") as f:
        for epoch_i in range(15):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'test']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                model.train(split == 'train')
                attention_model.train(split=='train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, spectrogram, labels, target_label, weights, )) in enumerate(dataloaders[split]):
                        if split=='train':
                            predict1, loss = train_one_step(model, attention_model, clip, spectrogram, labels, target_label, weights,)
                        else:
                            predict1, loss = validate_one_step(model, attention_model, clip, spectrogram,labels)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)

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

            BestLoss = total_loss / float(count)
            BestEpoch = epoch_i
            BestAcc = acc / float(count)
            save = {
                'epoch': epoch_i,
                'state_dict': model.state_dict(),
                'audio_state_dict': attention_model.state_dict(),
                'best_loss': total_loss / float(count),
                'best_acc': acc / float(count)
            }

            torch.save(save,
                       base_path + "best_%s2%s_1stStage.pt" % (args.source_domain, args.target_domain))

        f.write("BestEpoch,{},BestLoss,{},BestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc))
        f.flush()

    f.close()
