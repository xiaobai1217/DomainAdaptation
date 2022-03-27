from mmaction.apis import init_recognizer, inference_recognizer
import torch
from dataloader_audio import CharadesEgoAudio
import argparse
import tqdm
import os
import numpy as np
import math
import csv
import collections
import torch.nn as nn
from torch.optim import lr_scheduler
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from util1 import AveragePrecisionMeter
from torch.distributions import Categorical
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, help='input a str', default='3rd')
    parser.add_argument('--target_domain', type=str, help='input a str', default='1st')

    args = parser.parse_args()

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    checkpoint = torch.load("vggsound_avgpool.pth.tar")
    audio_model.load_state_dict(checkpoint['model_state_dict'])
    audio_model = audio_model.cuda()
    audio_model.eval()

    audio_cls_model = AudioAttGenModule()
    audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    audio_cls_model.fc = nn.Linear(512, 157)
    audio_cls_model = audio_cls_model.cuda()

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    log_path = base_path + "log%s2%s_audio.csv"%(args.source_domain, args.target_domain, )
    cmd = ['rm -rf ', log_path]
    os.system(' '.join(cmd))

    ap_meter = AveragePrecisionMeter(False)
    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    batch_size = 16
    lr = 1e-3#5e-4

    optim = torch.optim.Adam(list(audio_cls_model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optim, step_size=60, gamma=0.1)
    train_dataset = CharadesEgoAudio(split='train', domain=args.source_domain,)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    validate_dataset = CharadesEgoAudio(split='test', domain=args.target_domain,)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, num_workers=5, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    dataloaders = {'train': train_dataloader, 'val': validate_dataloader}
    BestLoss = float("inf")
    BestEpoch = 0
    BestMAP = 0
    with open(log_path, "a") as f:
        for epoch_i in range(120):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                count = 0
                total_loss = 0
                print(split)
                ap_meter.reset()
                audio_cls_model.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (spectrogram, labels)) in enumerate(dataloaders[split]):
                        labels = labels.cuda()
                        spectrogram = spectrogram.unsqueeze(1).cuda()

                        with torch.no_grad():
                            _, audio_feat,_ = audio_model(spectrogram)

                        audio_predict = audio_cls_model(audio_feat.detach())

                        ap_meter.add(audio_predict.data, labels)
                        loss = criterion(torch.sigmoid(audio_predict), labels)
                        total_loss += loss.item() * batch_size
                        if split == 'train':
                            optim.zero_grad()
                            loss.backward()
                            optim.step()

                        count += audio_predict.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, ".format(total_loss / float(count),
                                                                                                  loss.item(),))
                        pbar.update()
                    map = 100 * ap_meter.value().mean()
                    print(map)
                    f.write("{},{},{}\n".format(epoch_i, split, total_loss / float(count),))
                    f.flush()
                    # if epoch_i < 1:
            scheduler.step()

            if map > BestMAP:
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestMAP =map
                save = {
                    'epoch': epoch_i,
                    #'state_dict': model.state_dict(),
                    'audio_state_dict':audio_cls_model.state_dict(),
                    'best_loss': BestLoss,
                    'BestMAP': map
                }

                torch.save(save,
                           base_path + "best_%s2%s_audio.pt" % (args.source_domain, args.target_domain))

        f.write("BestEpoch,{},BestLoss,{},BestMAP,{} \n".format(BestEpoch, BestLoss, BestMAP))
        f.flush()

    f.close()
