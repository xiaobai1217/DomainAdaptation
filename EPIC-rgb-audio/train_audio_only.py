from mmaction.apis import init_recognizer, inference_recognizer
import torch
from dataloader_audio_adaptive_encoder import EPICDOMAINAudio
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
from torch.distributions import Categorical

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, help='input a str', default='D2')
    parser.add_argument('--target_domain', type=str, help='input a str', default='D3')

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
    audio_cls_model.fc = nn.Linear(512, 8)
    audio_cls_model = audio_cls_model.cuda()

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    log_path = base_path + "log%s2%s_audio.csv"%(args.source_domain, args.target_domain, )
    cmd = ['rm -rf ', log_path]
    os.system(' '.join(cmd))

    criterion = nn.CrossEntropyLoss()#nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()
    batch_size = 16
    lr = 1e-2#5e-4

    optim = torch.optim.Adam(list(audio_cls_model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optim, step_size=60, gamma=0.1)
    train_dataset = EPICDOMAINAudio(split='train', domain=args.source_domain,modality='rgb', )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    validate_dataset = EPICDOMAINAudio(split='test', domain=args.target_domain, modality='rgb',)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, num_workers=5, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)
    dataloaders = {'train': train_dataloader, 'val': validate_dataloader}
    BestLoss = float("inf")
    BestEpoch = 0
    BestAcc = 0
    with open(log_path, "a") as f:
        for epoch_i in range(4):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                audio_cls_model.train(split == 'train')
                #model.train(split=='train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (spectrogram, labels)) in enumerate(dataloaders[split]):
                        labels = labels.cuda()
                        spectrogram = spectrogram.unsqueeze(1).cuda()

                        with torch.no_grad():
                            _, audio_feat, _ = audio_model(spectrogram)

                        audio_predict,_ = audio_cls_model(audio_feat.detach())



                        loss = criterion(audio_predict, labels)
                        total_loss += loss.item() * batch_size
                        if split == 'train':
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                        _, predict = torch.max(audio_predict, dim=1)

                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += audio_predict.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                  loss.item(),
                                                                                                  acc / float(count)))
                        pbar.update()
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()
                    # if epoch_i < 1:
            scheduler.step()

            BestLoss = total_loss / float(count)
            BestEpoch = epoch_i
            BestAcc = acc / float(count)
            save = {
                'epoch': epoch_i,
                #'state_dict': model.state_dict(),
                'audio_state_dict':audio_cls_model.state_dict(),
                'best_loss': BestLoss,
                'best_acc': BestAcc
            }

            torch.save(save,
                       base_path + "best_%s2%s_audio.pt" % (args.source_domain, args.target_domain))

        f.write("BestEpoch,{},BestLoss,{},BestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc))
        f.flush()

    f.close()