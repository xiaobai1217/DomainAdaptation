# Audio-Adaptive Activity Recognition Across Video Domains (CVPR 2022)

More info can be found on our [project page](https://xiaobai1217.github.io/DomainAdaptation/)


<img width="400" alt="1st-figure" src="https://user-images.githubusercontent.com/22721775/159116800-2df8b1f2-c622-4e4e-8e9a-53be7bc1ae93.png">

## Demo Video


[![Watch the video](https://user-images.githubusercontent.com/22721775/159116907-5e4f934c-9ec9-41b2-acb4-7b59c8219cb6.png)](https://youtube.com/embed/Lh3gb6NMhB4)

## Pretrained weights we used

Audio model: [link](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar) </br>
SlowFast model for RGB modality: [link](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) </br>
Slow-Only model for optical flow modality: [link](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth)

## EPIC-Kitchens

```diff
- There are two streams in total, one is the audio-adaptive model with RGB and audio modalities, and the other is the audio-adaptive model with optical flow and audio modalities.
- We average the predictions from the two streams in the end for an mean accuracy of 61.0%.
```

* Prepare the audio files (.wav) from the videos:

```
python generate_sound_files.py
```

* Environments:

```
PyTorch 1.7.0
mmcv-full 1.2.7
mmaction2 0.13.0
cudatoolkit 10.1.243
```

* The directory structure should be modified to match:

```
├── rgb
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01
|   |   |   |     ├── frame_0000000000.jpg
|   |   |   |     ├── ...
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3


├── flow
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01 
|   |   |   |   ├── u
|   |   |   |   |   ├── frame_0000000000.jpg
|   |   |   |   |   ├── ...
|   |   |   |   ├── v
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3
```


### RGB and audio
**This is the demo code for training the audio-adaptive model with RGB (SlowFast backbone) and audio modalities on EPIC-Kitchens dataset, reproducing an mean accuracy of 59.2%.**

* Before running the code, you need to change the data paths to yours in `dataloader_*.py`, `train_*.py`, `test_*.py` and `get_*.py`. 

* First download the data following the code provided by an existing work https://github.com/jonmun/MM-SADA-code
* Go to the sub-directory
```
cd EPIC-rgb-audio
```


* To run the code on 4 NVIDIA 1080Ti GPUs:
```
sh bash.sh
```

### Optical flow and audio
**This is the demo code for training the audio-adaptive model with optical flow (Slow-Only backbone) and audio modalities on EPIC-Kitchens dataset, reproducing an mean accuracy of 53.9%.**

* Before running the code, you need to change the data paths to yours in `dataloader_*.py`, `train_*.py`, `test_*.py` and `get_*.py`. 

Note that the clusters and absent-pseudo labels generated by audio are the same as those in the "RGB and audio" code

* Go to the sub-directory
```
cd EPIC-flow-audio
```


* To run the code on 4 NVIDIA 1080Ti GPUs:
```
sh bash.sh
```

## CharadesEgo
This code conducts semi-supervised domain adaptation with all the source (3rd-person view) data and half of the target (1st-person view) data, based on RGB (SlowFast backbone) and audio modalities, reproducing an mAP of 26.3%. 

* The directory structure should be modified to match:

```
├── CharadesEgo
|   ├── audio
|   |   ├── 005BUEGO.wav
|   |   ├── 005BU.wav
|   |   ├── ...
|   ├── CharadesEgo_v1_rgb
|   |   ├── 005BU
|   |   |   ├── 005BU-000001.jpg
|   |   |   ├── 005BU-000002.jpg
|   |   |   ├── ...
|   |   ├── 005BUEGO
|   |   ├── ...
|   ├── Labels
|   |   ├── 005BU
|   |   |   ├── frame_0000000001_0000000174.csv
|   |   |   ├── ...
|   |   ├── 005BUEGO
|   |   ├── ...
|   ├── CharadesEgo_v1_train_only1st.csv
|   ├── CharadesEgo_v1_train_only3rd.csv
|   ├── CharadesEgo_v1_test_only1st.csv
|   ├── CharadesEgo_v1_test_only3rd.csv
```

Here the "Labels" directory contains the labels that we generated by ourselves according to the csv files provided by the CharadesEgo dataset. You can directly download it from this [link](https://drive.google.com/file/d/1FoHe_zIy6OWYB0ISpoPEY5M4doOwwm06/view?usp=sharing) or run `generate_labels.py` to create it by yourself. 


* Before running the code, you need to change the data paths to yours in `dataloader_*.py`, `train_*.py`, `test_*.py` and `get_*.py`. 

* Go to the sub-directory
```
cd CharadesEgo
```


* To run the code on 4 NVIDIA 1080Ti GPUs:
```
sh bash.sh
```


## ActorShift Dataset

This dataset can be downloaded at https://uvaauas.figshare.com/articles/dataset/ActorShift_zip/19387046


## Contact

If you have any questions, you can send an email to y.zhang9@uva.nl

## Citation
If you find the code useful in your research please cite:
```
@inproceedings{ZhangCVPR2022,
title = {Audio-Adaptive Activity Recognition Across Video Domains},
author = {Yunhua Zhang and Hazel Doughty and Ling Shao and Cees G M Snoek},
year = {2022},
date = {2022-06-02},
urldate = {2022-06-01},
booktitle = {CVPR},
keywords = {},
pubstate = {published},
tppubtype = {inproceedings}
}
```
