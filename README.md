# Code for Audio-Adaptive Activity Recognition Across Video Domains

## EPIC-Kitchens

### RGB and audio
**This is the demo code for training the audio-adaptive model with RGB and audio modalities on EPIC-Kitchens dataset, reproducing an mean accuracy of 59.2%.**

* First download the data following the code provided by an existing work https://github.com/jonmun/MM-SADA-code

* Then prepare the audio files (.wav) from the videos:

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

* To run the code on 4 NVIDIA 1080Ti GPUs:
```
sh bash.sh
```

### Optical flow and audio
To be uploaded in March, 2022

## CharadesEgo

To be uploaded in March, 2022

## ActorShift Dataset

This dataset can be downloaded at https://drive.google.com/file/d/11ubcWqu1CiHwXBrfM9Ln5w0Y1Bz6kyq4/view?usp=sharing
