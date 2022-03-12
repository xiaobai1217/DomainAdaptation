# Domain-Adaptation-Demo

**This is the demo code for training the audio-adaptive model with RGB and audio modalities on EPIC-Kitchens dataset, reproducing an mean accuracy of 59.2%. All code will be released once accepted.**

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
