#!/bin/bash
python train_audio_only.py
python extract_vggsound_features.py
python get_audio_cluster_feature.py
python get_audio_pred_on_train.py
python get_unlikely_cls_per_video.py
python get_per_class_frequency.py
python train_encoder.py
python train_encoder_finetune.py
python train_recognizer.py
python train_recognizer_finetune.py
python test_recognizer.py