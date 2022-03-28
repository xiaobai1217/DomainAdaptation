#!/bin/bash
#--------------------------------------------------- train the audio encoder ---------------------------------------
python train_audio_only.py --source_domain D1 --target_domain D2
python train_audio_only.py --source_domain D1 --target_domain D3
python train_audio_only.py --source_domain D2 --target_domain D1
python train_audio_only.py --source_domain D2 --target_domain D3
python train_audio_only.py --source_domain D3 --target_domain D1
python train_audio_only.py --source_domain D3 --target_domain D2
#--------------------------------------------------- extract audio features ---------------------------------------
python extract_sound_features.py --source_domain D1 --target_domain D2
python extract_sound_features.py --source_domain D1 --target_domain D3
python extract_sound_features.py --source_domain D2 --target_domain D1
python extract_sound_features.py --source_domain D2 --target_domain D3
python extract_sound_features.py --source_domain D3 --target_domain D1
python extract_sound_features.py --source_domain D3 --target_domain D2
#--------------------------------------------------- cluster source domain data by audio features ---------------------------------------
python get_audio_cluster_feature.py --source_domain D1 --target_domain D2
python get_audio_cluster_feature.py --source_domain D1 --target_domain D3
python get_audio_cluster_feature.py --source_domain D2 --target_domain D1
python get_audio_cluster_feature.py --source_domain D2 --target_domain D3
python get_audio_cluster_feature.py --source_domain D3 --target_domain D1
python get_audio_cluster_feature.py --source_domain D3 --target_domain D2
#--------------------------------------------------- get pseudo-absent labels for target domain training data ---------------------------------------
python get_pseudo_absent_labels.py --source_domain D1 --target_domain D2
python get_pseudo_absent_labels.py --source_domain D1 --target_domain D3
python get_pseudo_absent_labels.py --source_domain D2 --target_domain D1
python get_pseudo_absent_labels.py --source_domain D2 --target_domain D3
python get_pseudo_absent_labels.py --source_domain D3 --target_domain D1
python get_pseudo_absent_labels.py --source_domain D3 --target_domain D2
#--------------------------------------------------- train audio-adaptive activity encoder based on audio-adaptive loss ---------------------------------------
python train_audio_adaptive_encoder.py --source_domain D1 --target_domain D2
python train_audio_adaptive_encoder.py --source_domain D1 --target_domain D3
python train_audio_adaptive_encoder.py --source_domain D2 --target_domain D1
python train_audio_adaptive_encoder.py --source_domain D2 --target_domain D3
python train_audio_adaptive_encoder.py --source_domain D3 --target_domain D1
python train_audio_adaptive_encoder.py --source_domain D3 --target_domain D2
#--------------------------------------------------- get hard pseudo labels ---------------------------------------
python get_pseudo_labels.py --source_domain D1 --target_domain D2
python get_pseudo_labels.py --source_domain D1 --target_domain D3
python get_pseudo_labels.py --source_domain D2 --target_domain D1
python get_pseudo_labels.py --source_domain D2 --target_domain D3
python get_pseudo_labels.py --source_domain D3 --target_domain D1
python get_pseudo_labels.py --source_domain D3 --target_domain D2
#--------------------------------------------------- train activity sound transformer ---------------------------------------
python train_recognizer.py --source_domain D1 --target_domain D2
python train_recognizer.py --source_domain D1 --target_domain D3
python train_recognizer.py --source_domain D2 --target_domain D1
python train_recognizer.py --source_domain D2 --target_domain D3
python train_recognizer.py --source_domain D3 --target_domain D1
python train_recognizer.py --source_domain D3 --target_domain D2
