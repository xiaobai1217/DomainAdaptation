#!/bin/bash
python train_audio_adaptive_encoder.py --source_domain D1 --target_domain D2
python train_audio_adaptive_encoder.py --source_domain D1 --target_domain D3
python train_audio_adaptive_encoder.py --source_domain D2 --target_domain D1
python train_audio_adaptive_encoder.py --source_domain D2 --target_domain D3
python train_audio_adaptive_encoder.py --source_domain D3 --target_domain D1
python train_audio_adaptive_encoder.py --source_domain D3 --target_domain D2
python get_pseudo_labels.py --source_domain D1 --target_domain D2
python get_pseudo_labels.py --source_domain D1 --target_domain D3
python get_pseudo_labels.py --source_domain D2 --target_domain D1
python get_pseudo_labels.py --source_domain D2 --target_domain D3
python get_pseudo_labels.py --source_domain D3 --target_domain D1
python get_pseudo_labels.py --source_domain D3 --target_domain D2
python train_recognizer.py --source_domain D1 --target_domain D2
python train_recognizer.py --source_domain D1 --target_domain D3
python train_recognizer.py --source_domain D2 --target_domain D1
python train_recognizer.py --source_domain D2 --target_domain D3
python train_recognizer.py --source_domain D3 --target_domain D1
python train_recognizer.py --source_domain D3 --target_domain D2
python test_recognizer.py --source_domain D1 --target_domain D2
python test_recognizer.py --source_domain D1 --target_domain D3
python test_recognizer.py --source_domain D2 --target_domain D1
python test_recognizer.py --source_domain D2 --target_domain D3
python test_recognizer.py --source_domain D3 --target_domain D1
python test_recognizer.py --source_domain D3 --target_domain D2
