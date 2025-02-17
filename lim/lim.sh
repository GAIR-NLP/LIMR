#!/bin/bash
python filter_prompts.py \
    --train_samples_path ./data/train \
    --original_prompts_path ./data/prompts/math8k.json \
    --output_path ./filtered_prompts.json \
    --steps_per_epoch 8 \
    --max_epochs 21 \
    --similarity_threshold 0.6