#!/usr/bin/env bash
# grab the imagenet vqgan (f16, 16384 codes) the notebooks use.
# the checkpoint is about a gigabyte, go make tea.
set -e
mkdir -p checkpoints
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml \
  'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt \
  'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
