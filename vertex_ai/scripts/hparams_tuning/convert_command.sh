#!/bin/sh
# This file converts command line arguments from argparser format to hydra format
python3  \
    vertex_ai/scripts/hparams_tuning_job/train_hparams_tuning.py \
    $(echo $@ | sed -r 's/--([^= ]*)[= ]([^ ]*)/\1=\2/g')
# If the above command fails, keep it running for 5 minutes so that it can be accessed after the failure
# https://cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell?hl=ja#enable
if [ $? -ne 0 ]; then
    sleep 300
    exit 1
else
    exit 0
fi
