rm -rf /tmp/caicloud-dist-tf

export TF_MAX_STEPS=3000
export TF_SAVE_CHECKPOINTS_SECS=60
export TF_SAVE_SUMMARIES_STEPS=1000
python train.py
