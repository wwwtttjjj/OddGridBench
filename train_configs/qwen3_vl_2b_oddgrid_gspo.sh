#!/bin/bash

set -x

MODEL_PATH=/data/wengtengjin/models/Qwen3-VL-2B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/data/wengtengjin/colorsense/dapo_rl_train.jsonl \
    data.val_files=/data/wengtengjin/colorsense/dapo_rl_val.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.loss_type=gspo_token \
    worker.actor.loss_avg_mode=seq \
    worker.actor.clip_ratio_low=3e-4 \
    worker.actor.clip_ratio_high=4e-4 \
    algorithm.disable_kl=True \
    trainer.experiment_name=qwen3_vl_2b_oddgrid_gspo \
    trainer.n_gpus_per_node=4\
    worker.actor.global_batch_size=256\
    worker.actor.micro_batch_size_per_device_for_update=4\
    worker.rollout.tensor_parallel_size=1\
    trainer.max_steps=100\
