#!/bin/bash

TEST_DATASET=("ShanghaiA" "ShanghaiB" "qnrf")
DATSET_INDEX=(0 1 2)
MODE=("train" "test")

for i in "${DATSET_INDEX[@]}"
do
    TEST_DATASET_NAME="${TEST_DATASET[$i]}"
    
    for m in "${MODE[@]}"
    do
        DESC="${m}_${i}"
        TEST_DATASET_PATH="npydata/${TEST_DATASET_NAME}_${m}.npy" 

        CUDA_VISIBLE_DEVICES=1 python main.py \
            --seed 2 \
            --name crowd \
            --dataset crowd \
            --num_labeled 10000 \
            --expand_labels \
            --total_steps 50000 \
            --eval_step 1000 \
            --randaug 2 16 \
            --batch_size 2 \
            --teacher_lr 1e-4 \
            --student_lr 1e-4 \
            --weight_decay 5e-4 \
            --ema 0.995 \
            --nesterov \
            --mu 2 \
            --temperature 0.7 \
            --threshold 0.6 \
            --lambda_u 8 \
            --warmup_steps 150 \
            --uda_steps 150 \
            --student_wait_steps 100 \
            --teacher_dropout 0.2 \
            --student_dropout 0.2 \
            --finetune_epochs 625 \
            --finetune_batch_size 16 \
            --finetune_lr 1e-4 \
            --finetune_weight_decay 0 \
            --finetune_momentum 0.9 \
            --home "/work/wycho/project/2023bigdataproject/src/" \
            --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
            --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
            --train_qnrf_data "npydata/qnrf_train.npy" \
            --test_dataset "${TEST_DATASET_PATH}" \
            --description "${DESC}" \
            --do_crop \
            --use_lr_scheduler \
            --pretrained \
            --dataset_index "${DATSET_INDEX}" \
            --use_wandb \
            --amp
            $([[ $m == "test" ]] && echo "--evaluate")
    done
done
