#!/bin/bash

TEST_DATASET=("ShanghaiA" "ShanghaiB" "qnrf")
DATSET_INDEX=(2 1 0)
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
            --total_steps 150000 \
            --eval_step 500 \
            --randaug 2 16 \
            --batch_size 4 \
            --teacher_lr 1e-5 \
            --student_lr 1e-5 \
            --weight_decay 5e-4 \
            --ema 0.995 \
            --nesterov \
            --mu 1 \
            --temperature 0.7 \
            --threshold 0.6 \
            --lambda_u 8 \
            --warmup_steps 2500 \
            --uda_steps 2500 \
            --student_wait_steps 1500 \
            --teacher_dropout 0.2 \
            --student_dropout 0.2 \
            --finetune_epochs 1000 \
            --finetune_batch_size 8 \
            --finetune_lr 1e-5 \
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
            --accumulation_steps 1 \
            --pretrained \
            --dataset_index "${DATSET_INDEX}" \
            --use_wandb \
            $([[ $m == "test" ]] && echo "--evaluate")
    done
done
