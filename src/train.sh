#!/bin/bash

TEST_DATASET=("ShanghaiA" "ShanghaiB" "qnrf")
DATSET_INDEX=(1 0 2)
MODE=("train" "test")
AUG=("aug" "noaug")

for i in "${DATSET_INDEX[@]}"
do
    for a in "${AUG[@]}"
    do
        TEST_DATASET_NAME="${TEST_DATASET[$i]}"
        AUGMENT=""
        if [ "$a" = "aug" ]; then
            AUGMENT="--aug"
        fi
        
    
        DESC="train_${i}"
        TEST_DATASET_PATH="npydata/${TEST_DATASET_NAME}_test.npy" 

        CUDA_VISIBLE_DEVICES=1 python main.py \
            --seed 2 \
            --name crowd \
            --dataset crowd \
            --num_labeled 10000 \
            --expand_labels \
            --total_steps 10000 \
            --eval_step 1000 \
            --randaug 2 16 \
            --batch_size 4 \
            --teacher_lr 1e-5 \
            --student_lr 1e-5 \
            --weight_decay 5e-4 \
            --ema 0.995 \
            --nesterov \
            --mu 1 \
            --temperature 0.7 \
            --threshold 50 \
            --lambda_u 8 \
            --warmup_steps 150 \
            --uda_steps 150 \
            --student_wait_steps 100 \
            --teacher_dropout 0.2 \
            --student_dropout 0.2 \
            --finetune_epochs 100 \
            --finetune_batch_size 16 \
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
            --pretrained \
            --dataset_index "${i}" \
            --use_wandb \
            $AUGMENT \

        DESC="test_${i}"
        TEST_DATASET_PATH="npydata/${TEST_DATASET_NAME}_test.npy" 

        CUDA_VISIBLE_DEVICES=1 python main.py \
            --seed 2 \
            --name crowd \
            --dataset crowd \
            --num_labeled 10000 \
            --expand_labels \
            --total_steps 10000 \
            --eval_step 1000 \
            --randaug 2 16 \
            --batch_size 4 \
            --teacher_lr 1e-4 \
            --student_lr 1e-4 \
            --weight_decay 5e-4 \
            --ema 0.995 \
            --nesterov \
            --mu 1 \
            --temperature 0.7 \
            --threshold 50\
            --lambda_u 8 \
            --warmup_steps 150 \
            --uda_steps 150 \
            --student_wait_steps 100 \
            --teacher_dropout 0.2 \
            --student_dropout 0.2 \
            --finetune_epochs 200 \
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
            --dataset_index "${i}" \
            --use_wandb \
            --evaluate \
            
    done
done
