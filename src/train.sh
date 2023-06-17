#!/bin/bash

TAG="$(date +%Y%m%d%H%M%S)"

TEST_DATASET=("ShanghaiA" "ShanghaiB" "qnrf")
DATSET_INDEX=(2 0 1)
MODE=("baseline" "train" "test")
#AUG=("noaug" "aug")
AUG=("noaug")


for a in "${AUG[@]}"
do
    for i in "${DATSET_INDEX[@]}"
    do

        for m in "${MODE[@]}"
        do
            TEST_DATASET_NAME="${TEST_DATASET[$i]}"
            EXP="${TAG}"
            AUGMENT=""
            if [ "${a}" = "aug" ]; then
                AUGMENT="--aug"
            fi

            EVAL=""
            if [ "${m}" = "test" ]; then
                EVAL="--evaluate"
            elif [ "${m}" = "baseline" ]; then
                EVAL="--baseline"
            fi
            
        
            DESC="${m}_${i}_${a}_fixmatch"
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
                --threshold 0.6 \
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
                --pretrained \
                --dataset_index "${i}" \
                --use_wandb \
                ${AUGMENT} \
                --tag "${EXP}" \
                ${EVAL} \

        
        done    
    done
done
