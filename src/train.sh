#python -m torch.distributed.launch --nproc_per_node 2 

# TEST_DATASET=("ShanghaiA" "ShanghaiB" "qnrf")
# DATSET_INDEX=(-1 0 1 2)
# MODE=("tain" "test")

TEST_DATASET=("ShanghaiB")
DATSET_INDEX=(1)
MODE=("tain")

for i in $DATSET_INDEX
do
    for d in $TEST_DATASET
    do
        for m in $MODE
        do
            DESC="${d}_${m}"
            TEST_DATESET_PATH="npydata/${d}_test.npy" 

            CUDA_VISIBLE_DEVICES=1 python main.py \
                --seed 2 \
                --name crowd \
                --dataset crowd \
                --num_labeled 10000 \
                --expand_labels \
                --total_steps 300000 \
                --eval_step 1000 \
                --randaug 2 16 \
                --batch_size 4 \
                --teacher_lr 1e-5\
                --student_lr 1e-5 \
                --weight_decay 5e-4 \
                --ema 0.995 \
                --nesterov \
                --mu 1 \
                --temperature 0.7 \
                --threshold 0.6 \
                --lambda_u 8 \
                --warmup_steps 5000 \
                --uda_steps 5000 \
                --student_wait_steps 3000 \
                --teacher_dropout 0.2 \
                --student_dropout 0.2 \
                --finetune_epochs 625 \
                --finetune_batch_size 8 \
                --finetune_lr 1e-5 \
                --finetune_weight_decay 0 \
                --finetune_momentum 0.9 \
                --home "/work/wycho/project/2023bigdataproject/src/" \
                --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
                --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
                --train_qnrf_data "npydata/qnrf_train.npy" \
                --test_dataset ${TEST_DATESET_PATH} \
                --description ${DESC} \
                --do_crop \
                --use_lr_scheduler \
                --accumulation_steps 1 \
                --pretrained \
                --dataset_index ${DATSET_INDEX}\
                --use_wandb \
                $([[ $m == "test" ]] && echo "--evaludate")
        done
    done

done
    





    
