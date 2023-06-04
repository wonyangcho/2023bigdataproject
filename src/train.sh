#python -m torch.distributed.launch --nproc_per_node 2 
CUDA_VISIBLE_DEVICES=1 python main.py \
    --seed 2 \
    --name crowd \
    --dataset crowd \
    --num_labeled 10000 \
    --expand_labels \
    --total_steps 300000 \
    --eval_step 1000 \
    --randaug 2 16 \
    --batch_size 8 \
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
    --test_dataset "npydata/ShanghaiA_test.npy" \
    --description "ShanghaiA_train" \
    --do_crop \
    --use_lr_scheduler \
    --accumulation_steps 1 \
    --pretrained \
    --amp \
    --dataset_index 0\
    --use_wandb \
    

# 

CUDA_VISIBLE_DEVICES=1 python main.py \
    --seed 2 \
    --name crowd \
    --dataset crowd \
    --num_labeled 10000 \
    --expand_labels \
    --total_steps 100000 \
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
    --warmup_steps 1000 \
    --uda_steps 5000 \
    --student_wait_steps 0 \
    --teacher_dropout 0.2 \
    --student_dropout 0.2 \
    --finetune_epochs 250 \
    --finetune_batch_size 4 \
    --finetune_lr 1e-5 \
    --finetune_weight_decay 0 \
    --finetune_momentum 0.9 \
    --home "/work/wycho/project/2023bigdataproject/src/" \
    --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
    --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
    --train_qnrf_data "npydata/qnrf_train.npy" \
    --test_dataset "npydata/ShanghaiA_test.npy" \
    --description "ShanghaiA_test" \
    --use_wandb \
    --do_crop \
    --evaluate \
    --dataset_index 0\


# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --seed 2 \
#     --name crowd \
#     --dataset crowd \
#     --num_labeled 10000 \
#     --expand_labels \
#     --total_steps 300000 \
#     --eval_step 1000 \
#     --randaug 2 16 \
#     --batch_size 8 \
#     --teacher_lr 1e-5\
#     --student_lr 1e-5 \
#     --weight_decay 5e-4 \
#     --ema 0.995 \
#     --nesterov \
#     --mu 1 \
#     --temperature 0.7 \
#     --threshold 0.6 \
#     --lambda_u 8 \
#     --warmup_steps 5000 \
#     --uda_steps 5000 \
#     --student_wait_steps 3000 \
#     --teacher_dropout 0.2 \
#     --student_dropout 0.2 \
#     --finetune_epochs 625 \
#     --finetune_batch_size 8 \
#     --finetune_lr 1e-5 \
#     --finetune_weight_decay 0 \
#     --finetune_momentum 0.9 \
#     --home "/work/wycho/project/2023bigdataproject/src/" \
#     --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
#     --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
#     --train_qnrf_data "npydata/qnrf_train.npy" \
#     --test_dataset "npydata/ShanghaiB_test.npy" \
#     --description "ShanghaiB_train" \
#     --do_crop \
#     --use_lr_scheduler \
#     --accumulation_steps 1 \
#     --pretrained \
#     --amp \
#     --dataset_index 1\
#     --use_wandb \

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --seed 2 \
#     --name crowd \
#     --dataset crowd \
#     --num_labeled 10000 \
#     --expand_labels \
#     --total_steps 100000 \
#     --eval_step 1000 \
#     --randaug 2 16 \
#     --batch_size 4 \
#     --teacher_lr 1e-5\
#     --student_lr 1e-5 \
#     --weight_decay 5e-4 \
#     --ema 0.995 \
#     --nesterov \
#     --mu 1 \
#     --temperature 0.7 \
#     --threshold 0.6 \
#     --lambda_u 8 \
#     --warmup_steps 1000 \
#     --uda_steps 5000 \
#     --student_wait_steps 0 \
#     --teacher_dropout 0.2 \
#     --student_dropout 0.2 \
#     --finetune_epochs 250 \
#     --finetune_batch_size 4 \
#     --finetune_lr 1e-5 \
#     --finetune_weight_decay 0 \
#     --finetune_momentum 0.9 \
#     --home "/work/wycho/project/2023bigdataproject/src/" \
#     --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
#     --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
#     --train_qnrf_data "npydata/qnrf_train.npy" \
#     --test_dataset "npydata/ShanghaiB_test.npy" \
#     --description "ShanghaiB_test" \
#     --use_wandb \
#     --do_crop \
#     --dataset_index 1\
#     --evaluate


# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --seed 2 \
#     --name crowd \
#     --dataset crowd \
#     --num_labeled 10000 \
#     --expand_labels \
#     --total_steps 300000 \
#     --eval_step 1000 \
#     --randaug 2 16 \
#     --batch_size 8 \
#     --teacher_lr 1e-5\
#     --student_lr 1e-5 \
#     --weight_decay 5e-4 \
#     --ema 0.995 \
#     --nesterov \
#     --mu 1 \
#     --temperature 0.7 \
#     --threshold 0.6 \
#     --lambda_u 8 \
#     --warmup_steps 5000 \
#     --uda_steps 5000 \
#     --student_wait_steps 3000 \
#     --teacher_dropout 0.2 \
#     --student_dropout 0.2 \
#     --finetune_epochs 625 \
#     --finetune_batch_size 8 \
#     --finetune_lr 1e-5 \
#     --finetune_weight_decay 0 \
#     --finetune_momentum 0.9 \
#     --home "/work/wycho/project/2023bigdataproject/src/" \
#     --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
#     --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
#     --train_qnrf_data "npydata/qnrf_train.npy" \
#     --test_dataset "npydata/qnrf_test.npy" \
#     --description "Qnrf_train" \
#     --do_crop \
#     --use_lr_scheduler \
#     --accumulation_steps 1 \
#     --pretrained \
#     --amp \
#     --dataset_index 2\
#     --use_wandb \

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --seed 2 \
#     --name crowd \
#     --dataset crowd \
#     --num_labeled 10000 \
#     --expand_labels \
#     --total_steps 100000 \
#     --eval_step 1000 \
#     --randaug 2 16 \
#     --batch_size 4 \
#     --teacher_lr 1e-5\
#     --student_lr 1e-5 \
#     --weight_decay 5e-4 \
#     --ema 0.995 \
#     --nesterov \
#     --mu 1 \
#     --temperature 0.7 \
#     --threshold 0.6 \
#     --lambda_u 8 \
#     --warmup_steps 1000 \
#     --uda_steps 5000 \
#     --student_wait_steps 0 \
#     --teacher_dropout 0.2 \
#     --student_dropout 0.2 \
#     --finetune_epochs 250 \
#     --finetune_batch_size 4 \
#     --finetune_lr 1e-5 \
#     --finetune_weight_decay 0 \
#     --finetune_momentum 0.9 \
#     --home "/work/wycho/project/2023bigdataproject/src/" \
#     --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
#     --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
#     --train_qnrf_data "npydata/qnrf_train.npy" \
#     --test_dataset "npydata/qnrf_test.npy" \
#     --description "Qnrf_test" \
#     --use_wandb \
#     --do_crop \
#     --amp \
#     --dataset_index 2\
#     --evaluate
    
