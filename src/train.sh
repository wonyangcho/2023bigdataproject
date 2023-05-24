#python -m torch.distributed.launch --nproc_per_node 2 
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
    --teacher_lr 0.01\
    --student_lr 0.01 \
    --weight_decay 5e-4 \
    --nesterov \
    --mu 1 \
    --temperature 0.7 \
    --threshold 0.6 \
    --lambda_u 8 \
    --warmup_steps 0 \
    --uda_steps 5000 \
    --student_wait_steps 0 \
    --teacher_dropout 0.2 \
    --student_dropout 0.2 \
    --finetune_epochs 250 \
    --finetune_batch_size 4 \
    --finetune_lr 3e-5 \
    --finetune_weight_decay 0 \
    --finetune_momentum 0.9 \
    --home "/work/wycho/2023bigdataproject/src/" \
    --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
    --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
    --train_qnrf_data "npydata/qnrf_train.npy" \
    --test_dataset "npydata/ShanghaiA_test.npy" \
    --description "sch" \
    --use_wandb \
    --do_crop \
    --use_lr_scheduler \
    --amp

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
#     --teacher_lr 0.00001\
#     --student_lr 0.00001 \
#     --weight_decay 5e-4 \
#     --nesterov \
#     --mu 1 \
#     --temperature 0.7 \
#     --threshold 0.6 \
#     --lambda_u 8 \
#     --warmup_steps 0 \
#     --uda_steps 5000 \
#     --student_wait_steps 0 \
#     --teacher_dropout 0.2 \
#     --student_dropout 0.2 \
#     --finetune_epochs 250 \
#     --finetune_batch_size 4 \
#     --finetune_lr 3e-5 \
#     --finetune_weight_decay 0 \
#     --finetune_momentum 0.9 \
#     --home "/work/wycho/2023bigdataproject/src/" \
#     --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
#     --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
#     --train_qnrf_data "npydata/qnrf_train.npy" \
#     --test_dataset "npydata/ShanghaiA_test.npy" \
#     --description "sch_ShanghaiA_test" \
#     --use_wandb \
#     --do_crop \
#     --amp \
#     --evaluate

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
#     --teacher_lr 0.00001\
#     --student_lr 0.00001 \
#     --weight_decay 5e-4 \
#     --nesterov \
#     --mu 1 \
#     --temperature 0.7 \
#     --threshold 0.6 \
#     --lambda_u 8 \
#     --warmup_steps 0 \
#     --uda_steps 5000 \
#     --student_wait_steps 0 \
#     --teacher_dropout 0.2 \
#     --student_dropout 0.2 \
#     --finetune_epochs 250 \
#     --finetune_batch_size 4 \
#     --finetune_lr 3e-5 \
#     --finetune_weight_decay 0 \
#     --finetune_momentum 0.9 \
#     --home "/work/wycho/2023bigdataproject/src/" \
#     --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
#     --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
#     --train_qnrf_data "npydata/qnrf_train.npy" \
#     --test_dataset "npydata/ShanghaiB_test.npy" \
#     --description "sch_ShanghaiB_test" \
#     --use_wandb \
#     --do_crop \
#     --amp \
#     --evaluate

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
#     --teacher_lr 0.00001\
#     --student_lr 0.00001 \
#     --weight_decay 5e-4 \
#     --nesterov \
#     --mu 1 \
#     --temperature 0.7 \
#     --threshold 0.6 \
#     --lambda_u 8 \
#     --warmup_steps 0 \
#     --uda_steps 5000 \
#     --student_wait_steps 0 \
#     --teacher_dropout 0.2 \
#     --student_dropout 0.2 \
#     --finetune_epochs 250 \
#     --finetune_batch_size 4 \
#     --finetune_lr 3e-5 \
#     --finetune_weight_decay 0 \
#     --finetune_momentum 0.9 \
#     --home "/work/wycho/2023bigdataproject/src/" \
#     --train_ShanghaiA_data "npydata/ShanghaiA_train.npy" \
#     --train_ShanghaiB_data "npydata/ShanghaiB_train.npy" \
#     --train_qnrf_data "npydata/qnrf_train.npy" \
#     --test_dataset "npydata/qnrf_test.npy" \
#     --description "sch_qnrf_test" \
#     --use_wandb \
#     --do_crop \
#     --amp \
#     --evaluate
    
