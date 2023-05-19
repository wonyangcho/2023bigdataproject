import pdb
from logger import web_logger
from utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict)
from models.trans_crowd import base_patch16_384_token, base_patch16_384_gap
from data import DATASET_GETTERS
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
from torch.nn import functional as F
from torch import nn
from torch.cuda import amp
import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data_path', default='./data',
                    type=str, help='data path')
parser.add_argument('--save_path', default='./checkpoint',
                    type=str, help='save path')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'crowd'], help='dataset name')
parser.add_argument('--num_labeled', type=int, default=4000,
                    help='number of labeled data')
parser.add_argument("--expand_labels", action="store_true",
                    help="expand labels to fit eval steps")
parser.add_argument('--total_steps', default=300000,
                    type=int, help='number of total steps to run')
parser.add_argument('--eval_step', default=1000, type=int,
                    help='number of eval steps to run')
parser.add_argument('--start_step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
# parser.add_argument('--num-classes', default=10
#                     type=int, help='number of classes')
parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--batch_size', default=64,
                    type=int, help='train batch size')
parser.add_argument('--teacher_dropout', default=0,
                    type=float, help='dropout on last dense layer')
parser.add_argument('--student_dropout', default=0,
                    type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.01,
                    type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.01,
                    type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight_decay', default=0,
                    type=float, help='train weight decay')
# parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup_steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student_wait_steps', default=0,
                    type=int, help='warmup steps')
# parser.add_argument('--grad-clip', default=1e9, type=float,
#                     help='gradient norm clipping')
parser.add_argument('--grad_clip', default=0., type=float,
                    help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str,
                    help='path to checkpoint')
parser.add_argument('--evaluate', action='store_true',
                    help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
parser.add_argument('--finetune_epochs', default=625,
                    type=int, help='finetune epochs')
parser.add_argument('--finetune_batch_size', default=512,
                    type=int, help='finetune batch size')
parser.add_argument('--finetune_lr', default=3e-5,
                    type=float, help='finetune learning late')
parser.add_argument('--finetune_weight_decay', default=0,
                    type=float, help='finetune weight decay')
parser.add_argument('--finetune_momentum', default=0.9,
                    type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
# parser.add_argument('--label-smoothing', default=0,
#                     type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=7, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95,
                    type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float,
                    help='pseudo label temperature')
parser.add_argument('--lambda_u', default=1, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--uda_steps', default=1, type=float,
                    help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int,
                    help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true",
                    help="use 16-bit (mixed) precision")
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")


parser.add_argument('--home', default="", type=str, help='home path')
parser.add_argument('--train_l_data', default="", type=str,
                    help='labeld data file full path')
parser.add_argument('--train_ul_data', default="", type=str,
                    help='unlabeld data file full path')
parser.add_argument('--test_l_data', default="", type=str,
                    help='test data file full path')

# wandb
parser.add_argument("--use_wandb",  action="store_true", help="use wandb")
parser.add_argument(
    "--project_name",  default='2023BigDataProject', type=str, help='project name')
parser.add_argument("--description",  default='initial test',
                    type=str, help='experiment description')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    args = parser.parse_args()

    args.best_loss = float('inf')

    if args.local_rank != -1:
        args.gpu = args.local_rank
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
    else:
        args.gpu = 0
        args.world_size = 1

    if args.local_rank in [-1, 0]:
        args.local_time = f"{time.localtime().tm_mon:02d}{time.localtime().tm_mday:02d}{time.localtime().tm_hour:02d}{time.localtime().tm_min:02d}{time.localtime().tm_sec:02d}"

    args.device = torch.device('cuda', args.gpu)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"results/{args.name}")
#         wandb.init(name=args.name, project='MPL', config=args)

    if args.seed is not None:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset, finetune_dataset = DATASET_GETTERS[args.dataset](
        args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        drop_last=True)

    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=1,
                             num_workers=args.workers)

    teacher_model = base_patch16_384_token(pretrained=True)
    student_model = base_patch16_384_token(pretrained=True)

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    teacher_model = teacher_model.to(args.device)
    student_model = student_model.to(args.device)

    criterion = nn.L1Loss(size_average=False).cuda()

    t_optimizer = torch.optim.Adam(
        [  #
            {'params': teacher_model.parameters(), 'lr': args.teacher_lr},
        ], lr=args.teacher_lr, weight_decay=args.weight_decay)

    s_optimizer = torch.optim.Adam(
        [  #
            {'params': teacher_model.parameters(), 'lr': args.teacher_lr},
        ], lr=args.teacher_lr, weight_decay=args.weight_decay)

    t_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        t_optimizer, milestones=[300], gamma=0.1, last_epoch=-1)

    s_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        t_optimizer, milestones=[300], gamma=0.1, last_epoch=-1)

    train(args, labeled_loader, unlabeled_loader, test_loader,
          teacher_model, student_model,
          t_optimizer, s_optimizer, t_scheduler, s_scheduler, criterion)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train(args, labeled_loader, unlabeled_loader, test_loader,
          teacher_model, student_model,
          t_optimizer, s_optimizer, t_scheduler, s_scheduler, criterion):

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    for step in range(args.start_step, args.total_steps):

        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step),
                        disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()

        teacher_model.train()
        student_model.train()

        end = time.time()

        labeled_iter = iter(labeled_loader)
        images_l, targets = next(labeled_iter)
        unlabeled_iter = iter(unlabeled_loader)
        (images_uw, images_us), _ = next(unlabeled_iter)

        data_time.update(time.time() - end)

        images_l = images_l.to(args.device)
        images_uw = images_uw.to(args.device)
        images_us = images_us.to(args.device)
        targets = targets.to(args.device)

        with amp.autocast(enabled=args.amp):
            batch_size = images_l.shape[0]
            t_images = torch.cat((images_l, images_uw, images_us))
            t_logits = teacher_model(t_images)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits

            t_loss_l = criterion(t_logits_l, targets)
            t_loss_u = criterion(t_logits_us, t_logits_uw.detach())

            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_images = torch.cat((images_l, images_us))

            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]

            del s_logits

            s_loss_l_old = criterion(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, t_logits_uw.detach())

        s_loss.backward()
        s_optimizer.step()

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(images_l)

            s_loss_l_new = criterion(s_logits_l.detach(), targets)
            dot_product = s_loss_l_new - s_loss_l_old

            t_loss_mpl = dot_product * \
                criterion(t_logits_us, t_logits_uw.detach())
            t_loss = t_loss_uda + t_loss_mpl

        t_loss.backward()
        t_optimizer.step()

        t_scheduler.step()
        s_scheduler.step()

        t_optimizer.zero_grad()
        s_optimizer.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)

        s_losses.update(s_loss.item())

        t_losses.update(t_loss.mean().item())
        t_losses_l.update(t_loss_l.item())

        t_losses_u.update(t_loss_u.mean().item())
        t_losses_mpl.update(t_loss_mpl.item())

        batch_time.update(time.time() - end)
        end = time.time()

        batch_time.update(time.time() - end)

        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}.  ")
        pbar.update()

        # print(
        #     f"Epoch: {step}\t S_Loss {s_losses.avg} T_Loss {t_losses.avg}")

        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                test_loss = validate(test_loader, student_model, args)
                end2 = time.time()

                is_best = test_loss < args.best_loss
                args.best_loss = min(test_loss, args.best_loss)

                print(f" * best MAE {args.best_loss:.3f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    # 'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_loss': args.best_loss,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    # 'teacher_scaler': t_scaler.state_dict(),
                    # 'student_scaler': s_scaler.state_dict(),
                }, is_best)

                web_logger.log(args,
                               {"train/1.s_loss": s_losses.avg})
                web_logger.log(args,
                               {"train/2.t_loss": t_losses.avg})
                web_logger.log(args,
                               {"train/3.t_labeled": t_losses_l.avg})
                web_logger.log(args,
                               {"train/4.t_unlabeled": t_losses_u.avg})
                web_logger.log(args,
                               {"train/5.t_mpl": t_losses_mpl.avg})


def validate(data_loader, model, args):
    print('begin test')
    batch_size = 1
    test_loader = data_loader

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    for i, (img, gt_count) in enumerate(test_loader):

        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            out1 = model(img)
            count = torch.sum(out1).item()

        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 15 == 0:
            print('Gt {gt:.2f} Pred {pred}'.format(
                gt=gt_count, pred=count))

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    print(' \n* MAE {mae:.3f}\n'.format(mae=mae),
          '* MSE {mse:.3f}'.format(mse=mse))

    web_logger.log(args,
                   {"test/MAE": mae})
    web_logger.log(args,
                   {"test/MSE": mse})

    return mae


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
