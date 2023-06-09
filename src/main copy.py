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
# import wandb

# from models.models import WideResNet, ModelEMA


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


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def check_nan_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"============ NaN parameter found in {name}")


def train_loop(args, labeled_loader, unlabeled_loader, test_loader, finetune_dataset,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):

    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"   Total steps = {args.total_steps}")

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

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
            # mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            images_l, targets = next(labeled_iter)
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            images_l, targets = next(labeled_iter)

        targets = targets.unsqueeze(1)

        try:
            (images_uw, images_us), _ = next(unlabeled_iter)
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
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

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(
                student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        # if args.ema > 0:
        #     avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(images_l)

            s_loss_l_new = criterion(s_logits_l.detach(), targets)
            dot_product = s_loss_l_new - s_loss_l_old

            t_loss_mpl = dot_product * \
                criterion(t_logits_us, t_logits_uw.detach())
            t_loss = t_loss_uda + t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(
                teacher_model.parameters(), args.grad_clip)

        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
            # mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())

        t_losses.update(t_loss.mean().item())
        t_losses_l.update(t_loss_l.item())

        t_losses_u.update(t_loss_u.mean().item())
        t_losses_mpl.update(t_loss_mpl.item())
        # mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}.  ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)
            web_logger.log(args, {"lr": get_lr(s_optimizer)})
#             wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step // args.eval_step

        # print(f"{step+1} {args.eval_step} {(step + 1) % args.eval_step}")

        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:

                print("==================logging")

                args.writer.add_scalar(
                    "train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar(
                    "train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar(
                    "train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar(
                    "train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar(
                    "train/5.t_mpl", t_losses_mpl.avg, args.num_eval)

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

                test_model = avg_student_model if avg_student_model is not None else student_model
                test_loss = evaluate(
                    args, test_loader, test_model, criterion)

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar(
                    "test/loss", test_loss, args.num_eval)

                web_logger.log(args, {"test/loss": test_loss})
                web_logger.log(args,
                               {"test/loss": test_loss})

                is_best = test_loss < args.best_loss
                if is_best:
                    args.best_loss = test_loss

                logger.info(f"loss: {test_loss:.2f}")
                logger.info(f"best_loss: {args.best_loss:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_loss': args.best_loss,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)

    if args.local_rank in [-1, 0]:
        args.writer.add_scalar("result/test_loss", args.best_loss)
        web_logger.log(args, {"result/test_loss": args.best_loss})
#         wandb.log({"result/test_acc@1": args.best_top1})

    # finetune
    del t_scaler, t_scheduler, t_optimizer, teacher_model, labeled_loader, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])
    finetune(args, finetune_dataset, test_loader, student_model, criterion)
    return


# ref 논문 참조할것
def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_iter):

            if len(images.shape) == 5:
                images = images.squeeze(0)
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            images = images.float()
            targets = targets.float()

            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                outputs = model(images)
                count = torch.sum(outputs).item()

                loss = criterion(torch.tensor(
                    count).to(targets.device), targets)

            # acc1, acc5 = accuracy(outputs, targets, (1, 5))  # wycho
            losses.update(loss.item(), batch_size)
            # pred.update(pred[0], batch_size)

            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")

        test_iter.close()
        return losses.avg


def finetune(args, finetune_dataset, test_loader, model, criterion):
    model.drop = nn.Identity()
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        finetune_dataset,
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        pin_memory=True)
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=args.finetune_lr,
    #                       momentum=args.finetune_momentum,
    #                       weight_decay=args.finetune_weight_decay,
    #                       nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(
        f"   Finetuning steps = {len(labeled_loader)*args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch + 624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(
            labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets) in enumerate(labeled_iter):

            images = images.float()
            targets = targets.float()

            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                model.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.world_size > 1:
                loss = reduce_tensor(loss.detach(), args.world_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss = evaluate(
                args, test_loader, model, criterion)
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)

            web_logger.log(args, {"finetune/train_loss": losses.avg})
            web_logger.log(args, {"finetune/test_loss": test_loss})

            is_best = test_loss < args.best_loss
            if is_best:
                args.best_loss = test_loss

            logger.info(f"loss: {test_loss:.2f}")
            logger.info(f"best_loss: {args.best_loss:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'best_loss': args.best_loss,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("result/finetune_loss", args.best_loss)
            web_logger.log(args, {"result/finetune_loss": args.best_loss})
#             wandb.log({"result/finetune_acc@1": args.best_top1})
    return


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

    if args.dataset == "cifar10":
        depth, widen_factor = 28, 2
    elif args.dataset == 'cifar100':
        depth, widen_factor = 28, 8
    else:
        depth, widen_factor = 28, 2  # wycho 수정해야함

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # teacher_model = WideResNet(num_classes=args.num_classes,
    #                            depth=depth,
    #                            widen_factor=widen_factor,
    #                            dropout=0,
    #                            dense_dropout=args.teacher_dropout)
    # student_model = WideResNet(num_classes=args.num_classes,
    #                            depth=depth,
    #                            widen_factor=widen_factor,
    #                            dropout=0,
    #                            dense_dropout=args.student_dropout)

    # if args['model_type'] == 'token':
    #     model = base_patch16_384_token(pretrained=True)
    # else:
    teacher_model = base_patch16_384_token(pretrained=True)
    student_model = base_patch16_384_token(pretrained=True)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    logger.info(
        f"Params: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")

    avg_student_model = None
    # if args.ema > 0:
    #     avg_student_model = ModelEMA(student_model, args.ema)

    criterion = create_loss_fn(args)

    # t_optimizer = optim.SGD(teacher_model.parameters(),
    #                         lr=args.teacher_lr,
    #                         momentum=args.momentum,
    #                         nesterov=args.nesterov)
    # s_optimizer = optim.SGD(student_model.parameters(),
    #                         lr=args.student_lr,
    #                         momentum=args.momentum,
    #                         nesterov=args.nesterov)

    t_optimizer = torch.optim.Adam(
        [  #
            {'params': teacher_model.parameters(), 'lr': args.teacher_lr},
        ], lr=args.teacher_lr, weight_decay=args.weight_decay)

    s_optimizer = torch.optim.Adam(
        [  #
            {'params': teacher_model.parameters(), 'lr': args.student_lr},
        ], lr=args.student_lr, weight_decay=args.weight_decay)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_loss = checkpoint['best_loss'].to(torch.device('cpu'))

            if not (args.evaluate or args.finetune):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(
                    teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(
                        avg_student_model, checkpoint['avg_state_dict'])

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(
                        student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(
                        student_model, checkpoint['student_state_dict'])

            logger.info(
                f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    if args.finetune:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
        del s_scaler, s_scheduler, s_optimizer
        finetune(args, finetune_dataset, test_loader, student_model, criterion)
        return

    if args.evaluate:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        evaluate(args, test_loader, student_model, criterion)
        return

    teacher_model = teacher_model.to(args.device)
    student_model = student_model.to(args.device)

    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(args, labeled_loader, unlabeled_loader, test_loader, finetune_dataset,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler)
    return


if __name__ == '__main__':
    main()
