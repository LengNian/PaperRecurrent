from models.loca import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser
from utils.losses import ObjectNormalizedL2Loss

from time import perf_counter
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def train(args):
    # 判断是不是SLURM环境
    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        # multi-gpu
        # world_size = int(os.environ['WORLD_SIZE'])
        # rank = int(os.environ['RANK'])
        # gpu = int(os.environ['LOCAL_RANK'])

         #single-gpu
        world_size = 1
        rank = 0
        gpu = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"

        print("Running on Linux", world_size, rank, gpu)

    # 指定当前进程能用到的GPU卡的名称,传入索引
    torch.cuda.set_device(gpu)

    device = torch.device(gpu)

    # nccl是gpu之间的一种通信方式
    # world_size：当前节点上有多少张GPU卡
    # rank: rank指用当前前程在哪个GPU卡(args.lovcal_rank)
    # 也可以通过环境变量接受

    # 初始化进程组
    # multi-gpu
    # dist.init_process_group(
    #     backend='nccl', init_method='env://',
    #     world_size=world_size, rank=rank
    # )

    # single-gpu
    dist.init_process_group(
        backend='gloo', init_method='env://',
        world_size=world_size, rank=rank
    )


    # 将模型包裹,拷贝到gpu上
    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

    backbone_params = dict()

    non_backbone_params = dict()

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {'params': non_backbone_params.values()},
            {'params': backbone_params.values(), 'lr': args.backbone_lr}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)

    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_val_ae']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best = 10000000000000

    criterion = ObjectNormalizedL2Loss()

    train = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot
    )
    val = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p
    )

    # Dataloader中给了sampler就不需要shuffle了,他们是互斥的.
    train_loader = DataLoader(
        train,
        # 把样本分配到不同的GPU上,随机分配
        # DistributedSampler只需要传入一个dataset
        sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val,
        sampler=DistributedSampler(val),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers
    )
    print("Training...")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if rank == 0:
            start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        aux_train_loss = torch.tensor(0.0).to(device)
        aux_val_loss = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)

        # 调用set_epoch使数据充分打乱,如果不调用set_epoch中的epoch和seed都是0,打乱顺序就不会发生变化
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for img, bboxes, density_map in train_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)

            optimizer.zero_grad()
            # img: [1, 3, 512, 512]
            # bboxes: [1, 3, 4]
            # density_map: [1, 1, 512, 512]
            # model是由于build_model组成的，build_model返回一个LOCA，LOCA里包含了Backbone，OPE，regressionhead
            # 根据LOCA的forward(self, x, bboxes), 所以这里需要传入的是img和bboxes
            # out: [1, 1, 512, 512] aux_out: [2, 1, 1, 512, 512]
            out, aux_out = model(img, bboxes)

            # obtain the number of objects in batch
            with torch.no_grad():
                num_objects = density_map.sum()
                # print("++++", type(num_objects))
                # print("----", type([num_objects]))

                # dist.all_reduce_multigpu([num_objects])
                dist.all_reduce(num_objects)

            main_loss = criterion(out, density_map, num_objects)
            aux_loss = sum([
                args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
            ])
            loss = main_loss + aux_loss
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss += main_loss * img.size(0)
            aux_train_loss += aux_loss * img.size(0)
            train_ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()

        print("Evaling...")
        model.eval()
        with torch.no_grad():
            for img, bboxes, density_map in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)
                out, aux_out = model(img, bboxes)
                with torch.no_grad():
                    num_objects = density_map.sum()
                    # dist.all_reduce_multigpu([num_objects])
                    dist.all_reduce(num_objects)

                main_loss = criterion(out, density_map, num_objects)
                aux_loss = sum([
                    args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
                ])
                loss = main_loss + aux_loss

                val_loss += main_loss * img.size(0)
                aux_val_loss += aux_loss * img.size(0)
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()

        # dist.all_reduce_multigpu([train_loss])
        # dist.all_reduce_multigpu([val_loss])
        # dist.all_reduce_multigpu([aux_train_loss])
        # dist.all_reduce_multigpu([aux_val_loss])
        # dist.all_reduce_multigpu([train_ae])
        # dist.all_reduce_multigpu([val_ae])

        # print("+++", type(train_loss), type(val_loss), type(aux_train_loss), type(aux_val_loss), type(train_ae), type(val_ae))
        # print("---", type([train_loss]), type([val_loss]), type([aux_train_loss]), type([aux_val_loss]), type([train_ae]), type([val_ae]))

        dist.all_reduce(train_loss)
        dist.all_reduce(val_loss)
        dist.all_reduce(aux_train_loss)
        dist.all_reduce(aux_val_loss)
        dist.all_reduce(train_ae)
        dist.all_reduce(val_ae)


        scheduler.step()

        # 只在gpu0上进行保存
        if rank == 0:
            end = perf_counter()
            best_epoch = False
            if val_ae.item() / len(val) < best:
                best = val_ae.item() / len(val)
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_ae': val_ae.item() / len(val)
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.model_name}.pt')
                )
                best_epoch = True

            print(
                f"Epoch: {epoch}",
                f"Train loss: {train_loss.item():.3f}",
                f"Aux train loss: {aux_train_loss.item():.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Aux val loss: {aux_val_loss.item():.3f}",
                f"Train MAE: {train_ae.item() / len(train):.3f}",
                f"Val MAE: {val_ae.item() / len(val):.3f}",
                f"Epoch time: {end - start:.3f} seconds",
                'best' if best_epoch else ''
            )

    dist.destroy_process_group()


if __name__ == '__main__':
    # 训练时,要传入--nproc_per_node=GPU数量,构建不同的进程
    # 然后 torch.distributed.launch向代码中传入local_rank
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    train(args)
