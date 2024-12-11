import argparse
import datetime
import json

import PIL.Image
import numpy as np
import os
import time
import random
from pathlib import Path
import math
import sys
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import Dataset
import wandb
wandb.init(mode='offline')
import timm

assert "0.4.5" <= timm.__version__ <= "0.4.9"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.FSC147 import transform_pre_train
import models_mae_noct


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/FSC147_384_v2', type=str,
                        help='dataset path')
    parser.add_argument('--anno_file', default='annotation_FSC147_384.json', type=str,
                        help='annotation json file')
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str,
                        help='data split json file')
    parser.add_argument('--im_dir', default='images_384_VarV2', type=str,
                        help='images directory')
    parser.add_argument('--gt_dir', default='gt_density_map_adaptive_384_VarV2', type=str,
                        help='ground truth directory')
    parser.add_argument('--output_dir', default='./data/out/pre_4_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./weights/mae_pretrain_vit_base_full.pth',  # mae_visualize_vit_base
                        help='resume from checkpoint')

    # Training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Logging parameters
    parser.add_argument('--log_dir', default='./logs/pre_4_dir',
                        help='path where to tensorboard log')
    parser.add_argument("--title", default="CounTR_pretraining", type=str)
    parser.add_argument("--wandb", default="counting", type=str)
    parser.add_argument("--team", default="wsense", type=str)
    parser.add_argument("--wandb_id", default=None, type=str)
    parser.add_argument('--do_aug', default=None)

    return parser


os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


class TrainData(Dataset):
    def __init__(self):
        self.img = data_split['train']
        random.shuffle(self.img)
        self.img_dir = im_dir
        self.gt_dir = gt_dir
        self.annotations = annotations
        self.TransformPreTrain = transform_pre_train(args)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]

        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(self.img_dir, im_id))
        image.load()
        density_path = self.gt_dir / (im_id.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')
        sample = {'image': image, 'lines_boxes': rects, 'gt_density': density}
        sample = self.TransformPreTrain(sample)
        # 返回transform后的图像
        return sample['image']


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # 训练阶段只运用到训练数据
    dataset_train = TrainData()

    # 是否多卡训练
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        # 当前进程或者设备在分布式训练环境中的全局排名, 单卡应该是0
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0:
        # TensorBoard日志的路径
        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None

        # 用wandb可视化
        if args.wandb is not None:
            wandb_run = wandb.init(
                config=args,
                resume="allow",
                project=args.wandb,
                name=args.title,
                entity=args.team,
                tags=["CounTR", "pretraining"],
                id=args.wandb_id,
            )
        else:
            wandb_run = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    # 在这里定义了norm_pix_loss
    model = models_mae_noct.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))

    # args.accum_iter 累计梯度更新的次数, 为了节省可以通过梯度累积的方式，在多个小批次（mini-batches）上累积梯度，模拟一个更大的批次大小
    # misc.get_world_size()  获取当前分布式训练环境中的设备总数
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    # 用于管理混合精度训练过程中的损失缩放，确保数值稳定性并自动处理缩放逻辑。
    loss_scaler = NativeScaler()

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # train one epoch
        model.train(True)
        # MetricLogger 用于记录和格式化训练过程中的各种指标, 添加学习率记录器
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        # 设置日志的打印频率
        print_freq = 20
        accum_iter = args.accum_iter

        optimizer.zero_grad()

        # 日志记录器初始化
        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))

        # 指定从配置中加载模型架构
        model_ = getattr(models_mae_noct, args.model)()

        for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
            # 将当前迭代步数和 epoch 转换为一个基于 1000 的尺度，便于与学习率和日志记录对齐。
            epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)

            # 每隔accum_iter次迭代调整一次学习率
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

            samples = samples.to(device, non_blocking=True)

            # 启用自动混合精度, 减少显存使用并提高速度
            # 看完model_mae_noct之后来看这里的前向传播
            with torch.cuda.amp.autocast():
                #sample's shape  8, 3, ,384, 384
                loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)

            loss_value = loss.item()

            if data_iter_step % 2000 == 0:
                # 将预测结果反转换成完整的图像(model_mae_noct中用到的内容)

                preds = model_.unpatchify(pred)
                preds = preds.float()
                # 重新排列维度, 并裁剪到[0, 1]范围
                preds = torch.einsum('nchw->nhwc', preds)
                preds = torch.clip(preds, 0, 1)

                # TensorBoard与WandB可视化
                if log_writer is not None:
                    log_writer.add_images('reconstruction', preds, int(epoch), dataformats='NHWC')

                if wandb_run is not None:
                    wandb_images = []
                    w_samples = torch.einsum('nchw->nhwc', samples.float()).clip(0, 1)
                    masks = F.interpolate(
                        mask.reshape(shape=(mask.shape[0], 1, int(mask.shape[1] ** .5), int(mask.shape[1] ** .5))),
                        size=(preds.shape[1], preds.shape[2]))
                    masks = torch.einsum('nchw->nhwc', masks.float())
                    combos = (w_samples + masks.repeat(1, 1, 1, 3)).clip(0, 1)
                    w_images = (torch.cat([w_samples, combos, preds], dim=2) * 255).detach().cpu()
                    print("w_images:", w_samples.shape, combos.shape, preds.shape, "-->", w_images.shape)

                    for i in range(w_images.shape[0]):
                        wi = w_images[i, :, :, :]
                        wandb_images += [wandb.Image(wi.numpy().astype(np.uint8),
                                                     caption=f"Prediction {i} at epoch {epoch}")]
                    wandb.log({f"reconstruction": wandb_images}, step=epoch_1000x, commit=False)

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # 将模型输出的loss在这里 管理混合精度中的损失缩放和梯度更新。
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            # 累积步数达到指定值后清零优化器梯度
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            # 日志更新和进程同步
            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if (data_iter_step + 1) % accum_iter == 0:
                if log_writer is not None:
                    """ We use epoch_1000x as the x-axis in tensorboard.
                    This calibrates different curves when batch size changes.
                    """
                    log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                    log_writer.add_scalar('lr', lr, epoch_1000x)
                if wandb_run is not None:
                    log = {"train/loss": loss_value_reduce, "train/lr": lr}
                    wandb.log(log, step=epoch_1000x, commit=True if data_iter_step == 0 else False)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # save train status and model
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, suffix=f"pretraining_{epoch}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    wandb.run.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # load data
    data_path = Path(args.data_path)
    print("data_path:", data_path)
    anno_file = data_path / args.anno_file
    data_split_file = data_path / args.data_split_file
    im_dir = data_path / args.im_dir
    gt_dir = data_path / args.gt_dir
    with open(anno_file) as f:
        annotations = json.load(f)
    with open(data_split_file) as f:
        data_split = json.load(f)


    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # print(im_dir)
    main(args)
