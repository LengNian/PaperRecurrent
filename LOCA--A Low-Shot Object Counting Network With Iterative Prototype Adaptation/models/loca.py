from xml.sax.handler import all_properties

from .backbone import Backbone
from .transformer import TransformerEncoder
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LOCA(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_ope_iterative_steps: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        backbone_name: str,
        swav_backbone: bool,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
    ):

        super(LOCA, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers

        self.backbone = Backbone(
            backbone_name, pretrained=True, dilation=False, reduction=reduction,
            swav=swav_backbone, requires_grad=train_backbone
        )

        # 这里的self.backbone.num_channels就是3584
        # emb_dim= 256
        # 这个input_proj应该是将特征图大小调整到d维用的
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, emb_dim, kernel_size=1
        )

        # 全局自注意力块的层数
        # num_encoder_layers = 3

        if num_encoder_layers > 0:
            # 这里使用Transformer的encoder
            # 使用了3层Transformer Encoder
            self.encoder = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )

        # OPE模块
        # num_ope_iterative_steps = 3
        # emb_dim = 256
        # num_heads = 8
        # num_objects 应该是提示用几个样本计数，默认是3，下面
        # kernel_dim 是论文中提到的s
        self.ope = OPEModule(
            num_ope_iterative_steps, emb_dim, kernel_dim, num_objects, num_heads,
            reduction, layer_norm_eps, mlp_factor, norm_first, activation, norm, zero_shot
        )

        # 回归头
        # reduction在回归头实际没有用，只是判断回归头中的卷积层的组成方式
        self.regression_head = DensityMapRegressor(emb_dim, reduction)

        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
            for _ in range(num_ope_iterative_steps - 1)
        ])

        self.pos_emb = PositionalEncodingsFixed(emb_dim)


    def forward(self, x, bboxes):
        # x: [1,3, 512, 512]
        # bboxes: [1, 3, 4]
        # 如果self.zero_shot为True, num_objects = bboxes.size(1)
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects

        # backbone
        # backbone_features: [1, 3584, 64, 64]
        backbone_features = self.backbone(x)

        # prepare the encoder input
        # src: [1, 256, 64, 64]
        src = self.input_proj(backbone_features)
        # bs: batch   c: 256  h = w = 64
        bs, c, h, w = src.size()

        # pos_emb: [4096, 1, 256]
        pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)

        # # pos_emb: [1, 256, 64, 64]
        # pos_emb = self.pos_emb(bs, h, w, src.device)
        # # pos_emb: [4096, 1, 256]
        # pos_emb = pos_emb.flatten(2).permute(2, 0, 1)

        # src: [4096, 1, 256]
        src = src.flatten(2).permute(2, 0, 1)

        # push through the encoder
        if self.num_encoder_layers > 0:
            # image_features: [4096, 1, 256]
            image_features = self.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)
        else:
            image_features = src

        # prepare OPE input
        # image_features.permute(1, 2, 0): [1, 256, 4096]
        # f_e: [1, 256, 64, 64]
        f_e = image_features.permute(1, 2, 0).reshape(-1, self.emb_dim, h, w)

        # all_prototypes [3, 27, 1, 256]
        all_prototypes = self.ope(f_e, pos_emb, bboxes)

        outputs = list()

        for i in range(all_prototypes.size(0)):
            # prototypes: [3, 27, 1, 256] --> [27, 1, 256] -- > [1, 27, 256] --> [1, 3, 3, 3, 256] --> [1, 3, 256, 3, 3] --> [768, 3, 3] --> [768, 1, 3, 3]
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]

            # 使用prototypes作为卷积核对特征图进行操作
            # 沿着 num_objects维度取最大值, 响应图的维度变为(bs, emb_dim, h, w)
            # f_e:[1, 256, 64, 64], 沿着dim=1堆叠三次变为[1, 768, 64, 64]
            # flatten, unsqueeze: [1, 768, 64, 64] --> [768, 64, 64] --> [1, 768, 64, 64]
            # F.conv2d(): [1, 768, 64, 64]
            # F.conv2d().view().max(dim=1)[0] [1, 768, 64, 64] --> [1, 3, 256, 64, 64] --> [1, 256, 64, 64]
            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)
            ).view(
                bs, num_objects, self.emb_dim, h, w
            ).max(dim=1)[0]

            # send through regression heads
            if i == all_prototypes.size(0) - 1:
                # predicted_dmaps: [1, 1, 512, 512]
                predicted_dmaps = self.regression_head(response_maps)
            else:
                # predicted_dmaps: [1, 1, 512, 512]
                predicted_dmaps = self.aux_heads[i](response_maps)

            # outputs.shape [3, 1, 1, 512, 512]
            outputs.append(predicted_dmaps)

        # 最后一个是输出密度图, 其余都是辅助的密度图
        return outputs[-1], outputs[:-1]


def build_model(args):

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    return LOCA(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
        num_ope_iterative_steps=args.num_ope_iterative_steps,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        swav_backbone=args.swav_backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        dropout=args.dropout,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=args.pre_norm,
        activation=nn.GELU,
        norm=True,
    )

