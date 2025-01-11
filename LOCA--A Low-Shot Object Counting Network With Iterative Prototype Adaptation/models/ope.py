from .mlp import MLP
from .positional_encoding import PositionalEncodingsFixed

import torch
from torch import nn

from torchvision.ops import roi_align

# 物体原型提取模块
class OPEModule(nn.Module):

    def __init__(
        self,
        num_iterative_steps: int,
        emb_dim: int,
        kernel_dim: int,
        num_objects: int,
        num_heads: int,
        reduction: int,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
    ):

        super(OPEModule, self).__init__()

        # 迭代适应模块迭代次数，default=3
        self.num_iterative_steps = num_iterative_steps
        self.zero_shot = zero_shot
        # 实际就是论文中提到的s
        self.kernel_dim = kernel_dim
        self.num_objects = num_objects
        self.emb_dim = emb_dim
        self.reduction = reduction

        # 创建迭代适应模块
        if num_iterative_steps > 0:
            self.iterative_adaptation = IterativeAdaptationModule(
                num_layers=num_iterative_steps, emb_dim=emb_dim, num_heads=num_heads,
                dropout=0, layer_norm_eps=layer_norm_eps,
                mlp_factor=mlp_factor, norm_first=norm_first,
                activation=activation, norm=norm,
                zero_shot=zero_shot
            )

        # 提取shape query
        # 少样本
        if not self.zero_shot:
            self.shape_or_objectness = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, self.kernel_dim**2 * emb_dim)
            )
        # 零样本
        else:
            # 如果是零样本不存在外观和形状查询，就创建一个可训练的参数去代替
            self.shape_or_objectness = nn.Parameter(
                torch.empty((self.num_objects, self.kernel_dim**2, emb_dim))
            )
            nn.init.normal_(self.shape_or_objectness)

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    # 前向传播时, 传入的是图像特征图, 位置编码和边界框
    def forward(self, f_e, pos_emb, bboxes):
       # f_e:[1, 256, 64, 64]
       # pos_emb: [4096, 1, 256]
       # bboxes: [1, 3, 4]
        bs, _, h, w = f_e.size()

        # 少样本情况下
        # extract the shape features or objectness
        if not self.zero_shot:
            # 将创建一个形状为(bboxes.size(0), bboxes.size(1), 2)的全0张量
            # box_hw: [1, 3, 2]
            box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
            # 获取对应维度的高和宽(就是每个batch下每个边界框的高和宽)
            box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
            box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]

            # shape_or_objectness = self.shape_or_objectness(box_hw).reshape(
            #     bs, -1, self.kernel_dim ** 2, self.emb_dim
            # ).flatten(1, 2).transpose(0, 1)

            # shape_or_objectness: [1, 3, 2304]
            shape_or_objectness = self.shape_or_objectness(box_hw)
            # shape_or_objectness: [1, 3, 2304] --> [1, 3, 9, 256] --> [1, 27, 256]  --> [27, 1, 256]
            shape_or_objectness = shape_or_objectness.reshape(bs, -1, self.kernel_dim ** 2, self.emb_dim).flatten(1, 2).transpose(0, 1)

        # 零样本情况下
        else:
            # 将self.shape_or_objectness扩展到(bs, -1, -1, -1) -1表示维度不变
            shape_or_objectness = self.shape_or_objectness.expand(
                bs, -1, -1, -1
            ).flatten(1, 2).transpose(0, 1)


        # if not zero shot add appearance
       # 少样本
        if not self.zero_shot:
            # reshape bboxes into the format suitable for roi_align
            # 先用torch.arange生成一系列编号(0, bs-1)代表每个batch
            # 然后重复self.num_objects次, 就是有几个边界框
            # 然后reshape调整为(bs*self.num_objects, 1)
            # 再把bboxes摊平为(bs*self.num_objects, 4)
            # 二者合并后为(bs*self.num_objects, 5) ----> 5是指[batch_index, x1, y1, x2, y2]
            # bboes: [1, 3, 4] --> [3, 5]
            bboxes = torch.cat([
                torch.arange(bs, requires_grad=False).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
                bboxes.flatten(0, 1),
            ], dim=1)

            # appearance query
            # output_size: roi池化以后的尺寸
            # spatial_scale: 空间缩放因子, 用于将边界框坐标从输入图像空间映射到特征图空间
            # aligned: 用于对齐的RoI Align操作, 确保特征提取更加精确
            # appearances经过roi [bs * num_objects, C, kernel_dim, kernel_dim]
            # 经过一些列操作，appearance的形状变为 ns^2*d
            # appearence: [27, 1, 256]
            appearance = roi_align(
                f_e,
                boxes=bboxes, output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction, aligned=True
            ).permute(0, 2, 3, 1).reshape(
                bs, self.num_objects * self.kernel_dim ** 2, -1
            ).transpose(0, 1)

        # 零样本
        # 零样本就没有外观查询
        else:
            appearance = None

        # self.pos_emb()返回值形状为[1, 256, 3, 3]
       # flatten,permute,repeat  [27, 1, 256]
        # quert_pos_emb: [27, 1, 256]
        query_pos_emb = self.pos_emb(
            bs, self.kernel_dim, self.kernel_dim, f_e.device
        ).flatten(2).permute(2, 0, 1).repeat(self.num_objects, 1, 1)

        if self.num_iterative_steps > 0:
            # memory: [4096, 1, 256]
            memory = f_e.flatten(2).permute(2, 0, 1)

            # all_prototypes: [3, 27, 1, 256]
            all_prototypes = self.iterative_adaptation(
                shape_or_objectness, appearance, memory, pos_emb, query_pos_emb
            )
        else:
            if shape_or_objectness is not None and appearance is not None:
                all_prototypes = (shape_or_objectness + appearance).unsqueeze(0)
            else:
                all_prototypes = (
                    shape_or_objectness if shape_or_objectness is not None else appearance
                ).unsqueeze(0)

        return all_prototypes


# 迭代适应模块
class IterativeAdaptationModule(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool
    ):

        super(IterativeAdaptationModule, self).__init__()

        # 给迭代适应模块创建了num_layers(default=3)个迭代适应层
        self.layers = nn.ModuleList([
            IterativeAdaptationLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, zero_shot
            ) for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    # memory是输入特征图
    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):

        output = tgt
        outputs = list()
        # self.layers是迭代适应层
        for i, layer in enumerate(self.layers):
            # output是形状查询
            # appearance是外观查询
            # 这里会调用IterativeAdaptationLayer的forward() output: [27, 1, 256]
            output = layer(
                output, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask
            )

            outputs.append(self.norm(output))
        # 把每一次迭代适应层的结果堆叠起来
        # 堆叠前outputs中有3个元素，每个元素都是[27, 1, 256]
        # 堆叠以后变为[3, 27, 1, 256]
        return torch.stack(outputs)


# 迭代适应层
class IterativeAdaptationLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        zero_shot: bool
    ):
        super(IterativeAdaptationLayer, self).__init__()


        self.norm_first = norm_first
        self.zero_shot = zero_shot

        # 这里的norm应该是对应论文迭代适应模块中的三个norm，当零样本情况下就不需要处理形状查询的norm了，drop同理。
        # 判断self.zero_shot是否为False
        if not self.zero_shot:
            self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)

        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm3 = nn.LayerNorm(emb_dim, layer_norm_eps)

        if not self.zero_shot:
            self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


        if not self.zero_shot:
            self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)

        # 这里对应的是论文图中的FNN
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask
    ):
        # 这里对应论文中迭代适应模块那个图的传递过程
        if self.norm_first:

            if not self.zero_shot:
                # tgt_norm应该就是形状查询
                # 对形状特征进行norm
                tgt_norm = self.norm1(tgt)

                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt_norm, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])

            tgt_norm = self.norm2(tgt)
            # 对比图这个里的memory应该是特征图
            tgt = tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt_norm, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0])

            tgt_norm = self.norm3(tgt)
            tgt = tgt + self.dropout3(self.mlp(tgt_norm))

        else:
            if not self.zero_shot:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key = self.with_emb(appearance),
                    # key=self.with_emb(appearance,query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

            tgt = self.norm2(tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0]))

            tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))

        return tgt