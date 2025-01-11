import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d


class Backbone(nn.Module):

    def __init__(
        self,
        name: str,
        pretrained: bool,
        dilation: bool,
        reduction: int,
        swav: bool,
        requires_grad: bool
    ):

        super(Backbone, self).__init__()

        # 根据name动态的获取对应的模型
        resnet = getattr(models, name)(
            # 控制是否在resnet的最后几层使用空洞卷积
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained,
            # 使用冻结的batchnorm层
            norm_layer=FrozenBatchNorm2d
        )

        self.backbone = resnet
        self.reduction = reduction


        if name == 'resnet50' and swav:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',
                map_location="cpu"
            )
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        # concatenation of layers 2, 3 and 4
        # 在resnet中layer2，3，4通道分别为512，1024， 2048，加起来就是3584
        self.num_channels = 896 if name in ['resnet18', 'resnet34'] else 3584

        for n, param in self.backbone.named_parameters():
            # 冻结除layer2, 3, 4以外的层的参数
            if 'layer2' not in n and 'layer3' not in n and 'layer4' not in n:
                param.requires_grad_(False)
            else:
                param.requires_grad_(requires_grad)

    def forward(self, x):
        # x: [1, 3, 512, 512]

        # size: (64, 64)
        # 取输入图像的长和宽 // self.reduction
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction

        # [1, 3,  512, 512]  -->  [1, 64, 256, 256]
        x = self.backbone.conv1(x)
        # [1, 64, 256, 256]
        x = self.backbone.bn1(x)
        # [1, 64, 256, 256]
        x = self.backbone.relu(x)
        # [1, 64, 256, 256]  -->  [1, 64, 128, 128]
        x = self.backbone.maxpool(x)

        # [1, 64, 256, 256]  -->  [1, 256, 128, 128]
        x = self.backbone.layer1(x)
        # [1, 256, 256, 256]  -->  [1, 512, 64, 64]
        x = layer2 = self.backbone.layer2(x)
        # # [1, 512, 64, 64]  -->  [1, 1024, 32, 32]
        x = layer3 = self.backbone.layer3(x)
        # [1, 1024, 32, 32]  -->  [1, 2048, 16, 16]
        x = layer4 = self.backbone.layer4(x)

        # [1, 2048, 16, 16]  -->  [1, 3584, 64, 64]
        x = torch.cat([
            # 使用双线性插值将特征图缩放到指定尺寸64
            F.interpolate(f, size=size, mode='bilinear', align_corners=True)
            for f in [layer2, layer3, layer4]
        ], dim=1)

        # x: [1, 3584, 64, 64]

        return x
