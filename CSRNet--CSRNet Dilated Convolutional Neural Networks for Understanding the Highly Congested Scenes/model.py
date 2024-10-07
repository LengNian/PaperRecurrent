import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        # 前景网络
        self.frontend = make_layers(self.frontend_feat)
        # 后景网络
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        # 输出层 1×1的卷积层
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                # self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    # 前向传播
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


    def _initialize_weights(self):
        # self.modules是nn.Module类的一个方法，返回一个迭代器，遍历模块中所有的子模块。
        for m in self.modules():
            # 检查是否是Conv2d，如果是卷积层，使用正态分布初始化其权重，
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # 检查卷积层是否有偏置项，如果有用常数0进行初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 用常数1初始化批量归一化层的权重，0初始化偏置
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
# 参数: 网络配置，输入通道，是否batchnormal批量归一化，空洞卷积膨胀率
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    # 论文中提到了对CSRNet的四个配置进行测试，得出B部分的效果最好，所以这里的dilation均为2
    if dilation:
        d_rate = 2
    else:
        # d_rate=1时，空洞卷积就退化成了普通的卷积
        d_rate = 1
    layers = []
    for v in cfg:
        # 如果是M，就是最大池化层，否则就是卷积层+ReLU层
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)