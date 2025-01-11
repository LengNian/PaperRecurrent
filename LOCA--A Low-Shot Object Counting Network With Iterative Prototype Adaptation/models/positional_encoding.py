import torch
from torch import nn


class PositionalEncodingsFixed(nn.Module):

    def __init__(self, emb_dim, temperature=10000):

        super(PositionalEncodingsFixed, self).__init__()

        self.emb_dim = emb_dim
        # 用于控制位置编码的频率范围
        self.temperature = temperature

    # 生成一维位置编码
    def _1d_pos_enc(self, mask, dim):
        # mask: bool, 表示哪些位置填充, True表示填充, False表示有效位置

        # 计算频率因子
        temp = torch.arange(self.emb_dim // 2).float().to(mask.device)
        temp = self.temperature ** (2 * (temp.div(2, rounding_mode='floor')) / self.emb_dim)

        # 计算位置编码
        enc = (~mask).cumsum(dim).float().unsqueeze(-1) / temp
        # 将正弦和余弦函数应用于编码, 并将结果拼接在一起，同时将最后两个维度展平
        enc = torch.stack([
            enc[..., 0::2].sin(), enc[..., 1::2].cos()
        ], dim=-1).flatten(-2)

        return enc

    # 生成二维编码
    def forward(self, bs, h, w, device):
        # mask: [1, 64, 64]
        mask = torch.zeros(bs, h, w, dtype=torch.bool, requires_grad=False, device=device)
        # 在宽度维度上生成一维位置编码
        # x: [1, 64, 64, 128]
        x = self._1d_pos_enc(mask, dim=2)

        # 在高度维度上生成一维位置编码
        # y: [1, 64, 64, 128]
        y = self._1d_pos_enc(mask, dim=1)
        # cat([y, x]): [1, 64, 64, 256]
        # permute(0, 3, 1, 2): [1, 256, 64, 64]

        return torch.cat([y, x], dim=3).permute(0, 3, 1, 2)
