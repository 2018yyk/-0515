import torch
import torch.nn as nn
import numpy as np


def get_time_embedding(timesteps, dim=128):
    device = timesteps.device
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), mode='constant')
    return emb


class ExtendedLinearUNet(nn.Module):
    def __init__(self):
        super(ExtendedLinearUNet, self).__init__()

        # 扩展层
        self.expand = nn.Linear(25, 64)
        self.hidden1 = nn.Linear(64, 128)
        # 隐藏层
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 64)

        # 压缩层
        self.compress = nn.Linear(64, 25)

        # 激活函数
        self.gelu = nn.GELU()

    def forward(self, x, timesteps):
        # 获取时间嵌入
        time_emb = get_time_embedding(timesteps)

        # 扩展阶段
        skip1 = self.gelu(self.expand(x))
        x = self.gelu(self.hidden1(skip1))

        x = x + time_emb
        # 隐藏层处理
        x = self.gelu(self.hidden2(x))
        # x = x + time_emb
        x = self.gelu(self.hidden3(x))

        # 跳跃连接
        x = x + skip1

        # 压缩阶段
        x = self.compress(x)

        return x

if __name__ == '__main__':
    model = ExtendedLinearUNet()
    input_data = torch.randn(1, 25)
    timesteps = torch.tensor([0])
    output = model(input_data, timesteps)
    print(output.shape)
    