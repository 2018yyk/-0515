import torch
import torch.nn as nn
import numpy as np


def get_time_embedding(timesteps, dim=128):
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), mode='constant')
    return emb


class TimeEmbeddingLayer(nn.Module):
    def __init__(self, time_emb_dim, out_channels):
        super(TimeEmbeddingLayer, self).__init__()
        self.linear1 = nn.Linear(time_emb_dim, time_emb_dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(time_emb_dim * 4, out_channels)

    def forward(self, time_emb):
        time_emb = self.linear1(time_emb)
        time_emb = self.act(time_emb)
        time_emb = self.linear2(time_emb)
        return time_emb


class DownBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(DownBlock1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.time_emb_layer = TimeEmbeddingLayer(time_emb_dim, out_channels)

    def forward(self, x, time_emb):
        time_emb_bias = self.time_emb_layer(time_emb).unsqueeze(-1)
        x = self.conv(x)
        x = x + time_emb_bias
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class UpBlock1D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super(UpBlock1D, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Conv1d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.time_emb_layer = TimeEmbeddingLayer(time_emb_dim, out_channels)

    def forward(self, x, skip, time_emb):
        if x.shape[-1] != skip.shape[-1]:
            x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        time_emb_bias = self.time_emb_layer(time_emb).unsqueeze(-1)
        x = self.conv(x)
        x = x + time_emb_bias
        x = self.bn(x)
        x = self.relu(x)
        return x


class FinalUpBlock1D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super(FinalUpBlock1D, self).__init__()
        self.conv = nn.Conv1d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.time_emb_layer = TimeEmbeddingLayer(time_emb_dim, out_channels)

    def forward(self, x, skip, time_emb):
        x = torch.cat([x, skip], dim=1)
        time_emb_bias = self.time_emb_layer(time_emb).unsqueeze(-1)
        x = self.conv(x)
        x = x + time_emb_bias
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet1DModel(nn.Module):
    def __init__(self, in_channels=17, out_channels=1, block_out_channels=(64, 128), time_emb_dim=128):
        super(UNet1DModel, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.encoder = nn.Linear(25, 32)
        self.decoder = nn.Linear(16, 25)
        self.down_block1 = DownBlock1D(in_channels, block_out_channels[0], time_emb_dim)
        self.down_block2 = DownBlock1D(block_out_channels[0], block_out_channels[1], time_emb_dim)

        self.up_block1 = UpBlock1D(block_out_channels[1], block_out_channels[0], block_out_channels[0], time_emb_dim)
        self.up_block2 = FinalUpBlock1D(block_out_channels[0], block_out_channels[0], out_channels, time_emb_dim)

    def forward(self, x, timesteps):
        time_emb = get_time_embedding(timesteps, dim=self.time_emb_dim)
        # 编码
        x = self.encoder(x)

        # 下采样
        skip1 = self.down_block1(x, time_emb)
        skip2 = self.down_block2(skip1, time_emb)

        # 上采样
        up1 = self.up_block1(skip2, skip1, time_emb)
        x = self.up_block2(up1, skip1, time_emb)

        # 解码
        x = self.decoder(x)

        return x
    