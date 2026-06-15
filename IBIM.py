import torch
import torch.nn as nn
import numpy as np

class IBIM(nn.Module):
    def __init__(self, in_channel):
        super(IBIM, self).__init__()
        self.gelu = nn.GELU()
        self.convbranch1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel,
                      kernel_size=(3, 3), stride=1,
                      padding=1, dilation=1,  bias=False),
            nn.BatchNorm2d(in_channel),

            nn.Conv2d(in_channel, in_channel,
                      kernel_size=(5, 5), stride=1,
                      padding=2, dilation=1,  bias=False),
            nn.BatchNorm2d(in_channel),

            nn.Conv2d(in_channel, in_channel,
                      kernel_size=(7, 7), stride=1,
                      padding=3, dilation=1,  bias=False),
            nn.BatchNorm2d(in_channel)
        )

        self.mlpbranch = nn.Sequential(
            Mixer_chan(dim=in_channel, channel_dim=4 * in_channel, out_channel=in_channel),
        )

    def forward(self, x):
        # xr1 [2, 64, 96, 96] t1
        # xr2 [2, 64, 48, 48] t2
        # xr3 [2, 64, 24, 24] t3
        # xr4 [2, 64, 12, 12] t4

        convx = self.convbranch1(x) + x
        convx = convx.transpose(1, 3)
        mlpx = self.mlpbranch(convx)
        mlpx = mlpx.transpose(1, 3)
        x = self.gelu(mlpx)
        return x

class Mixer_chan(nn.Module):
    def __init__(self, dim=64, channel_dim=64*4, out_channel=16, dropout=0.75):
        super(Mixer_chan, self).__init__()

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            MLPBlock(dim, channel_dim, out_channel)
        )

    def forward(self, x):
        out = self.channel_mixer(x)
        return out


class MLPBlock(nn.Module):
    def __init__(self, mlp_dim:int, hidden_dim:int, out_dim:int, dropout = 0.1):
        super(MLPBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, out_dim)
    def forward(self,x):
        x = self.Linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.dropout(x)
        return x


