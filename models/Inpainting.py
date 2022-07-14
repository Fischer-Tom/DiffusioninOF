import torch
import torch.nn as nn

class NaiveInpaintingBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, groups=2,
                              bias=False, padding_mode='reflect')
        self.conv.requires_grad = False
        self.init_weights()

    def forward(self, x, goodFlow, time_steps=20):
        mask = goodFlow == 0
        for _ in range(time_steps):
            diff_x = self.conv(x)
            x = goodFlow + diff_x*mask
        return x

    def init_weights(self, tau=0.25, h1=1, h2=1):
        hx = tau / (h1 * h1)
        hy = tau / (h2 * h2)
        weight = torch.zeros_like(self.conv.weight)
        weight[0][0][1][0] = hx
        weight[0][0][1][2] = hx
        weight[0][0][0][1] = hy
        weight[0][0][2][1] = hy
        weight[0][0][1][1] = (1 - 2 * hx - 2 * hy)
        weight[1][0][1][0] = hx
        weight[1][0][1][2] = hx
        weight[1][0][0][1] = hy
        weight[1][0][2][1] = hy
        weight[1][0][1][1] = (1 - 2 * hx - 2 * hy)
        self.conv.weight = nn.Parameter(weight)
