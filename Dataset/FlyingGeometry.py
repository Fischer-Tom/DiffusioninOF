from os.path import join
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from imagelib.io import read
from imagelib.core import display_flow_tensor, display_tensor
from torchvision import transforms


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


class FlyingGeometryDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.IDs = [remove_suffix(file, '_flow.flo') for file in os.listdir(self.img_dir) if file.endswith('.flo')]

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        path = join(self.img_dir, self.IDs[idx])
        im1 = read(f'{path}_img1.ppm')
        im2 = read(f'{path}_img2.ppm')
        edges = read(f'{path}_flow.ppm')
        flow = read(f'{path}_flow.flo')
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
            flow = self.transform(flow)
            edges = self.transform(edges)
            noisyFlow, cleanFlow = self.addNoise(edges, flow)

        return im1, im2, flow, noisyFlow, cleanFlow

    def addNoise(self, edges, flowField):
        mask = edges[0,:,:]
        mask = mask == 1
        cleanflowField = torch.zeros_like(flowField)
        cleanflowField[:, mask] = flowField[:, mask]

        noisyFlowField = torch.empty_like(flowField).normal_(mean=0, std=20)
        noisyFlowField[:, mask] = flowField[:, mask]
        return noisyFlowField, cleanflowField


transforms = transforms.Compose([transforms.ToTensor()])
dataset = FlyingGeometryDataset('Dataset/Data', transform=transforms)

im1, im2, flow, noisyFlow, cleanFlow = next(iter(dataset))




"""
inp = InpaintingBlock().to('cuda')
noisyFlow = noisyFlow.to('cuda')
cleanFlow = cleanFlow.to('cuda')
with torch.no_grad():
    clean = inp(noisyFlow,cleanFlow, 3000)
"""
display_tensor(im1)
display_flow_tensor(flow)
display_flow_tensor(noisyFlow.cpu().detach())
display_flow_tensor(cleanFlow.cpu().detach())
#display_flow_tensor(clean.cpu().detach())
