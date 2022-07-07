from os.path import join
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from image_lib.io import read
from image_lib.core import display_flow_tensor, display_tensor
from torchvision import transforms

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

class FlyingGeometryDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.IDs = [remove_suffix(file,'_flow.flo') for file in os.listdir(self.img_dir) if file.endswith('.flo')]

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        path = join(self.img_dir, self.IDs[idx])
        im1 = read(f'{path}_img1.ppm')
        im2 = read(f'{path}_img2.ppm')
        flow = read(f'{path}_flow.flo')
        im1 = read(f'Data/095_img1.ppm')
        im2 = read(f'Data/095_img2.ppm')
        flow = read(f'Data/095_flow.flo')
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
            flow = self.transform(flow) / 255.
            noisyFlow = self.addNoise(im1, flow)

        return im1, im2, flow, noisyFlow

    def addNoise(self, im1, flowField):
        im1P = F.pad(im1, (0,1,0,1), value=0.)
        fx,fy = torch.gradient(im1P, dim=[1,2])
        fxx,  = torch.gradient(fx, dim=1)
        fyy, = torch.gradient(fy, dim=2)
        laplacian = torch.sum(fxx + fyy, dim=0)
        sign = torch.sign(laplacian)
        diff_x = sign[:-1, :-1] - sign[:-1, 1:] < 0
        diff_y = sign[:-1, :-1] - sign[1:, :-1] < 0
        mask = torch.logical_or(diff_x, diff_y)
        cleanflowField = torch.zeros_like(flowField)
        cleanflowField[:,mask] = flowField[:,mask]
        noise = torch.rand_like(flowField)
        noisyFlowField = flowField + (0.001**0.5)*noise
        noisyFlowField[:,mask] = flowField[:,mask]
        return noisyFlowField

transforms = transforms.Compose([transforms.ToTensor()])
dataset = FlyingGeometryDataset('Data', transform=transforms)

im1, im2, flow, noisyFlow = dataset.__getitem__(95)
display_tensor(im1)
display_flow_tensor(flow)
display_flow_tensor(noisyFlow)