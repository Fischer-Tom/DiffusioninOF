import torch
import flow_vis
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math
from PIL import Image
from imagelib.io import read
from torchvision import transforms, utils
import torch.nn.functional as F

def image_to_tensor(path: str, mode='RGB'):
    if mode == 'GS':
        im = Image.open(path).convert('L')
        im_np = np.expand_dims(np.asarray(im), axis=2)
    else:
        im = Image.open(path)
        im_np = np.asarray(im)
    return torch.from_numpy(im_np).permute(2, 0, 1)

def image_to_numpy(path: str, mode='RGB'):
    if mode == 'GS':
        im = Image.open(path).convert('L')
        im_np = np.expand_dims(np.asarray(im), axis=2)
    else:
        im = Image.open(path)
        im_np = np.asarray(im)
    return im_np


def tensor_to_image(tensor):
    pass


def display_numpy_image(array):
    if len(array.shape) == 2:
        plt.imshow(array, cmap='gray')
        plt.axis('off')
        plt.show()
        return
    plt.imshow(array)
    plt.axis('off')
    plt.show()

def display_flow_tensor(tensor):
    if len(tensor.shape) == 4:
        display_batch_flow_tensor(tensor)
    else:
        np_flow = tensor.permute((1, 2, 0)).numpy()
        flow_color = flow_vis.flow_to_color(np_flow, convert_to_bgr=False)
        plt.imshow(flow_color)
        plt.axis('off')
        plt.imsave('edges.png', flow_color)
        plt.show()

def display_flow_numpy(np_array):
    flow_color = flow_vis.flow_to_color(np_array, convert_to_bgr=False)
    plt.imshow(flow_color)
    plt.axis('off')
    plt.show()

def display_batch_flow_tensor(tensor):
    raise Exception("not implemented!")


def display_tensor(tensor):
    if tensor.shape[0] == 3:
        plt.imshow(tensor.permute(1, 2, 0))
    else:
        plt.imshow(tensor.permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    plt.imsave('im1.png', tensor.permute(1,2,0).numpy())
    plt.show()


def hom_diff_gs(image, diff_steps, tau=0.25, h1=1, h2=1):
    assert (tau <= 0.25)
    width = image.shape[1]
    height = image.shape[2]
    hx = tau / (h1 * h1)
    hy = tau / (h2 * h2)
    for t in range(diff_steps):
        copy = image
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                image[0][i][j] = (1 - 2 * hx - 2 * hy) * copy[0][i][j] + hx * copy[0][i - 1][j] + \
                                 hx * copy[0][i + 1][j] + hy * copy[0][i][j - 1] + hy * copy[0][i][j + 1]
    return image


def hom_diff_inpaint_gs(image, mask, diff_steps, tau=0.25, h1=1, h2=1):
    assert mask.shape == image.shape

    width = image.shape[1]
    height = image.shape[2]
    hx = tau / (h1 * h1)
    hy = tau / (h2 * h2)
    for t in range(diff_steps):
        copy = image
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if mask[0][i][j]:
                    image[0][i][j] = copy[0][i][j]
                else:
                    image[0][i][j] = (1 - 2 * hx - 2 * hy) * copy[0][i][j] + hx * copy[0][i - 1][j] + \
                                     hx * copy[0][i + 1][j] + hy * copy[0][i][j - 1] + hy * copy[0][i][j + 1]
    return image


class DiffusionBlock(nn.Module):

    def __init__(self):
        super(DiffusionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, groups=2,
                              bias=False, padding_mode='reflect')
        self.conv.requires_grad = False
        self.init_weights()

    def forward(self, x, time_steps=20):
        for _ in range(time_steps):
            x = self.conv(x)
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


class InpaintingBlock(nn.Module):

    def __init__(self):
        super(InpaintingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, groups=2,
                              bias=False, padding_mode='reflect')
        self.conv.requires_grad = False
        self.init_weights()

    def forward(self, x, time_steps=20):
        masked_x = x
        mask = x == 0
        for _ in range(time_steps):
            diff_x = self.conv(x)
            x = masked_x + diff_x*mask
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


def EED(flow_tensor, t=10, lamb=0.5):

    c, h, w = flow_tensor.size()

    grad = torch.gradient(flow_tensor, spacing=1, dim= [1,2])
    grad = torch.stack((grad[0], grad[1]), dim=1)
    grad_norm = torch.norm(grad, p=2, dim=1)
    g = 1./(1.+grad_norm**2/lamb**2)
    grad_orth = torch.stack((-grad[1], grad[0]), dim=1)
    grad = F.normalize(grad, p=2.0, dim=1)
    grad_orth = F.normalize(grad_orth,p=2.0, dim=1)
    print(grad.shape)
    print(grad_orth.shape)
    diff_tensors_grad = torch.einsum('bchw,bkhw->bckhw',grad,grad)
    diff_tensors_orth = torch.einsum('bchw,bkhw->bckhw', grad_orth, grad_orth)

    diff_tensor = torch.sum(g,dim=0)*torch.sum(diff_tensors_grad,dim=0) + torch.sum(diff_tensors_orth,dim=0)

    kernel = get_weight(diff_tensor,t)
    padded_flow = F.pad(flow_tensor, (2, 2, 2, 2), mode='constant', value=0.0).unsqueeze(0)
    unfolded_flow = padded_flow.unfold(2,5,1).unfold(3,5,1).reshape(c, h, w, 5,5)
    out = torch.sum(unfolded_flow*kernel, dim=[-1,-2])

    #display_tensor(flow_tensor.cpu().detach())
    #display_tensor(out.cpu().detach())
    display_flow_tensor(flow_tensor.cpu().detach())
    display_flow_tensor(out.cpu().detach())

def get_weight(diff_tensor, t):
    _,_,h,w = diff_tensor.size()

    diff_tensor = diff_tensor.flatten(2).permute(2,0,1)
    diffI = torch.linalg.pinv(diff_tensor) #b_inv(diff_tensor)
    b,_,_ = diffI.size()
    offsetx,offsety = torch.meshgrid([torch.arange(0,5),torch.arange(0,5)])
    output = torch.zeros(b,5,5).to('cuda')
    for i,j in zip(offsetx.reshape(-1),offsety.reshape(-1)):
        xT = torch.Tensor([[[float(i-2),float(j-2)]]]).to('cuda')
        x = torch.tensor([[[float(i-2)],[float(j-2)]]]).to('cuda')
        val = 1./(4.*torch.pi*t)*torch.exp(-torch.matmul(torch.matmul(xT,diffI),x).squeeze() / (4*t))
        output[:,i,j] = val
        #output[:,i,j].detach().cpu().apply_(flux)
    output = output / torch.sum(output, dim=[1,2], keepdim=True)
    return output.reshape(h,w,5,5).to('cuda')

def b_inv(b_mat):
    eye = torch.eye(2).reshape((1,2,2)).repeat(b_mat.size(0),1,1).to('cuda')
    b_inv = torch.linalg.solve(b_mat,eye)
    return b_inv

def flux(x):
    if x <= 0.:
        return 1.
    else:
        return 1.-math.exp(-3.31488/(x/2.)**4.)

def iso_diff(flow_tensor, t=10000, lamb=5e-2):

    c, h, w = flow_tensor.size()

    grad = torch.gradient(flow_tensor, spacing=1, dim= [1,2])
    grad = torch.stack((grad[0], grad[1]), dim=1)
    print(grad.shape)
    diff_tensor = 1./(1.+torch.sum(torch.norm(grad, p=2,dim=1)**2, dim=0)**2/lamb**2 )
    print(diff_tensor.mean())
    kernel = get_weight_scalar(diff_tensor,t)
    kernel = torch.nan_to_num(kernel,nan=0.0)
    padded_flow = F.pad(flow_tensor, (2, 2, 2, 2), mode='constant', value=0.0).unsqueeze(0)
    unfolded_flow = padded_flow.unfold(2,5,1).unfold(3,5,1).reshape(c, h, w, 5,5)
    out = torch.sum(unfolded_flow * kernel, dim=[-1,-2])

    #display_tensor(flow_tensor.cpu().detach())
    #display_tensor(out.cpu().detach())
    display_flow_tensor(flow_tensor.cpu().detach())
    display_flow_tensor(out.cpu().detach())

def get_weight_scalar(diff_tensor, t):
    h,w = diff_tensor.size()
    print(diff_tensor.shape)
    offsetx,offsety = torch.meshgrid([torch.arange(0,5),torch.arange(0,5)])
    diff_tensor = 1/diff_tensor.flatten()
    output = torch.zeros(h*w,5,5)
    for i,j in zip(offsetx.reshape(-1),offsety.reshape(-1)):
        xT = torch.Tensor([[[float(i-2),float(j-2)]]]).to('cuda')
        x = torch.tensor([[[float(i-2)],[float(j-2)]]]).to('cuda')
        val = 1./(4.*torch.pi*t)*torch.exp(diff_tensor*torch.matmul(xT,x) / (4*t))
        output[:,i,j] = val
        #output[:,i,j].detach().cpu().apply_(flux)
    output = output / torch.sum(output, dim=[1,2], keepdim=True)
    return output.reshape(h,w,5,5).to('cuda')
"""
tens1 = torch.Tensor(read('../datasets/FlyingChairs_release/data/00001_flow.flo')).permute(2, 0, 1)
tens2 = torch.Tensor(read('../datasets/FlyingChairs_release/data/00002_flow.flo')).permute(2, 0, 1)
tens = torch.stack((te"ns1, tens2), dim=0)
display_flow_tensor(tens[0])
block = DiffusionBlock()
diff_tens = block(tens)
display_flow_tensor(diff_tens[0].detach())
masc = torch.FloatTensor(1, 1, tens.shape[2], tens.shape[3]).uniform_() > 0.9
masc = masc.repeat(1, 2, 1, 1)
block = InpaintingBlock()
inp_tens = block(tens * masc, 50)
masked_tens = inp_tens * masc
display_flow_tensor(masked_tens[0].detach())
display_flow_tensor(inp_tens[0].detach())
"""
