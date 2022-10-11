import torch
import torch.nn as nn
import torch.nn.functional as F
import models.basicblock as B
import numpy as np
from utils import utils_image as util
from math import sqrt
import os
import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""
def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


def filter_tensor(x, sf=3):
    z = torch.zeros(x.shape)
    z[..., ::sf, ::sf].copy_(x[..., ::sf, ::sf])
    return z

def o_leary_batch(x, masks, kernels):
    device = x.device
    n_kernels = kernels.size(1)
    Hk = kernels.size(2)
    Wk = kernels.size(3)
    N = x.size(0)
    C = x.size(1)
    H = x.size(2)
    W = x.size(3)
    
    kernels = torch.flip(kernels, dims =(2,3))

    x = torch.nn.functional.pad(x, (Wk//2,Wk//2,Hk//2,Hk//2), mode='circular')

    output = torch.empty(N, n_kernels, C, H, W).to(device)

    for num in range(N):
        for c in range(C):
            conv_output = F.conv2d(x[num:num + 1, c:c + 1, :, :],
                                       kernels[num][:, np.newaxis, :, :])

            output[num:num + 1, :, c, :, :] = conv_output * masks[num:num + 1]
            del conv_output

    output = torch.sum(output, (1))

    return output

def transpose_o_leary_batch(x, masks, kernels):
    device = x.device
    n_kernels = kernels.size(1)
    Hk = kernels.size(2)
    Wk = kernels.size(3)
    N = x.size(0)
    C = x.size(1)
    H = x.size(2)
    W = x.size(3)

    x = torch.nn.functional.pad(x, (Wk//2,Wk//2,Hk//2,Hk//2), mode='circular')

    output = torch.empty(N, n_kernels, C, H, W).to(device)

    for num in range(N):
        for c in range(C):
            conv_output = F.conv2d(x[num:num + 1, c:c + 1, :, :],
                                       kernels[num][:, np.newaxis, :, :])

            output[num:num + 1, :, c, :, :] = conv_output * masks[num:num + 1]
            del conv_output

    output = torch.sum(output, (1))

    return output

"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x

"""
# --------------------------------------------
# (2) Data module, closed-form solution
# --------------------------------------------
"""
class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()
        
    def forward(self, x, STy, alpha, sf):
        I = torch.ones_like(STy) * alpha
        I[...,::sf,::sf] += 1
        return (STy + alpha * x) / I

class DataNetDeblur(nn.Module):
    def __init__(self):
        super(DataNetDeblur, self).__init__()
        
    def forward(self, x, y, alpha, sf):
        return (y + alpha * x) / (1 + alpha) 
