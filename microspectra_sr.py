# 1. 第三方库导入
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QScrollArea, QFrame, QSizePolicy, QListWidget, QListWidgetItem,
    QGridLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect
import scipy.io as sio
import math
from math import ceil
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
from ui_core import strip_inline_styles
# 2. 工具函数和imresize
# =====================
def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale

def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x+1), lessthanzero) + np.multiply((1-x), greaterthanzero)
    return f

def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5 *
                                                                        absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        def h(x): return scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)
    weights = np.divide(weights, np.expand_dims(
        np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(
        in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(
                    np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(
                    np.squeeze(im_slice, axis=0), w.T), axis=0)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(
            weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(
            weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        raise ValueError('unidentified kernel method supplied')

    kernel_width = 4.0
    if scalar_scale is not None and output_shape is not None:
        raise ValueError(
            'either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError(
            'either scalar_scale OR output_shape should be defined')
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(
            I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B

# 3. HSTestData类
# ===============
class HSTestData(torch.utils.data.Dataset):
    def __init__(self, image_path, dataset):
        if os.path.isdir(image_path):
            self.image_files = [os.path.join(image_path, x) for x in os.listdir(image_path)]
        else:
            self.image_files = [image_path]
        self.dataset = dataset

    def __getitem__(self, index):
        file_index = index
        load_dir = self.image_files[file_index]
        test_data = sio.loadmat(load_dir)
        self.ms = np.array(test_data['ms'][...], dtype=np.float32)
        self.lms = np.array(test_data['ms_bicubic'][...], dtype=np.float32)
        self.gt = np.array(test_data['gt'][...], dtype=np.float32)
        gt = self.gt[:, :, :]
        ms = self.ms[:, :, :]
        lms = self.lms[:, :, :]
        ms = ms / 25000
        lms = lms / 25000
        gt = gt / 25000
        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return ms, lms, gt, self.image_files[file_index]

    def __len__(self):
        return len(self.image_files)

# 4. common.py依赖（default_conv、DeformConv2d、Upsampler等）
# =========================================================
def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)
    else:
       padding = int((kernel_size - 1) / 2) * dilation
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           stride, padding=padding, bias=bias, dilation=dilation)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride), indexing='ij')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N]*padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        return x_offset

# 5. csa.py依赖（window_partition、window_reverse、WindowAttention、SwinT等）
# =====================================================================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x1, x2, mask=None):
        B_, N, C = x1.shape
        qkv1 = self.qkv(x1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        qkv2 = self.qkv(x2).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        q = q1 * self.scale
        attn = (q @ k2.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v2).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class SwinT(nn.Module):
    def __init__(self, dim, input_resolution=[50,50], num_heads=6, window_size=5, shift_size=2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim*2, dim, 1, 1, 0, bias=False)
        )
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    def forward(self, x1, x2, x_size):
        B, H, W, C = x1.shape
        B2, H, W, C2 = x2.shape
        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x2 = torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x1 = x1
            shifted_x2 = x2
        x_windows1 = window_partition(shifted_x1, self.window_size)
        x_windows1 = x_windows1.view(-1, self.window_size * self.window_size, C)
        x_windows2 = window_partition(shifted_x2, self.window_size)
        x_windows2 = x_windows2.view(-1, self.window_size * self.window_size, C2)
        attn_windows = self.attn(x_windows1, x_windows2, mask=self.calculate_mask(x_size).to(x1.device))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x0 = x.view(B, C, H, W)
        x1 = x1.permute(0, 3, 1, 2)
        x = torch.cat([x0, x1], dim=1)
        x = self.conv(x)
        return x
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops

# 6. MSDformer_ours.py依赖（MSDformer、MSAMG、ESA、DMSA、DCTM、FFN等）
# ================================================================
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.2),
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.2),
            nn.Conv2d(hidden_features, in_features, 1, 1, 0)
        )
    def forward(self, x):
        x = self.ffn(x)
        return x

class DMSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DMSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.deformconv = DeformConv2d(dim, dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, h, w = x.shape
        q = self.deformconv(x)
        _, k, v = self.qkv(x).chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.dconv1 = conv(f, f, 3, dilation=1)
        self.dconv2 = conv(f, f, 3, dilation=3)
        self.dconv3 = conv(f, f, 3, dilation=5)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        b, c, h, w = x.size()
        c1_ = (self.conv1(x))
        c1 = self.dconv1(c1_)
        c2 = self.dconv2(c1_)
        c3 = self.dconv3(c1_)
        c = c1 + c2 + c3
        v_max1 = F.max_pool2d(c, kernel_size=7, stride=3)
        c = self.relu(self.conv_max(v_max1))
        c = self.relu(self.conv3(c))
        c = self.conv3_(c)
        c = F.interpolate(c, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c + cf)
        m = self.sigmoid(c4)
        out = x * m
        return out

class DCTM(nn.Module):
    def __init__(self, dim, num_heads, shift_size=0, drop_path=0.0,
                 mlp_ratio=4., drop=0., act_layer=nn.GELU, bias=False):
        super(DCTM, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim)
        self.num_heads = num_heads
        self.esa = ESA(dim)
        self.global_attn = DMSA(dim, num_heads, bias)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        shortcut = x
        x = self.norm1(x)
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x1 = self.esa(x)
        x2 = self.global_attn(x)
        x = x1 + x2
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = shortcut + self.drop_path(x)
        x = self.norm2(x)
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.mlp(x)
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = shortcut + self.drop_path(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x

class MSAMG(nn.Module):
    def __init__(self, n_subs, n_ovls, n_feats, conv=default_conv):
        super(MSAMG, self).__init__()
        self.G = math.ceil((n_feats - n_ovls) / (n_subs - n_ovls))
        self.n_feats = n_feats
        self.start_idx = []
        self.end_idx = []
        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_feats :
                end_ind = n_feats
                sta_ind = n_feats - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)
        self.conv2 = BSConvU(n_subs, n_subs, kernel_size=3,  stride=1, padding=1, bias=False)
        self.attn = SwinT(n_subs)
        self.spc = nn.ModuleList()
        for n in range(self.G):
            self.spc.append(ResAttentionBlock(conv, n_subs, 1, res_scale=0.1))
    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.zeros(b, c, h, w).cuda()
        channel_counter = torch.zeros(c).cuda()
        xi_out = []
        xi_mid = []
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = x[:, sta_ind:end_ind, :, :]
            if g > 0:
                xi = xi + xi_mid[g-1]
            xi = self.conv2(xi)
            xi_mid.append(xi)
        for i in range(self.G):
            if i < self.G - 1:
                xi = self.attn(xi_mid[i].permute(0, 2, 3, 1),xi_mid[i+1].permute(0, 2, 3, 1),(h,w))
            else:
                xi = self.attn(xi_mid[i].permute(0, 2, 3, 1),xi_mid[i].permute(0, 2, 3, 1), (h,w))
            xi = self.spc[i](xi)
            xi_out.append(xi)
        for i in range(self.G):
            sta_ind = self.start_idx[i]
            end_ind = self.end_idx[i]
            y[:, sta_ind:end_ind, :, :] += xi_out[i]
            channel_counter[sta_ind:end_ind] += 1
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = y + x
        return y

class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        if bn_kwargs is None:
            bn_kwargs = {}
        self.pw = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )
    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        m.append(CALayer(n_feats, 16))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class MSDformer(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, scale, n_feats, n_DCTM,  conv=default_conv):
        super(MSDformer, self).__init__()
        self.conv1 = nn.Conv2d(n_colors, n_feats,3,1,1)
        self.head = MSAMG(n_subs, n_ovls, n_feats)
        self.body = nn.ModuleList()
        self.N = n_DCTM
        for i in range(self.N):
            self.body.append(DCTM(n_feats, 6, False))
        self.skip_conv = conv(n_colors, n_feats, 3)
        self.upsample = Upsampler(conv, scale, n_feats)
        self.tail = conv(n_feats, n_colors, 3)
    def forward(self, x, lms):
        x = self.conv1(x)
        x = self.head(x)
        xi = self.body[0](x)
        for i in range(1,self.N):
            xi = self.body[i](xi)
        y = x + xi
        y = self.upsample(y)
        y = y + self.skip_conv(lms)
        y = self.tail(y)
        return y

# 7. DetectionApp和main函数（de.py主界面和逻辑）
# ================================================================
class MicrospectraSRApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("显微高光谱图像超分辨率重建")
        self.setMinimumSize(1200, 800)
        self.current_model = None
        self.current_image = None
        self.model_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_labels = []
        self.init_ui()
        strip_inline_styles(self)
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        button_font = QFont()
        button_font.setPointSize(10)
        # 模型选择分组
        model_group = QFrame()
        model_group.setFrameStyle(QFrame.Panel | QFrame.Raised)
        model_layout = QVBoxLayout(model_group)
        self.select_model_btn = QPushButton("选择模型文件")
        self.select_model_btn.setFont(button_font)
        self.model_path_label = QLabel("未选择模型文件")
        self.model_path_label.setWordWrap(True)
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setFont(button_font)
        model_layout.addWidget(self.select_model_btn)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.load_model_btn)
        left_layout.addWidget(model_group)
        # 数据加载分组
        image_group = QFrame()
        image_group.setFrameStyle(QFrame.Panel | QFrame.Raised)
        image_layout = QVBoxLayout(image_group)
        self.load_image_btn = QPushButton("加载数据")
        self.load_image_btn.setFont(button_font)
        self.image_list = QListWidget()
        self.image_list.setMinimumHeight(200)
        image_layout.addWidget(self.load_image_btn)
        image_layout.addWidget(self.image_list)
        left_layout.addWidget(image_group)
        # 重建与保存分组
        action_group = QFrame()
        action_group.setFrameStyle(QFrame.Panel | QFrame.Raised)
        action_layout = QVBoxLayout(action_group)
        self.run_prediction_btn = QPushButton("开始重建")
        self.run_prediction_btn.setFont(button_font)
        self.visualize_btn = QPushButton("保存结果")
        self.visualize_btn.setFont(button_font)
        action_layout.addWidget(self.run_prediction_btn)
        action_layout.addWidget(self.visualize_btn)
        left_layout.addWidget(action_group)
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        main_layout.addWidget(left_panel)
        # 右侧结果显示区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.grid_layout = QGridLayout(scroll_content)
        self.grid_layout.setSpacing(10)
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area)
        main_layout.addWidget(right_panel)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 4)
        # 信号连接
        self.select_model_btn.clicked.connect(self.select_model_file)
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_image_btn.clicked.connect(self.load_image)
        self.run_prediction_btn.clicked.connect(self.run_prediction)
        self.visualize_btn.clicked.connect(self.visualize_results)
        self.load_model_btn.setEnabled(False)
        self.load_image_btn.setEnabled(False)
        self.run_prediction_btn.setEnabled(False)
        self.visualize_btn.setEnabled(False)
    def clear_results(self):
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)
        self.result_labels.clear()
    def add_result_image(self, image_data, title):
        frame = QFrame()
        frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        layout = QVBoxLayout(frame)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(200, 200)
        height, width = image_data.shape
        bytes_per_line = width
        normalized_data = ((image_data - image_data.min()) * 255 /
                           (image_data.max() - image_data.min())).astype(np.uint8)
        q_img = QImage(normalized_data.data, width, height,
                       bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            200, 200,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        image_label.setPixmap(scaled_pixmap)
        layout.addWidget(image_label)
        row = len(self.result_labels) // 4
        col = len(self.result_labels) % 4
        self.grid_layout.addWidget(frame, row, col)
        self.result_labels.append(image_label)
    def select_model_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择模型文件",
                "./model",
                "PyTorch模型文件 (*.pth);;所有文件 (*.*)"
            )
            if file_path:
                self.model_path = file_path
                self.model_path_label.setText(f"已选择: {os.path.basename(file_path)}")
                self.load_model_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"选择模型文件失败: {str(e)}")
    def load_model(self):
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
            return
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.current_model = MSDformer(
                n_subs=60,
                n_ovls=0,
                n_colors=60,
                scale=4,
                n_feats=240,
                n_DCTM=4
            )
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.current_model.load_state_dict(state_dict['model'])
            self.current_model.to(self.device).eval()
            self.progress_bar.setValue(100)
            self.load_image_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    def load_image(self):
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "选择数据",
                "./data",
                "Images (*.png *.jpg *.jpeg *.bmp *.mat)"
            )
            for file_path in file_paths:
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.UserRole, file_path)
                self.image_list.addItem(item)
                if self.image_list.count() == 1:
                    self.current_image = file_path
                if self.current_model:
                    self.run_prediction_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据加载失败: {str(e)}")
    def run_prediction(self):
        if not self.current_model:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        import datetime
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            test_set = HSTestData(image_path=self.current_image, dataset='Cho')
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
            output = []
            with torch.no_grad():
                for i, (ms, lms, gt, image_path) in enumerate(test_loader):
                    ms, lms = ms.to(self.device), lms.to(self.device)
                    y = self.current_model(ms, lms)
                    y = y.squeeze().cpu().numpy().transpose(1, 2, 0)
                    y = y[:gt.shape[2], :gt.shape[3], :]
                    output.append(y)
                    self.progress_bar.setValue(int((i + 1) / len(test_loader) * 100))
            # 统一保存到 ./result/microspectra_sr/当前批次/
            batch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_root = os.path.join(os.getcwd(), "result", "microspectra_sr", batch_time)
            os.makedirs(save_root, exist_ok=True)
            np.save(os.path.join(save_root, "MSDformer_x4.npy"), output)
            # 直接显示切片
            data = np.array(output)
            if len(data.shape) == 4:
                time_steps, height, width, depth = data.shape
                self.clear_results()
                for i in range(depth):
                    slice_data = data[0, :, :, i]
                    save_path = os.path.join(save_root, f'slice_{i:02d}.png')
                    count = 1
                    while os.path.exists(save_path):
                        save_path = os.path.join(save_root, f'slice_{i:02d}_{count}.png')
                        count += 1
                    plt.imsave(save_path, slice_data, cmap='gray')
                    self.add_result_image(slice_data, f"切片 {i + 1}")
                    self.progress_bar.setValue(int((i + 1) / depth * 100))
            self.visualize_btn.setEnabled(True)
            QMessageBox.information(self, "成功", f"处理结果已保存到 {save_root} 并已显示")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    def visualize_results(self):
        import datetime
        try:
            # 选择批次文件夹
            save_root = QFileDialog.getExistingDirectory(self, "选择结果批次文件夹", os.path.join(os.getcwd(), "result", "microspectra_sr"))
            if not save_root:
                return
            file_path = os.path.join(save_root, "MSDformer_x4.npy")
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "警告", "未找到.npy结果文件")
                return
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            data = np.load(file_path)
            if len(data.shape) != 4:
                QMessageBox.warning(self, "警告", f"结果数据不是4维，实际shape: {data.shape}")
                return
            time_steps, height, width, depth = data.shape
            self.clear_results()
            for i in range(depth):
                slice_data = data[0, :, :, i]
                save_path = os.path.join(save_root, f'slice_{i:02d}.png')
                count = 1
                while os.path.exists(save_path):
                    save_path = os.path.join(save_root, f'slice_{i:02d}_{count}.png')
                    count += 1
                plt.imsave(save_path, slice_data, cmap='gray')
                self.add_result_image(slice_data, f"切片 {i + 1}")
                self.progress_bar.setValue(int((i + 1) / depth * 100))
            QMessageBox.information(self, "成功", f"所有切片已保存到 {save_root} 并显示")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"可视化失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

def main():
    app = QApplication(sys.argv)
    window = MicrospectraSRApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()