from math import ceil

import torch
from torch import nn

from revin import RevIN


class TLDC(nn.Module):
    def __init__(self, var_dim, emb_dim, size, kernel_size, dilation):
        super(TLDC, self).__init__()
        self.var_dim = var_dim
        self.emb_dim = emb_dim
        self.size = size
        self.dwconv = nn.Conv1d(var_dim * emb_dim, var_dim * emb_dim, 2*dilation - 1, padding="same", groups=var_dim * emb_dim, padding_mode="circular")
        self.dwdconv = nn.Conv1d(var_dim * emb_dim, var_dim * emb_dim, ceil(kernel_size/dilation), padding="same", groups=var_dim * emb_dim, dilation=dilation, padding_mode="circular")

    def forward(self, x):
        local = self.dwconv(x.reshape(x.size(0), self.var_dim * self.emb_dim, -1))
        expanded = self.dwdconv(local)
        return (local + expanded).reshape(x.size(0), self.var_dim, self.emb_dim, self.size)


class IVGC(nn.Module):
    def __init__(self, var_dim, emb_dim, size):
        super(IVGC, self).__init__()
        self.window_size = 4
        self.var_dim = var_dim
        self.emb_dim = emb_dim
        self.size = size
        self.npad = 0
        if self.size % self.window_size != 0:
            self.npad = self.window_size - (self.size % self.window_size)
        self.conv_standard = nn.Conv1d(self.var_dim * (self.size + self.npad), self.var_dim * (self.size + self.npad),
                                       1, groups=(size + self.npad) // self.window_size)
        self.initial_pad = nn.ConstantPad3d((0, self.npad, 0, 0, 0, 0), 0)
        self.pad = nn.ConstantPad3d((self.window_size // 2, self.window_size // 2 + self.npad, 0, 0, 0, 0), 0)
        self.conv_ht = nn.Conv1d(self.var_dim * (self.size + self.window_size + self.npad),
                                 self.var_dim * (self.size + self.window_size + self.npad), 1,
                                 groups=(size + self.window_size + self.npad) // self.window_size)
        self.conv_out = nn.Conv1d(self.var_dim * (self.size + self.npad), self.var_dim * (self.size + self.npad), 1,
                                  groups=(size + self.npad) // self.window_size)

    def forward(self, x):
        padded = self.initial_pad(x)
        s = self.conv_standard(padded.reshape(x.size(0), self.var_dim * (self.size + self.npad), self.emb_dim))
        ht = self.conv_ht(
            self.pad(x).reshape(x.size(0), self.var_dim * (self.size + self.window_size + self.npad), self.emb_dim))[:,
             :self.var_dim * (self.size + self.npad)]
        return self.conv_out(s + ht)[:, :self.var_dim * self.size].reshape_as(x)


class GTVA(nn.Module):
    def __init__(self, var_dim, emb_dim, size):
        super(GTVA, self).__init__()
        self.var_dim = var_dim
        self.emb_dim = emb_dim
        self.size = size
        self.reduction = 16
        self.temporal_pool = nn.AvgPool1d(kernel_size=self.var_dim)
        self.variable_pool = nn.AvgPool1d(kernel_size=self.size)
        reduced_temporal = self.size * self.emb_dim // self.reduction
        reduced_variable = self.var_dim * self.emb_dim // self.reduction
        self.temporal_attn = nn.Sequential(nn.Linear(self.size * self.emb_dim, reduced_temporal), nn.ReLU(),
                                           nn.Linear(reduced_temporal, self.size * self.emb_dim), nn.Sigmoid())
        self.variable_attn = nn.Sequential(nn.Linear(self.var_dim * self.emb_dim, reduced_variable), nn.ReLU(),
                                           nn.Linear(reduced_variable, self.var_dim * self.emb_dim), nn.Sigmoid())

    def forward(self, x):
        pooled_temporal = self.temporal_pool(x.reshape(x.size(0), self.size * self.emb_dim, self.var_dim)).squeeze(-1)
        pooled_variable = self.variable_pool(x.reshape(x.size(0), self.var_dim * self.emb_dim, self.size)).squeeze(-1)
        temp_attn = self.temporal_attn(pooled_temporal).reshape(x.size(0), 1, self.emb_dim, self.size)
        var_attn = self.variable_attn(pooled_variable).reshape(x.size(0), self.var_dim, self.emb_dim, 1)
        return nn.functional.sigmoid(temp_attn * var_attn * x)


class Block(nn.Module):
    def __init__(self, var_dim, emb_dim, size):
        super(Block, self).__init__()
        self.var_dim = var_dim
        self.emb_dim = emb_dim
        self.size = size
        self.tldc = TLDC(var_dim, emb_dim, size, 55, 5)
        self.ivgc = IVGC(var_dim, emb_dim, size)
        self.gtva = GTVA(var_dim, emb_dim, size)

    def forward(self, x):
        t = self.tldc(x)
        t = self.ivgc(t)
        t = self.gtva(t)
        return t * x


class Head(nn.Module):
    def __init__(self, var_dim, emb_dim, size, out_len):
        super(Head, self).__init__()
        self.var_dim = var_dim
        self.emb_dim = emb_dim
        self.size = size
        self.heads = nn.ModuleList()
        self.out_len = out_len
        for i in range(self.var_dim):
            self.heads.append(nn.Sequential(nn.Flatten(start_dim=-2), nn.Linear(emb_dim * size, out_len)))

    def forward(self, x):
        outs = []
        for i in range(self.var_dim):
            outs.append(self.heads[i](x[:, i, :, :]))
        return torch.stack(outs, dim=1)


class EfficaNet(nn.Module):
    def __init__(self, embedding_dim, var_dim, size, output_dim, blocks):
        super(EfficaNet, self).__init__()
        patching_kernel = 8
        patching_stride = 4
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.var_dim = var_dim
        self.size_orig = size
        self.size = (size - patching_kernel) // patching_stride + 1
        self.stem = nn.Conv1d(1, self.embedding_dim, patching_kernel, stride=patching_stride)
        self.blocks = nn.Sequential(*[Block(self.var_dim, self.embedding_dim, self.size) for _ in range(blocks)])
        self.revin = RevIN(var_dim)
        self.head = Head(self.var_dim, self.embedding_dim, self.size, output_dim[0])

    def forward(self, x):
        data = self.revin(x, "norm").permute(0, 2, 1).unsqueeze(-2)
        data = data.reshape(x.size(0) * self.var_dim, 1, self.size_orig)
        patched = self.stem(data)
        patched = patched.reshape(x.size(0), self.var_dim, self.embedding_dim, self.size)
        processed = self.blocks(patched)
        return self.revin(self.head(processed).permute(0, 2, 1), "denorm")
