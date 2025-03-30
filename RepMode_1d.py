# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 22:54:53 2025

@author: Mateo-drr
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

'''
Sample usage

# encoder
self.encoder_block1 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels, self.in_channels * self.mult_chan)
self.encoder_block2 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan, self.in_channels * self.mult_chan * 2)
self.encoder_block3 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 2, self.in_channels * self.mult_chan * 4)
self.encoder_block4 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 4, self.in_channels * self.mult_chan * 8)

# bottle
self.bottle_block = MoDESubNet2Conv(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 8, self.in_channels * self.mult_chan * 16)

# decoder
self.decoder_block4 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 16, self.in_channels * self.mult_chan * 8)
self.decoder_block3 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 8, self.in_channels * self.mult_chan * 4)
self.decoder_block2 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 4, self.in_channels * self.mult_chan * 2)
self.decoder_block1 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 2, self.in_channels * self.mult_chan)

# conv out
self.conv_out = MoDEConv(self.num_experts, self.num_tasks, self.mult_chan, self.out_channels, kernel_size=5, padding='same', conv_type='final')

def forward(self, x, t):
    # task embedding
    task_emb = self.one_hot_task_embedding(t)

    # encoding
    print(x.shape)
    x, x_skip1 = self.encoder_block1(x, task_emb)
    x, x_skip2 = self.encoder_block2(x, task_emb)
    x, x_skip3 = self.encoder_block3(x, task_emb)
    x, x_skip4 = self.encoder_block4(x, task_emb)
    
    # bottle
    x = self.bottle_block(x, task_emb)

    # decoding
    x = self.dropout_latent(x)
    x = self.decoder_block4(x, x_skip4, task_emb)
    x = self.decoder_block3(x, x_skip3, task_emb)
    x = self.decoder_block2(x, x_skip2, task_emb)
    x = self.decoder_block1(x, x_skip1, task_emb)
    outputs = self.conv_out(x, task_emb)
'''

def one_hot_task_embedding(self, task_id):
    N = task_id.shape[0]
    task_embedding = torch.zeros((N, self.num_tasks))
    for i in range(N):
        task_embedding[i, task_id[i]] = 1
    return task_embedding.to(self.device)

class MoDEEncoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv_more = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan)
        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv1d(out_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm1d(out_chan, affine=True),
            torch.nn.Mish(inplace=True),
        )

    def forward(self, x, t):
        x_skip = self.conv_more(x, t)
        x = self.conv_down(x_skip)
        return x, x_skip


class MoDEDecoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.convt = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm1d(out_chan, affine=True),
            torch.nn.Mish(inplace=True),
        )
        self.conv_less = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan)

    def forward(self, x, x_skip, t):
        x = self.convt(x)
        x_cat = torch.cat((x_skip, x), 1)  # concatenate
        x_cat = self.conv_less(x_cat, t)
        return x_cat


class MoDESubNet2Conv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, n_in, n_out):
        super().__init__()
        self.conv1 = MoDEConv(num_experts, num_tasks, n_in, n_out, kernel_size=5, padding='same')
        self.conv2 = MoDEConv(num_experts, num_tasks, n_out, n_out, kernel_size=5, padding='same')

    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        return x


class MoDEConv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan, kernel_size=5, stride=1, padding='same', conv_type='normal'):
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.stride = stride
        self.padding = padding

        self.expert_conv5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 5)
        self.expert_conv3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 3)
        self.expert_conv1x1_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg3x3_pool', self.gen_avgpool_kernel(3))
        self.expert_avg3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg5x5_pool', self.gen_avgpool_kernel(5))
        self.expert_avg5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)

        assert self.conv_type in ['normal', 'final']
        if self.conv_type == 'normal':
            self.subsequent_layer = torch.nn.Sequential(
                nn.InstanceNorm1d(out_chan, affine=True),
                torch.nn.Mish(inplace=True),
            )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(num_tasks, num_experts * self.out_chan, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)


    def gen_conv_kernel(self, Co, Ci, K):
        # For 1D convolution, kernel shape is (Co, Ci, K)
        weight = torch.nn.Parameter(torch.empty(Co, Ci, K))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode='fan_out')
        return weight

    def gen_avgpool_kernel(self, K):
        # For 1D convolution, kernel shape is (K)
        weight = torch.ones(K).mul(1.0 / K)
        return weight

    def trans_kernel(self, kernel, target_size):
        # For 1D convolution, padding is only applied to the last dimension
        # to always look at the token in question which is shifted to the end of 
        # the kernel, padding has to be applied only to the left
        #Wp = (target_size - kernel.shape[2]) // 2
        #return F.pad(kernel, [Wp, Wp])
        Wp = target_size - kernel.shape[2]
        return F.pad(kernel, [Wp, 0])

    def routing(self, g, N):
        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(self.expert_conv1x1_conv, self.kernel_size)
        
        # For 1D convolution, we use einsum with appropriate dimensions
        expert_avg3x3 = self.trans_kernel(
            torch.einsum('oiw,w->oiw', self.expert_avg3x3_conv, self.expert_avg3x3_pool),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum('oiw,w->oiw', self.expert_avg5x5_conv, self.expert_avg5x5_pool)

        weights = list()
        for n in range(N):
            weight_nth_sample = torch.einsum('oiw,o->oiw', expert_conv5x5, g[n, 0, :]) + \
                               torch.einsum('oiw,o->oiw', expert_conv3x3, g[n, 1, :]) + \
                               torch.einsum('oiw,o->oiw', expert_conv1x1, g[n, 2, :]) + \
                               torch.einsum('oiw,o->oiw', expert_avg3x3, g[n, 3, :]) + \
                               torch.einsum('oiw,o->oiw', expert_avg5x5, g[n, 4, :])
            weights.append(weight_nth_sample)
        weights = torch.stack(weights)

        return weights

    def forward(self, x, t):
        N = x.shape[0]  # batch size

        g = self.gate(t)  # [b, x out channels * experts]
        g = g.view((N, self.num_experts, self.out_chan))  # [b,experts,x out channels]
        g = self.softmax(g)

        w = self.routing(g, N)  # [b,x out chann, 1, 5] mix expert kernels

        padLeft=self.kernel_size-1
        x = x.transpose(1,2) #makes [b,seqlen,dmodel] to [b,dmodel,seqlen]
        # if self.training:
        #     y = list()
        #     for i in range(N):
        #         xpad = F.pad(x[i].unsqueeze(0), (padLeft,0), 'constant', 0)
        #         y.append(F.conv1d(xpad, w[i], bias=None, stride=1, padding=0))
        #     y = torch.cat(y, dim=0)
        # else:
        #     xpad = F.pad(x, (padLeft,0), 'constant', 0)
        #     y = F.conv1d(xpad, w[0], bias=None, stride=1, padding=0)

        '''
        processing everything as one batch
        '''

        # Pad the entire batch at once
        x_padded = F.pad(x, (padLeft, 0), "constant", 0)  # Shape: [48, 768, 387]
        
        # Create a single large batch for all examples
        x_batched = x_padded.view(1, -1, x_padded.shape[2])  # Shape: [1, 48*768, 387]
        
        # Reshape and concatenate all weights
        w_batched = w.view(N * self.out_chan, self.in_chan, self.kernel_size)  # Shape: [48*768, 768, 5]
        
        # Use grouped convolution
        y_batched = F.conv1d(
            x_batched,        # [1, 48*768, 387]
            w_batched,        # [48*768, 768, 5] 
            bias=None, 
            stride=1, 
            padding=0, 
            groups=N          # 48 groups
        )
        
        # Reshape result back to original batch format
        y = y_batched.view(N, self.out_chan, -1)  # Shape: [48, 768, 383]

        '''
        manual conv
        '''
        # # Pad the entire batch at once
        # x_padded = F.pad(x, (padLeft, 0), "constant", 0)  # Shape: [48, 768, 387]
        
        # # Extract unfold patches (simulate convolution)
        # patches = x_padded.unfold(2, self.kernel_size, 1)  # Shape: [48, 768, 383, 5]
        # patches = patches.permute(0, 2, 1, 3)  # Shape: [48, 383, 768, 5]
        # patches = patches.reshape(N, x.shape[2], self.in_chan * self.kernel_size)  # Shape: [48, 383, 768*5]
        
        # # Reshape weights for batch matrix multiplication
        # w_reshaped = w.view(N, self.out_chan, -1)  # Shape: [48, 768, 768*5]
        
        # # Perform batched matrix multiplication
        # y = torch.bmm(w_reshaped, patches.transpose(1, 2))  # Shape: [48, 768, 383]


        y = self.subsequent_layer(y)

        return y.transpose(1,2)