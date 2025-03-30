# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:54:17 2025

@author: Mateo-drr

Virtual Intelligence model
"""
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class InputEmbeddings(nn.Module):
    
    def __init__(self, dModel: int, vocabSize: int):
        
        super().__init__()
        self.dModel = dModel #size of each token
        self.vocabSize = vocabSize
        self.embedding = nn.Embedding(vocabSize, dModel)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dModel)

class PositionalEncoding(nn.Module):
    
    def __init__(self, dModel: int, seqLen: int, dropout: float):
        
        super().__init__()
        self.dModel = dModel
        self.seqLen = seqLen #size of sentence
        self.dropout = nn.Dropout(dropout)
        
        posEnc = torch.zeros(seqLen, dModel) 
        
        #the formula is a division of pos/denom
        pos = torch.arange(0, seqLen, dtype=torch.float).unsqueeze(1)
        #this denom formula is the same from the paper just with log for numstab
        denom = torch.exp(torch.arange(0, dModel, 2, dtype=torch.float) * (-math.log(10000.0) / dModel))
        
        #evens -> sin | odd -> cos
        posEnc[:,0::2] = torch.sin(pos*denom)
        posEnc[:,1::2] = torch.cos(pos*denom)
        
        posEnc = posEnc.unsqueeze(0) # [1, seqLen, dModel]
        
        #use this buffer thing to make the model save this
        self.register_buffer('posEnc', posEnc)
        
    def forward(self,x):
        x = x + (self.posEnc[:, :x.shape[1], :]).requires_grad_(False)
        # return self.dropout(x)
        return x
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, dModel: int, hidSize: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(dModel, hidSize)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidSize, dModel)
        self.actfunc = nn.GELU()#nn.Mish(inplace=True)#nn.ReLU(inplace=True)
        
    def forward(self,x):
        #[b,seqlen,dmodel]
        x = self.lin1(x)    
        x = self.dropout(self.actfunc(x))
        x = self.lin2(x)
        return x
    
class DyT(nn.Module):
    '''
    https://jiachenzhu.github.io/DyT/
    '''
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
    
class Residual(nn.Module):
    
    def __init__(self, dModel, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout()
        self.norm = nn.LayerNorm(dModel)#DyT(num_features=dModel)
        
    def forward(self, x, prevLayerX):
        # return x + self.dropout(self.norm(prevLayerX(x))) 
        #his implementation puts norm first
        return x + self.dropout(prevLayerX(self.norm(x))) #norm before ff
        # return x + self.dropout(self.norm(prevLayerX))
    
class LastLayer(nn.Module):
    
    def __init__(self, dModel: int, vocabSize: int):
        super().__init__()
        self.lin = nn.Linear(dModel, vocabSize)
    
    def forward(self,x):
        #return torch.log_softmax(self.lin(x), dim=-1)
        return self.lin(x)
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, dModel: int, numheads: int, dropout: float):
        super().__init__()
        assert dModel % numheads == 0, 'dModel is not divisible by numheads'
        
        self.dModel = dModel
        self.numheads = numheads
        #dk is dmodel / numheads aka h
        self.dk = dModel//numheads
        
        self.wq = nn.Linear(dModel, dModel) #query mux
        self.wk = nn.Linear(dModel, dModel) #key mux
        self.wv = nn.Linear(dModel, dModel) #value mux
        
        self.wo = nn.Linear(dModel, dModel) #wo
        self.dropout = nn.Dropout(dropout)
    
    def attentionCalc(self, query, key, value, mask):
        dk = query.shape[-1]
        
        #swap the last two dimensions
        attentionScores = (query @ key.transpose(-2, -1)) / math.sqrt(dk)
        #the output of [b, numheads, seqlen, dk] @ [b, numheads, dk, seqlen]
        #is [b,numheads,seqlen,seqlen]
        
        #apply mask if available
        if mask is not None:
            min_value = torch.finfo(attentionScores.dtype).min
            attentionScores.masked_fill_(mask == 0, min_value) #mask with -1e9 so softmax will put as zero
        
        attentionScores = attentionScores.softmax(dim = -1)
        if self.dropout is not None:
            attentionScores = self.dropout(attentionScores)
            
        #we go back to the og shape [b,numheads,seqlen,dk]
        return (attentionScores @ value), attentionScores
    
    def forward(self,q,k,v,mask):
        #all these dont change shape [b, seqlen, dmodel]
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        
        #reshape into [b, seqlen, numheads, dk] and then into [b, numheads, seqlen, dk]
        query = query.view(query.shape[0], query.shape[1], self.numheads, self.dk)
        query = query.transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.numheads, self.dk)
        key = key.transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.numheads, self.dk)
        value = value.transpose(1,2)
        
        x, self.attentionScores = self.attentionCalc(query, key, value, mask)
        
        x = x.transpose(1,2) #from [b,numheads,seqlen,dk] to shape [b,seqlen,numheads,dk]
        x = x.reshape(x.shape[0], -1, self.numheads * self.dk) # now to [b, seqlen, dmodel] 
        #view has some memory issue here so using reshape
        
        x = self.wo(x) #no change in shape
        
        return x
    
class DecoderBlock(nn.Module):
    
    def __init__(self, dModel, heads, hidSize, dropout):
        super().__init__()
        
        self.maskedAttention = nn.MultiheadAttention(dModel,
                                                     heads,
                                                     dropout=dropout,
                                                     add_bias_kv=False,
                                                     batch_first=True)
        
        # self.maskedAttention2 = nn.MultiheadAttention(dModel,
        #                                             heads,
        #                                             dropout=dropout,
        #                                             batch_first=True)
        
        self.feedForward = FeedForwardBlock(dModel, hidSize, dropout)
        
        self.residual = nn.ModuleList([Residual(dModel,dropout),
                                       # Residual(dModel,dropout),
                                       Residual(dModel,dropout)])
    
    def forward(self,x,padMask, cauMask):
        
        
        x = self.residual[0](x, lambda x: self.maskedAttention(query=x,
                                                               key=x,
                                                               value=x,
                                                               key_padding_mask=padMask,
                                                               attn_mask=cauMask,
                                                               is_causal=True)[0])
        
        # x = self.residual[1](x, lambda x: self.maskedAttention(query=x,
        #                                                    key=x,
        #                                                    value=x,
        #                                                    key_padding_mask=padMask,
        #                                                    attn_mask=cauMask,
        #                                                    is_causal=True)[0])
        
        x = self.residual[1](x, self.feedForward)
        
        return x
    
class DecoderBlockCstm(nn.Module):
    
    def __init__(self, dModel, heads, hidSize, dropout):
        super().__init__()
        
        
        self.maskedAttention = MultiHeadAttentionBlock(dModel,
                                                           heads,
                                                           dropout)
        
        self.feedForward = FeedForwardBlock(dModel, hidSize, dropout)
        
        self.residual = nn.ModuleList([Residual(dModel,dropout),
                                       # Residual(dModel,dropout),
                                       Residual(dModel,dropout)])
        
    def combine_masks(self, padding_mask, causal_mask):
        # padding_mask shape: [batch, seqlen]
        # causal_mask shape: [seqlen, seqlen]
        #flip masks from pytorch true=mask approach
        padding_mask,causal_mask = ~padding_mask,~causal_mask
        
        # 1. Convert boolean masks to binary (0 and 1)
        padding_mask = padding_mask.float()
        causal_mask = causal_mask.float()
        
        # 2. Expand padding mask to [batch, 1, 1, seqlen]
        # This will broadcast against the last dimension of attention scores
        expanded_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        
        # 3. Expand causal mask to [1, 1, seqlen, seqlen]
        # This will broadcast against the batch and numheads dimensions
        expanded_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # 4. Combine masks - need to use element-wise multiplication since both
        # must be satisfied (both masks have 1 for valid positions)
        combined_mask = expanded_padding_mask * expanded_causal_mask
        
        return combined_mask    
    
    def forward(self,x,padMask, cauMask):
        
        srcMask = self.combine_masks(padMask, cauMask)
        x = self.residual[0](x, lambda x: self.maskedAttention(q=x,k=x,v=x,
                                                                 mask=srcMask))
        
        # x = self.residual[1](x, lambda x: self.maskedAttention(query=x,
        #                                                    key=x,
        #                                                    value=x,
        #                                                    key_padding_mask=padMask,
        #                                                    attn_mask=cauMask,
        #                                                    is_causal=True)[0])
        
        x = self.residual[1](x, self.feedForward)
        
        return x
    
class VImodel(nn.Module):
    
    def __init__(self, dModel, seqLen, heads, layers, dropout, hidSize, vocabSize):
        super().__init__()
        
        self.embed = InputEmbeddings(dModel, vocabSize)
        
        self.posEnc = PositionalEncoding(dModel, seqLen, dropout)
        
        self.decoders = nn.ModuleList([
            DecoderBlock(dModel, heads, hidSize, dropout)
            for _ in range(layers)
        ])
        
        self.lastlayer = LastLayer(dModel, vocabSize)
        
        self._initialize_weights()

    def _initialize_weights(self):
        print('Weights initialized')
        for param in self.parameters():
            if param.dim() > 1:  
                nn.init.xavier_normal_(param)
                #nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
        
    # def _initialize_weights(self):
    #     print('Weights initialized')
    #     for param in self.parameters():
    #         if param.dim() > 1:  # Check if the parameter is a weight (not a bias)
    #             # Normal initialization (mean=0, std=0.02, similar to tf.random_normal_initializer)
    #             nn.init.normal_(param, mean=0.0, std=0.02)  # Initialize weights using normal distribution
    #         else:
    #             # Initialize bias with zeros (like tf.constant_initializer(0))
    #             nn.init.constant_(param, 0)  # Initialize bias to zero
        
    def forward(self, x, padMask, cauMask):
        
        x = self.embed(x)
        x = self.posEnc(x)
        
        for decoder in self.decoders:
            x = decoder(x, padMask, cauMask)
            
        x = self.lastlayer(x)
        
        return x
    
        
class VImodelCstm(nn.Module):
    
    def __init__(self, dModel, seqLen, heads, layers, dropout, hidSize, vocabSize):
        super().__init__()
        
        self.embed = InputEmbeddings(dModel, vocabSize)
        
        self.posEnc = PositionalEncoding(dModel, seqLen, dropout)
        
        self.decoders = nn.ModuleList([
            DecoderBlockCstm(dModel, heads, hidSize, dropout)
            for _ in range(layers)
        ])
        
        self.lastlayer = LastLayer(dModel, vocabSize)
        
        self._initialize_weights()

    def _initialize_weights(self):
        print('Weights initialized')
        for param in self.parameters():
            if param.dim() > 1:  
                nn.init.xavier_normal_(param)
                #nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
        
    # def _initialize_weights(self):
    #     print('Weights initialized')
    #     for param in self.parameters():
    #         if param.dim() > 1:  # Check if the parameter is a weight (not a bias)
    #             # Normal initialization (mean=0, std=0.02, similar to tf.random_normal_initializer)
    #             nn.init.normal_(param, mean=0.0, std=0.02)  # Initialize weights using normal distribution
    #         else:
    #             # Initialize bias with zeros (like tf.constant_initializer(0))
    #             nn.init.constant_(param, 0)  # Initialize bias to zero
        
    def forward(self, x, padMask, cauMask):
        
        x = self.embed(x)
        x = self.posEnc(x)
        
        for decoder in self.decoders:
            x = decoder(x, padMask, cauMask)
            
        x = self.lastlayer(x)
        
        return x        

'''
Repmode version
'''
from RepMode_1d import MoDESubNet2Conv, one_hot_task_embedding, MoDEConv

class MultiHeadAttentionBlockRepMode(nn.Module):
    
    def __init__(self, dModel: int, numheads: int, dropout: float):
        super().__init__()
        assert dModel % numheads == 0, 'dModel is not divisible by numheads'
        
        self.dModel = dModel
        self.numheads = numheads
        #dk is dmodel / numheads aka h
        self.dk = dModel//numheads
        
        self.num_tasks=1
        self.device='cuda'
        
        self.wq = MoDEConv(num_experts=5,
                            num_tasks=self.num_tasks,
                            in_chan=dModel,
                            out_chan=dModel)
        #self.wq = nn.Linear(dModel, dModel) #query mux
        
        self.wk = MoDEConv(num_experts=5,
                            num_tasks=self.num_tasks,
                            in_chan=dModel,
                            out_chan=dModel)
        #self.wk = nn.Linear(dModel, dModel) #key mux
        
        self.wv = MoDEConv(num_experts=5,
                            num_tasks=self.num_tasks,
                            in_chan=dModel,
                            out_chan=dModel)
        #self.wv = nn.Linear(dModel, dModel) #value mux
        
        self.wo = MoDEConv(num_experts=5,
                            num_tasks=self.num_tasks,
                            in_chan=dModel,
                            out_chan=dModel,
                            conv_type='final')
        # self.wo = nn.Linear(dModel, dModel) #wo
        self.dropout = nn.Dropout(dropout)
    
    def attentionCalc(self, query, key, value, mask):
        dk = query.shape[-1]
        
        #swap the last two dimensions
        attentionScores = (query @ key.transpose(-2, -1)) / math.sqrt(dk)
        #the output of [b, numheads, seqlen, dk] @ [b, numheads, dk, seqlen]
        #is [b,numheads,seqlen,seqlen]
        
        #apply mask if available
        if mask is not None:
            min_value = torch.finfo(attentionScores.dtype).min
            attentionScores.masked_fill_(mask == 0, min_value) #mask with -1e9 so softmax will put as zero
        
        attentionScores = attentionScores.softmax(dim = -1)
        if self.dropout is not None:
            attentionScores = self.dropout(attentionScores)
            
        #we go back to the og shape [b,numheads,seqlen,dk]
        return (attentionScores @ value), attentionScores
    
    def forward(self,q,k,v,mask):
        #all these dont change shape [b, seqlen, dmodel]
        t = one_hot_task_embedding(self,torch.zeros(q.shape[0], dtype=int))
        query = self.wq(q, t)
        key = self.wk(k, t)
        value = self.wv(v, t)
        
        #reshape into [b, seqlen, numheads, dk] and then into [b, numheads, seqlen, dk]
        query = query.view(query.shape[0], query.shape[1], self.numheads, self.dk)
        query = query.transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.numheads, self.dk)
        key = key.transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.numheads, self.dk)
        value = value.transpose(1,2)
        
        x, self.attentionScores = self.attentionCalc(query, key, value, mask)
        # Flash Attention implementation
        # This is much more memory-efficient and faster on modern GPUs
        # x = torch.nn.functional.scaled_dot_product_attention(
        #     query, key, value,
        #     attn_mask=mask, #torch mask doesnt work + didnt see much improvement
        #     dropout_p=self.dropout.p if self.dropout is not None else 0.0,
        #     is_causal=False  # Set to True if using causal masking
        # )
        # # Note: Flash Attention doesn't return attention scores directly
        # # If you need attention scores for visualization, you'd need to compute them separately
        # self.attentionScores = None
        
        x = x.transpose(1,2) #from [b,numheads,seqlen,dk] to shape [b,seqlen,numheads,dk]
        x = x.reshape(x.shape[0], -1, self.numheads * self.dk) # now to [b, seqlen, dmodel] 
        #view has some memory issue here so using reshape
        
        x = self.wo(x,t) #no change in shape
        
        return x

class CausalConv(nn.Module):
    def __init__(self, dModel, kernel):
        super().__init__()
        
        self.cauconv = nn.Conv1d(dModel, dModel, kernel_size=kernel,
                                 padding=0, stride=1)
        self.k=kernel
        
    def forward(self,x):
        #assuming input is shape [b,seqlen, dmodel]
        x = x.transpose(1,2)
        
        return self.cauconv(F.pad(x, (self.k-1, 0), "constant", 0)).transpose(1,2)
        

class DecoderBlockRepMode(nn.Module):
    
    def __init__(self, dModel, heads, hidSize, dropout):
        super().__init__()
        
        
        self.maskedAttention = MultiHeadAttentionBlockRepMode(dModel,
                                                           heads,
                                                           dropout)
        self.num_tasks=1
        self.device='cuda'
        #self.feedForward = FeedForwardBlock(dModel, hidSize, dropout)
        self.feedForward = CausalConv(dModel, kernel=7)
        # self.feedForward = MoDEConv(num_experts=5,
        #                             num_tasks=self.num_tasks,
        #                             in_chan=dModel,
        #                             out_chan=dModel,
        #                             conv_type='final')
        # MoDESubNet2Conv(num_experts=5,
        #                                    num_tasks=self.num_tasks,
        #                                    n_in=dModel,
        #                                    n_out=dModel)
        
        self.residual = nn.ModuleList([Residual(dModel,dropout),
                                       # Residual(dModel,dropout),
                                       Residual(dModel,dropout)])
        
    def combine_masks(self, padding_mask, causal_mask):
        # padding_mask shape: [batch, seqlen]
        # causal_mask shape: [seqlen, seqlen]
        #flip masks from pytorch true=mask approach
        padding_mask,causal_mask = ~padding_mask,~causal_mask
        
        # 1. Convert boolean masks to binary (0 and 1)
        padding_mask = padding_mask.float()
        causal_mask = causal_mask.float()
        
        # 2. Expand padding mask to [batch, 1, 1, seqlen]
        # This will broadcast against the last dimension of attention scores
        expanded_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        
        # 3. Expand causal mask to [1, 1, seqlen, seqlen]
        # This will broadcast against the batch and numheads dimensions
        expanded_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # 4. Combine masks - need to use element-wise multiplication since both
        # must be satisfied (both masks have 1 for valid positions)
        combined_mask = expanded_padding_mask * expanded_causal_mask
        
        return combined_mask    
    
    def forward(self,x,padMask, cauMask):
        
        srcMask = self.combine_masks(padMask, cauMask)
        x = self.residual[0](x, lambda x: self.maskedAttention(q=x,k=x,v=x,
                                                                 mask=srcMask))
        
        # x = self.residual[1](x, lambda x: self.maskedAttention(query=x,
        #                                                    key=x,
        #                                                    value=x,
        #                                                    key_padding_mask=padMask,
        #                                                    attn_mask=cauMask,
        #                                                    is_causal=True)[0])
        
        # x = self.residual[1](x, lambda x: self.feedForward(x,
        #                                                    one_hot_task_embedding(self,torch.zeros(x.shape[0], dtype=int))))
        x = self.residual[1](x, self.feedForward)
        
        return x
        
class VImodelRepMode(nn.Module):
    
    def __init__(self, dModel, seqLen, heads, layers, dropout, hidSize, vocabSize):
        super().__init__()
        
        self.embed = InputEmbeddings(dModel, vocabSize)
        
        self.posEnc = PositionalEncoding(dModel, seqLen, dropout)
        
        self.decoders = nn.ModuleList([
            DecoderBlockRepMode(dModel, heads, hidSize, dropout)
            for _ in range(layers)
        ])
        
        self.lastlayer = LastLayer(dModel, vocabSize)
        
        self._initialize_weights()

    def _initialize_weights(self):
        print('Weights initialized')
        for param in self.parameters():
            if param.dim() > 1:  
                nn.init.xavier_normal_(param)
                #nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
        
    # def _initialize_weights(self):
    #     print('Weights initialized')
    #     for param in self.parameters():
    #         if param.dim() > 1:  # Check if the parameter is a weight (not a bias)
    #             # Normal initialization (mean=0, std=0.02, similar to tf.random_normal_initializer)
    #             nn.init.normal_(param, mean=0.0, std=0.02)  # Initialize weights using normal distribution
    #         else:
    #             # Initialize bias with zeros (like tf.constant_initializer(0))
    #             nn.init.constant_(param, 0)  # Initialize bias to zero
        
    def forward(self, x, padMask, cauMask):
        
        x = self.embed(x)
        x = self.posEnc(x)
        
        for decoder in self.decoders:
            x = decoder(x, padMask, cauMask)
            
        x = self.lastlayer(x)
        
        return x    
        
'''
Semantic understanding encoder
'''        





























