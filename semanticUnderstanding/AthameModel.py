# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:46:45 2025

@author: Mateo-drr
"""

import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import random

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
        self.actfunc = nn.Mish(inplace=True)#nn.ReLU(inplace=True)
        
    def forward(self,x):
        #[b,seqlen,dmodel]
        x = self.lin1(x)    
        x = self.dropout(self.actfunc(x))
        x = self.lin2(x)
        return x
    
class Residual(nn.Module):
    
    def __init__(self, dModel, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
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
    
class EncoderBlockCstm(nn.Module):
    
    def __init__(self, dModel, heads, hidSize, dropout):
        super().__init__()
        
        
        self.maskedAttention = MultiHeadAttentionBlock(dModel,
                                                           heads,
                                                           dropout)
        
        self.feedForward = FeedForwardBlock(dModel, hidSize, dropout)
        
        self.residual = nn.ModuleList([Residual(dModel,dropout),
                                       # Residual(dModel,dropout),
                                       Residual(dModel,dropout)])
    
    def forward(self,x,padMask):
        
        srcMask = ~padMask.unsqueeze(1).unsqueeze(2)
        x = self.residual[0](x, lambda x: self.maskedAttention(q=x,k=x,v=x,
                                                                 mask=srcMask))
        
        x = self.residual[1](x, self.feedForward)
        
        return x
    
def combine_masks(padding_mask, causal_mask):
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
    
class DecoderBlockCstm(nn.Module):
    
    def __init__(self, dModel, heads, hidSize, dropout):
        super().__init__()
        
        
        self.maskedAttention = MultiHeadAttentionBlock(dModel,
                                                           heads,
                                                           dropout)
        
        self.feedForward = FeedForwardBlock(dModel, hidSize, dropout)
        
        self.residual = nn.ModuleList([Residual(dModel,dropout),
                                       Residual(dModel,dropout),
                                       Residual(dModel,dropout)])
    
    def forward(self,x, encOut, padMaskT, cauMaskT):
        
        
        
        if x is None: #dont run self attention in inference
            crossMask=None
        else:
            tgtMask = combine_masks(padMaskT, cauMaskT)
            crossMask = ~cauMaskT.unsqueeze(0).unsqueeze(0)
            #run self attention
            xself = self.residual[0](x, lambda x: self.maskedAttention(q=x,k=x,v=x,
                                                                 mask=tgtMask))
        if not self.training or random.random() < 0.5:
            #Change cross-attention to self-attention during inference
            xself = encOut #replace the decoder data with encoder output
            
        x = self.residual[1](xself, lambda x: self.maskedAttention(q=xself,
                                                           k=encOut,
                                                           v=encOut,
                                                           mask=crossMask))

        x = self.residual[2](x, self.feedForward)
        
        return x,xself
    
class AdaptiveProjection(nn.Module):
    def __init__(self, dModel, tgtSeqlen=None):
        super().__init__()
        self.model_dim = dModel
        self.tgtSeqlen = tgtSeqlen
        
        # Adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=tgtSeqlen) if tgtSeqlen else None
        # Projection layer
        self.projection = nn.Sequential(nn.Linear(dModel, dModel*2),
                                        nn.Mish(inplace=True),
                                        nn.Linear(dModel*2, dModel)
                                        )
    
    def forward(self, x, tgtSeqlen=None):
        """
        x: [batch_size, seqlen, hidsize]
        tgtseqlen: optional target sequence length
        """
        # Transpose for pooling
        x_transposed = x.transpose(1, 2)
        
        # Use target_len if provided, otherwise use self.target_seq_len
        tgtSeqlen = tgtSeqlen or self.tgtSeqlen
        
        # Adaptive pooling to target length
        pooled = F.adaptive_avg_pool1d(x_transposed, output_size=tgtSeqlen)
        
        # Transpose back
        pooled = pooled.transpose(1, 2)
        
        # Apply projection
        projected = self.projection(pooled)
        
        return projected    
    
class Athame(nn.Module):
    
    def __init__(self, dModelDef, dModelWrd,
                 seqLenDef, seqLenWrd,
                 heads, layers, dropout, hidSize, srcVsize: int, tgtVsize: int):
        super().__init__()
        
        self.embedEnc = InputEmbeddings(dModelWrd, srcVsize)
        self.embedDec = InputEmbeddings(dModelDef, tgtVsize)
        self.posEncEnc = PositionalEncoding(dModelWrd, seqLenWrd, dropout)
        self.posEncDec = PositionalEncoding(dModelDef, seqLenDef, dropout)
        
        self.encoders = nn.ModuleList([
            EncoderBlockCstm(dModelWrd, heads, hidSize, dropout)
            for _ in range(layers)
        ])
        
        self.project = AdaptiveProjection(dModelDef, seqLenDef)
        
        self.decoders = nn.ModuleList([
            DecoderBlockCstm(dModelDef, heads, hidSize, dropout)
            for _ in range(layers)
        ])
        
        self.lastlayer = LastLayer(dModelDef, tgtVsize)
        
        self._initialize_weights()

    def _initialize_weights(self):
        print('Weights initialized')
        for param in self.parameters():
            if param.dim() > 1:  
                nn.init.xavier_normal_(param)
    
    def encode(self, x, padMaskS):
        x = self.embedEnc(x)
        x = self.posEncEnc(x)
        
        for encoder in self.encoders:
            x = encoder(x, padMaskS)
            
        return x
    
    def decode(self, xenc, xtgt, padMaskT, cauMaskT):
        if xtgt is not None:
            xtgt = self.embedDec(xtgt)
            xtgt = self.posEncDec(xtgt)
        
        #project encoder output into decoder shape
        xenc = self.project(xenc)
        
        selfattdef=[]
        for decoder in self.decoders:
            x,xself = decoder(xtgt, xenc, padMaskT, cauMaskT)
            selfattdef.append(xself)
            
        x = self.lastlayer(x)
            
        return x, selfattdef, xenc        
    
    def forward(self, x, padMaskS, cauMaskS, tgt, padMaskT, cauMaskT):
        
        x = self.encode(x, padMaskS)
        
        x, seldattdef, xenc = self.decode(x, tgt, padMaskT, cauMaskT)
        
        return x        