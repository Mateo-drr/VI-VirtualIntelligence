# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:54:17 2025

@author: Mateo-drr

Virtual Intelligence model
"""
import torch
import torch.nn as nn
import math

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
        return self.dropout(x)
    
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
        self.dropout = nn.Dropout()
        self.norm = nn.LayerNorm(dModel)
        
    def forward(self, x, prevLayerX):
        #return x + self.dropout(self.norm(prevLayer(x))) 
        #his implementation puts norm first
        return x + self.dropout(prevLayerX(self.norm(x)))
        # return x + self.dropout(self.norm(prevLayerX))
    
class LastLayer(nn.Module):
    
    def __init__(self, dModel: int, vocabSize: int):
        super().__init__()
        self.lin = nn.Linear(dModel, vocabSize)
    
    def forward(self,x):
        return torch.log_softmax(self.lin(x), dim=-1)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, dModel, heads, hidSize, dropout):
        super().__init__()
        self.maskedAttention = nn.MultiheadAttention(dModel,
                                                     heads,
                                                     dropout=dropout,
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
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
        
        
    def forward(self, x, padMask, cauMask):
        
        x = self.embed(x)
        x = self.posEnc(x)
        
        for decoder in self.decoders:
            x = decoder(x, padMask, cauMask)
            
        x = self.lastlayer(x)
        
        return x
    
        
        
        
        
        
        