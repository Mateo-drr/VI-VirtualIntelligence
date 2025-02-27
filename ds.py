# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:48:59 2025

@author: Mateo-drr
"""

from torch.utils.data import Dataset, random_split
import torch
import copy

def splitDataset(cropped, tokenAtron, config, split=0.9):
    #create split
    train_ds_size = int(split * len(cropped))
    valid_ds_size = len(cropped) - train_ds_size
    train_ds_raw, valid_ds_raw = random_split(cropped, [train_ds_size, valid_ds_size])

    #build ds
    train_ds = CustomDataset(train_ds_raw, tokenAtron, config.seq_len)   
    valid_ds = CustomDataset(valid_ds_raw, tokenAtron, config.seq_len)   
    
    return train_ds, valid_ds

class CustomDataset(Dataset):

    def __init__(self, data, tokenAtron, seqLen):
        super().__init__()
        self.data = copy.deepcopy(data)
        self.tokenAtron = tokenAtron
        self.seqLen = seqLen
        
        self.bosToken = torch.tensor([tokenAtron.token_to_id('[BOS]')], dtype=torch.int64)
        self.eosToken = torch.tensor([tokenAtron.token_to_id('[EOS]')], dtype=torch.int64)
        self.padToken = torch.tensor([tokenAtron.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):    
    #TAKE ONE ITEM FROM THE DATASET
        data = self.data[idx]
        
        while True:
            tknData = self.tokenAtron.encode(data).ids
    
            padding = self.seqLen - len(tknData) - 2 #both bos and eos
    
            #check if seq len is enough
            if padding < 0:
                # print(data[-5], len(tknData), self.seqLen)
                data = data[:-1]
                # raise ValueError('Not enough tokens: sentece is longer than limit')
            else:
                break
                
            
            
        #add special tokens
        tknData = torch.cat([self.bosToken,
                              torch.tensor(tknData,dtype=torch.int64),
                              self.eosToken,
                              torch.tensor([self.padToken] * padding, dtype=torch.int64)])

        #should have the seqLen
        assert tknData.size(0) == self.seqLen
        
        #need to shift text ex:
        # input =  ["the", "dog", "is"]
        # target = ["dog", "is", "cute"]
        indata = tknData[:-1]
        target = tknData[1:]
        
        
        #the encoder mask only needs to mask pad tokens
        maskPad = (indata != self.padToken).int()
        maskCau = causalMask(indata.size(0))
        
        #mask tokens are True
        maskPad,maskCau = ~maskPad.bool(),~maskCau.bool()
        
        return {'data': tknData, #[seqLen]
                'indata': indata,
                'target': target,
                'maskPad': maskPad,
                'maskCau': maskCau,
                'raw': data
                }
            
def causalMask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0









