# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:48:59 2025

@author: Mateo-drr
"""

from torch.utils.data import Dataset, random_split
import torch
import copy

class CustomDataset(Dataset):

    def __init__(self, data, tokenAtron, seqLen, wordLen):
        super().__init__()
        self.data = copy.deepcopy(data)
        self.tokenAtron = tokenAtron
        self.seqLen = seqLen
        self.wordLen = wordLen
        
        self.bosToken = torch.tensor([tokenAtron.token_to_id('[BOS]')], dtype=torch.int64)
        self.eosToken = torch.tensor([tokenAtron.token_to_id('[EOS]')], dtype=torch.int64)
        self.padToken = torch.tensor([tokenAtron.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):    
    #TAKE ONE ITEM FROM THE DATASET
        data = self.data[idx]
        
        word = data['word']
        defi = data['full_entry']
        
        tknData = self.tokenAtron.encode(defi).ids

        padding = self.seqLen - len(tknData) - 2 #both bos and eos



        #add special tokens
        tknData = torch.cat([self.bosToken,
                              torch.tensor(tknData,dtype=torch.int64),
                              self.eosToken,
                              torch.tensor([self.padToken] * padding, dtype=torch.int64)])

        #should have the seqLen
        assert tknData.size(0) == self.seqLen
        
        tknDef = tknData#[1:]
        
        #the encoder mask only needs to mask pad tokens
        maskPad = (tknDef != self.padToken).int()
        maskCau = causalMask(tknDef.size(0))
        
        #mask tokens are True
        maskPad,maskCau = ~maskPad.bool(),~maskCau.bool()
        
        '''
        FOR WORD WE'LL USE LETTER LEVEL TKNZTN
        '''
        
        tknData = self.tokenAtron.encode(word).ids
        padding = self.wordLen - len(tknData) - 2
        
        
        #add special tokens
        tknWord = torch.cat([self.bosToken,
                              torch.tensor(tknData,dtype=torch.int64),
                              self.eosToken,
                              torch.tensor([self.padToken] * padding, dtype=torch.int64)])
        
        #should have the wordLen
        if tknWord.size(0) != self.wordLen:
            print('a')
        assert tknWord.size(0) == self.wordLen
        
        #the word requires no causal masking
        maskPadW = (tknWord != self.padToken).int()
        maskPadW = ~maskPadW.bool()
        
        # print(tknDef.shape, tknWord.shape, maskPad.shape, maskCau.shape)
        
        return {'tknWrd': tknWord,
                'tknDef': tknDef,
                'maskPad': maskPad,
                'maskCau': maskCau,
                'maskPadW': maskPadW,
                'raw': data
                }
            
def causalMask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0









