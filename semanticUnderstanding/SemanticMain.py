# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:58:56 2025

@author: Mateo-drr

Train a transformer-encoder based model to learn word - definition concepts
Ideally to be used in conjunction to add definitions to another model
"""

from ds import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
import torch
import wandb
import copy
from pathlib import Path
from AthameModel import Athame
import json
from tokenizers import Tokenizer
import torch.nn.functional as F
import gc


#PARAMS
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark = True

configD = {'lr':2.5e-4,
           'num_epochs': 100,
           'batch':100,
           'seqLen_def': 107 + 2, #to add bos and eos
           'seqLen_wrd': 10 + 2,
           'd_model': 768, #token embbeddings size
           'hidSize':1024,
           'dropout':0.2,
           'heads':8,
           'layers':4,
           'wb':True,
           'device': 'cuda',
           'project_name': 'Athame',
           'basePath': Path(__file__).resolve().parent, #base dir
           'modelDir': Path(__file__).resolve().parent / 'weights',
           'tokenizer_file': 'unitknz.json'
           }
config = SimpleNamespace(**configD)

with open(config.basePath / 'WebstersUnabridgedDictionary' / 'redux.json', 'r') as file:
    data = json.load(file)
    
tokenAtron = Tokenizer.from_file((config.basePath / 'WebstersUnabridgedDictionary' / config.tokenizer_file).as_posix())

train_ds = CustomDataset(data, tokenAtron, seqLen=config.seqLen_def,
                         wordLen=config.seqLen_wrd)
train_dl = DataLoader(train_ds, batch_size=config.batch, pin_memory=True, shuffle=True)

# Instantiate the model
athame = Athame(dModelDef=config.d_model,
               dModelWrd=config.d_model,
               seqLenDef=config.seqLen_def,
               seqLenWrd=config.seqLen_wrd,
               heads=config.heads,
               layers=config.layers,
               dropout=config.dropout,
               hidSize=config.hidSize,
               srcVsize=tokenAtron.get_vocab_size(),
               tgtVsize=tokenAtron.get_vocab_size()).to(config.device)
# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenAtron.token_to_id('[PAD]'),
                                label_smoothing=0.1).to(config.device)

def cosSim(selfdef, xencp):
    totL=0
    for sdef in selfdef:
        loss = 1 - F.cosine_similarity(sdef, xencp).mean()
        loss = loss * F.mse_loss(sdef, xencp)
        totL+=loss
    return totL/len(selfdef)

optimizer = optim.AdamW(athame.parameters(), lr=config.lr)

#init weights & biases
if config.wb:
    wandb.init(project=config.project_name,
               config=configD)

# Calculate the total number of parameters
total_params = sum(p.numel() for p in athame.parameters())
# Print the total number of parameters
print(f'Total number of parameters: {total_params}')

bestTloss=1e9
bestVloss=1e9
for epoch in range(config.num_epochs):    
    
    torch.cuda.empty_cache()
    gc.collect()
    
    athame.train()
    trainLoss=0
    for sample in tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
        
        word=sample['tknWrd'].to(config.device)
        maskPadW = sample['maskPadW'].to(config.device)
        
        defi=sample['tknDef'].to(config.device)
        maskPad=sample['maskPad'].to(config.device)
        #og is [b,1,seqlen,seqlen] just need [seqlen,seqlen]
        maskCau=sample['maskCau'][0].squeeze(0).to(config.device)
        
        
        xenc = athame.encode(x=word, padMaskS=maskPadW)
        out, selfdef, xencp, = athame.decode(xenc, xtgt=defi,
                                             padMaskT=maskPad,
                                             cauMaskT=maskCau)
        
        #reshape for loss calc
        out = out.view(-1, tokenAtron.get_vocab_size())
        lbl = defi.view(-1)
        
        optimizer.zero_grad()    
        csim = cosSim(selfdef, xencp)
        ce = criterion(out, lbl)
        loss = ce + csim  # Compute loss
        loss.backward()             # Backward pass
        optimizer.step()        

        trainLoss += loss.item()
        
        # break
        
        if config.wb:
            wandb.log({"TLoss": loss,
                       'Cosine Sim': csim,
                       'Cross Ent': ce,
                       'Learning Rate': optimizer.param_groups[0]['lr']})
            
    avg_loss = trainLoss / len(train_dl)    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    athame.eval()
    validLoss=0
    valce,valcsim=0,0
    with torch.no_grad():
        for sample in tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            
            word=sample['tknWrd'].to(config.device)
            maskPadW = sample['maskPadW'].to(config.device)
            
            defi=sample['tknDef'].to(config.device)
            maskPad=sample['maskPad'].to(config.device)
            #og is [b,1,seqlen,seqlen] just need [seqlen,seqlen]
            maskCau=sample['maskCau'][0].squeeze(0).to(config.device)
            
            
            xenc = athame.encode(x=word, padMaskS=maskPadW)
            out, selfdef, xencp = athame.decode(xenc, xtgt=defi,
                                                 padMaskT=maskPad,
                                                 cauMaskT=maskCau)
            
            #reshape for loss calc
            out = out.view(-1, tokenAtron.get_vocab_size())
            lbl = defi.view(-1)
              
            # csim = cosSim(selfdef, xencp)
            ce = criterion(out, lbl)
            loss = ce   # Compute loss
            
            validLoss += loss.item()
            # valcsim += csim.item()
            valce += ce.item()
            
        avg_lossV = validLoss / len(train_dl)    
        print(f'Epoch {epoch+1}, Loss: {avg_lossV}') 


    if config.wb:
        wandb.log({"Validation Loss": avg_lossV,
                   'Val CE': valce / len(train_dl),
                   # 'Val Csim': valcsim / len(train_dl),
                   "Training Loss": avg_loss})
    
    if avg_loss <= bestTloss and avg_lossV <= bestVloss:
        bestModel = copy.deepcopy(athame)
        bestTloss = avg_loss
        bestVloss = avg_lossV
        torch.save({
        'model_state_dict': athame.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losstv': (avg_loss,avg_lossV),
        'config':config,
        }, config.modelDir / 'best.pth')
    
if config.wb:
    wandb.finish()    










