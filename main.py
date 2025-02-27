# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:18:41 2025

@author: Mateo-drr

movies ds in https://www.cs.cornell.edu/~cristian/Chameleons_in_imagined_conversations.html

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import wandb
import numpy as np

import ds as dataset
from VImodel import VImodel
from dsPreproc import buildDataset
import pickle
from tokenizers import Tokenizer
import copy

FirstTime=False #needed to create ds and tokenizer, otherwise load them


def main():
# if True:
    
    configD = {
        'batch_size': 36,
        'num_epochs': 3,
        'lr': 1e-4,
        'dropout':0.1,
        'seq_len': 384, #max tokens in utter
        'd_model': 512, #token embbeddings size
        'hid_size': 2048, #feed forward layer size
        'heads': 2,
        'layers': 1,
        'tokenizer_file': 'tokenizer_redpj.json',
        'device': 'cuda',
        'wb':True,
        'basePath': Path(__file__).resolve().parent, #base dir
        'modelDir': Path(__file__).resolve().parent / 'weights',
        'projName': 'VImodel'
        }
    config = SimpleNamespace(**configD)
    
    if FirstTime:
        #build ds and process it
        preprocDs, tokenAtron = buildDataset(config)
        #save them
        tokenAtron.save((config.basePath / 'redpajama' / config.tokenizer_file).as_posix())
        with open(config.basePath / 'redpajama' / 'preprocDs.pkl', 'wb') as f:
            pickle.dump(preprocDs, f)
    else:
        #load built ds and tknz
        tokenAtron = Tokenizer.from_file((config.basePath / 'redpajama' / config.tokenizer_file).as_posix())
        with open(config.basePath / 'redpajama' / 'preprocDs.pkl', 'rb') as f:
            preprocDs = pickle.load(f)
    
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    torch.backends.cudnn.benchmark = True
    
    #reduce ds size
    preprocDs = preprocDs[:int(len(preprocDs)*0.3)]
    #split ds
    train_ds,valid_ds = dataset.splitDataset(preprocDs, tokenAtron, config, split=0.9)
    del preprocDs
    #creat dls
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, pin_memory=True, shuffle=True, num_workers=2)
    valid_dl = DataLoader(valid_ds, batch_size=config.batch_size, pin_memory=True, shuffle=False)
    
    # Instantiate the model
    model = VImodel(vocabSize=tokenAtron.get_vocab_size(),
                    seqLen=config.seq_len,
                    dModel=config.d_model,
                    hidSize=config.hid_size,
                    heads=1,
                    layers=1,
                    dropout=config.dropout).to(config.device)
    

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenAtron.token_to_id('[PAD]'),
                                    label_smoothing=0.1).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    
    #initialize wb
    if config.wb:
        wandb.init(project="VImodel",
                   config=configD)
        
    bestModel=None
    bestTloss=1e6
    bestVloss=1e6
    
    for epoch in range(config.num_epochs):    
        
        model.train()
        trainLoss=0
        k=0
        for data in tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            
            indata = data['indata'].to(config.device)
            target = data['target'].to(config.device)
            maskPad = data['maskPad'].to(config.device) #[b, seq len]
            maskCau = data['maskCau'][0].to(config.device).squeeze(0) #only need one mask [seqlen,seqlen]
            
            #run the model
            out = model(indata, maskPad, maskCau) #[b , seqlen, vocabsize]
            
            #loss and param update
            optimizer.zero_grad()  
            #for some reason he changes shape of out to [b * seqlen, vocabsize]
            out = out.view(-1, tokenAtron.get_vocab_size())
            
            target = target.view(-1) #and [b * seqlen]
            loss = criterion(out, target)  # Compute loss
            loss.backward()             # Backward pass
            optimizer.step()        
            
            if config.wb:
                wandb.log({"TLoss": loss})
    
            trainLoss += loss.item()
            # k+=1
            # if k== 5:
            #     break
            
        avg_loss = trainLoss / len(train_dl)    
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')    
        
        model.eval()
        with torch.no_grad():
            validLoss=0
            for data in tqdm(valid_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                    
                indata = data['indata'].to(config.device)
                target = data['target'].to(config.device)
                maskPad = data['maskPad'].to(config.device) #[b, seq len]
                maskCau = data['maskCau'][0].to(config.device).squeeze(0) #only need one mask [seqlen,seqlen]
                
                #run the model
                out = model(indata, maskPad, maskCau) #[b , seqlen, vocabsize]
                
                #for some reason he changes shape of out to [b * seqlen, vocabsize]
                out = out.view(-1, tokenAtron.get_vocab_size())
                target = target.view(-1) #and [b * seqlen]
                loss = criterion(out, target)  # Compute loss      
        
                validLoss += loss.item()
                
            avg_lossV = validLoss / len(valid_dl)    
            print(f'Epoch {epoch+1}, Loss: {avg_lossV}') 
        
        if config.wb:
            wandb.log({"Validation Loss": avg_lossV, "Training Loss": avg_loss})
        
        if avg_loss <= bestTloss and avg_lossV <= bestVloss:
            bestModel = copy.deepcopy(model)
            bestTloss = avg_loss
            bestVloss = avg_lossV
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'losstv': (avg_loss,avg_lossV),
            # Add any other info you want to save
            }, config.modelDir / 'best.pth')
        
    if config.wb:
        wandb.finish()    
        



# '''
if __name__ == "__main__":
    main()
#'''