# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:18:41 2025

@author: Mateo-drr

This is the main code to run the train and validation loops
of the custom transformer-decoder based model to attain similar 
performace to gpt2 but trained in a much smaller scale and less data

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
from torch.amp import autocast, GradScaler

import ds as dataset
from VImodel import VImodel, VImodelCstm, VImodelRepMode
from transformers import GPT2LMHeadModel, GPT2Config
from bitsandbytes.optim import Lion
from dsPreproc import buildDataset
import pickle
from tokenizers import Tokenizer
import copy
import gc

FirstTime=False #needed to create ds and tokenizer, otherwise load them
gptMod=False
debug=False
pytorch=False
repmode=True

def main():
# if True:
    
    configD = {
        'batch_size': 24,
        'num_epochs': 12,
        'lr': 2.5e-4,
        'dropout':0.1,
        'seq_len': 384+1, #max tokens in utter
        'd_model': 768, #token embbeddings size
        'hid_size': 1024, #feed forward layer size
        'grad_clip':1,
        'heads': 12,
        'layers': 2,
        'tokenizer_file': 'en_bpe_redpj_top.json',
        'ds_name': 'en_bpeDs_top.pkl',
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
        # np.mean('a')
        tokenAtron.save((config.basePath / 'redpajama' / config.tokenizer_file).as_posix())
        with open(config.basePath / 'redpajama' / config.ds_name, 'wb') as f:
            pickle.dump(preprocDs, f)
    else:
        #load built ds and tknz
        tokenAtron = Tokenizer.from_file((config.basePath / 'redpajama' / config.tokenizer_file).as_posix())
        with open(config.basePath / 'redpajama' / config.ds_name, 'rb') as f:
            preprocDs = pickle.load(f)
    # np.mean('a')
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
    if not gptMod:
        if pytorch:
            model = VImodel(vocabSize=tokenAtron.get_vocab_size(),
                            seqLen=config.seq_len,
                            dModel=config.d_model,
                            hidSize=config.hid_size,
                            heads=config.heads,
                            layers=config.layers,
                            dropout=config.dropout).to(config.device)
        else:
            if not repmode:
                model = VImodelCstm(vocabSize=tokenAtron.get_vocab_size(),
                                seqLen=config.seq_len,
                                dModel=config.d_model,
                                hidSize=config.hid_size,
                                heads=config.heads,
                                layers=config.layers,
                                dropout=config.dropout).to(config.device)
            else:
                model = VImodelRepMode(vocabSize=tokenAtron.get_vocab_size(),
                                seqLen=config.seq_len,
                                dModel=config.d_model,
                                hidSize=config.hid_size,
                                heads=config.heads,
                                layers=config.layers,
                                dropout=config.dropout).to(config.device)
    else:
        cf = GPT2Config.from_pretrained('gpt2',
                                        vocab_size=tokenAtron.get_vocab_size())
        model = GPT2LMHeadModel(cf)
        model = model.to(config.device)
    
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Print the total number of parameters
    print(f'Total number of parameters: {total_params}')
    
    #compile
    #model = torch.compile(model, mode='max-autotune')

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenAtron.token_to_id('[PAD]'),
                                    label_smoothing=0.1).to(config.device)
    
    #optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    optimizer = Lion(model.parameters(), lr=config.lr)
    
    config.tot_steps = len(train_dl)#config.num_epochs * len(train_dl)

    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=int(config.tot_steps),  # First period (number of iterations before the first restart)
    #     T_mult=1,  # No change in period after each restart (this can be adjusted if needed)
    #     eta_min=config.lr * 0.1,  # Minimum learning rate (10% of initial)
    # )
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer,
    # mode='min',          # Use 'min' for loss (reduce when loss plateaus)
    # factor=0.5,          # Factor by which the learning rate will be reduced
    # patience=3,          # Number of epochs with no improvement before reducing LR
    # verbose=True,        # Print a message when the learning rate is reduced
    # min_lr=1e-6          # Minimum learning rate, ensures LR does not decay below this value
    # )   

    
    scaler = GradScaler(device=config.device)
    
    #initialize wb
    if config.wb:
        wandb.init(project="VImodel",
                   config=configD)
        
    bestModel=None
    bestTloss=1e6
    bestVloss=1e6
    

    
    for epoch in range(config.num_epochs):    
        
        torch.cuda.empty_cache()
        gc.collect()
        
        model.train()
        trainLoss=0
        k=0
        

        
        for data in tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            
            indata = data['indata'].to(config.device)
            target = data['target'].to(config.device)
            maskPad = data['maskPad'].to(config.device) #[b, seq len]
            maskCau = data['maskCau'][0].to(config.device).squeeze(0) #only need one mask [seqlen,seqlen]
            
            optimizer.zero_grad() 
            
            #run the model
            with autocast(device_type=config.device):
                
                if not gptMod:
                    out = model(indata, maskPad, maskCau) #[b , seqlen, vocabsize]
                else:
                    # maskPad,maskCau = ~maskPad.bool(),~maskCau.bool()
                    # maskPad = maskPad.float()
                    out = model(input_ids=indata, attention_mask=maskPad, labels=target)
                    # np.mean('a')
                    out = out.logits
                
                # np.mean('a')# print(out.shape)
                #for some reason he changes shape of out to [b * seqlen, vocabsize]
                out = out.view(-1, tokenAtron.get_vocab_size())
                # print(out.shape)
                
                target = target.view(-1) #and [b * seqlen]
                loss = criterion(out, target)  # Compute loss
                    
            if debug:
                for name, param in model.named_parameters():
                    if param.dim() > 1:  # Weight parameter
                        print(f"{name} requires_grad: {param.requires_grad}")
                
            scaler.scale(loss).backward()
            
            if debug:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # print(f"Gradients for {name}:")
                        # print(param.grad)
                    
                        # Optionally, print the norm of the gradient (for scale)
                        print(f"Norm of {name}'s gradient: {param.grad.norm().item()}")
            
                np.mean('a')
            # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            scaler.step(optimizer)    
            scaler.update()
            
            # if epoch > 0:
                # Step the scheduler
            # scheduler.step()
            # else:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = config.lr * (k+1 / config.tot_steps)
            
            if config.wb:
                wandb.log({"TLoss": loss, 'Learning Rate': optimizer.param_groups[0]['lr']})
    
            trainLoss += loss.item()
            # k+=1
            # if k== 5:
            #     break
            
        
        if epoch < 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
        # scheduler.step()
        
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
                if not gptMod:
                    out = model(indata, maskPad, maskCau) #[b , seqlen, vocabsize]
                else:
                    #TODO review masks
                    # maskPad,maskCau = ~maskPad.bool(),~maskCau.bool() 
                    # maskPad = maskPad.float()
                    out = model(input_ids=indata, attention_mask=maskPad, labels=target)
                    out = out.logits
                
                #for some reason he changes shape of out to [b * seqlen, vocabsize]
                out = out.view(-1, tokenAtron.get_vocab_size())
                target = target.view(-1) #and [b * seqlen]
                loss = criterion(out, target)  # Compute loss      
        
                validLoss += loss.item()
                
                # break
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
            'config':config,
            }, config.modelDir / 'best.pth')
        
    if config.wb:
        wandb.finish()    
        



# '''
if __name__ == "__main__":
    main()
#'''