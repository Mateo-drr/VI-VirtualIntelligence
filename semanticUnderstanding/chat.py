# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 19:47:31 2025

@author: Mateo-drr
"""

import torch
from pathlib import Path
from tokenizers import Tokenizer
from AthameModel import Athame

device='cuda'

# Load saved model
checkpoint_path = Path(__file__).resolve().parent / 'weights' / 'best.pth'
checkpoint = torch.load(checkpoint_path, weights_only=False)
config = checkpoint['config']

#load tokenizer
tokenAtron = Tokenizer.from_file((Path(__file__).resolve().parent / 'WebstersUnabridgedDictionary' / config.tokenizer_file).as_posix())

model = Athame(dModelDef=config.d_model,
               dModelWrd=config.d_model,
               seqLenDef=config.seqLen_def,
               seqLenWrd=config.seqLen_wrd,
               heads=config.heads,
               layers=config.layers,
               dropout=config.dropout,
               hidSize=config.hidSize,
               srcVsize=tokenAtron.get_vocab_size(),
               tgtVsize=tokenAtron.get_vocab_size()).to(config.device)

model = model.to('cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Chat started! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
        
    # Add bos tkn
    prompt = f"{user_input}"
    
    # Start with just the prompt
    input_ids = tokenAtron.encode(prompt).ids
    input_tensor = torch.cat([torch.tensor([tokenAtron.token_to_id('[BOS]')], dtype=torch.int64).to(device),
                              torch.tensor(input_ids).to(device),
                              torch.tensor([tokenAtron.token_to_id('[EOS]')], dtype=torch.int64).to(device),
                              ])
    input_tensor = input_tensor.unsqueeze(0) #add batch dim
    
    maskPad = (input_tensor != tokenAtron.token_to_id('[PAD]')).int()
    maskCau = torch.triu(torch.ones(input_tensor.size(1), input_tensor.size(1))).bool()
    maskPad, maskCau = ~maskPad.bool(), ~maskCau.bool()
    
    out = model(input_tensor, padMaskS=maskPad, cauMaskS=maskCau, tgt=None, padMaskT=None, cauMaskT=None)
    
    # Assuming the output is logits of shape [1, 122, 30000]
    out = out.squeeze(0)  # Remove the batch dimension (shape becomes [122, 30000])
    
    # Get the token indices with the highest probability (argmax across the vocabulary dimension)
    token_indices = out.argmax(dim=-1)  # Shape becomes [122]
    
    # Assuming you have a 'tokenizer' object with a method to decode token indices
    # or a dictionary that maps token indices to words.
    response = tokenAtron.decode(token_indices.tolist())  # Converts indices to text
    
    print("Avina:", response)