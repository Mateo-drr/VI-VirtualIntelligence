# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:29:06 2025

@author: Mateo-drr
"""

import torch
from pathlib import Path
from VImodel import VImodel
from tokenizers import Tokenizer


def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  
    return model

def generate_response(model, tokenizer, prompt, max_length=100, device='cuda'):
    # Start with just the prompt
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids]).to(device)
    
    # Generate one token at a time
    for _ in range(max_length):
        # Create masks
        maskPad = (input_tensor != tokenizer.token_to_id('[PAD]')).int()
        maskCau = torch.triu(torch.ones(input_tensor.size(1), input_tensor.size(1))).bool()
        maskPad, maskCau = ~maskPad.bool(), ~maskCau.bool()
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor, maskPad.to(device), maskCau.to(device))
        
        # Get next token (simple greedy)
        next_token_id = output[0, -1].argmax().item()
        
        # Stop if we hit EOS
        if next_token_id == tokenizer.token_to_id('[EOS]'):
            break
            
        # Add to input
        input_tensor = torch.cat([input_tensor, 
                                torch.tensor([[next_token_id]]).to(device)], dim=1)
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(input_tensor[0].tolist())
    return generated_text

# Load model and start chat
def chat():
    # Load saved model
    checkpoint_path = Path(__file__).resolve().parent / 'weights' / 'best.pth'
    #load tokenizer
    tokenAtron = Tokenizer.from_file((Path(__file__).resolve().parent / 'redpajama' / 'tokenizer_redpj.json').as_posix())
    model = load_model(checkpoint_path,
                       VImodel(vocabSize=tokenAtron.get_vocab_size(),
                                seqLen=384,
                                dModel=512,
                                hidSize=2048,
                                heads=2,
                                layers=1,
                                dropout=0.1))
    model = model.to('cuda')
    
    print("Chat started! (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        # Add bos tkn
        prompt = f"[BOS] {user_input}"
        
        response = generate_response(model, tokenAtron, prompt)
        print("Avina:", response)

if __name__ == "__main__":
    chat()