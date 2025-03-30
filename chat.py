# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:29:06 2025

@author: Mateo-drr
"""

import torch
from pathlib import Path
from VImodel import VImodel, VImodelRepMode
from tokenizers import Tokenizer

repmode=True

def generate_response(model, tokenizer, prompt, 
                     max_length=100, 
                     temperature=1.0, 
                     top_k=0, 
                     top_p=0.0, 
                     repetition_penalty=1.0,
                     device='cuda'):
    """
    Generate text with various sampling parameters for better output quality
    
    Args:
        model: The language model
        tokenizer: The tokenizer to encode/decode text
        prompt: Starting text prompt
        max_length: Maximum number of tokens to generate
        temperature: Controls randomness (lower = more deterministic)
        top_k: Limits sampling to top k most likely tokens (0 = disabled)
        top_p: Nucleus sampling parameter (0.0 = disabled, 0.9 = sample from tokens comprising 90% of probability mass)
        repetition_penalty: Penalizes repetition (1.0 = no penalty, >1.0 = penalize)
        device: Device to run generation on ('cuda' or 'cpu')
    
    Returns:
        str: Generated text including the prompt
    """
    # Start with just the prompt
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids]).to(device)
    
    # Track generated tokens to apply repetition penalty
    generated = []
    
    # Generate one token at a time
    for _ in range(max_length):
        # Create masks
        maskPad = (input_tensor != tokenizer.token_to_id('[PAD]')).int()
        maskCau = torch.triu(torch.ones(input_tensor.size(1), input_tensor.size(1))).bool()
        maskPad, maskCau = ~maskPad.bool(), ~maskCau.bool()
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor, maskPad.to(device), maskCau.to(device))
        
        # Get logits for the next token only
        next_token_logits = output[0, -1, :].clone()
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Apply repetition penalty
        if repetition_penalty > 1.0:
            for token_id in set(generated):
                if token_id < len(next_token_logits):
                    next_token_logits[token_id] /= repetition_penalty
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, next_token_logits.size(-1))
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')
        
        # Apply top-p (nucleus) filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                -1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = -float('Inf')
        
        # Convert logits to probabilities
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Sample from the distribution or take the most likely token
        if temperature == 0 or (top_k == 1 and top_p == 0):
            next_token_id = torch.argmax(next_token_logits).item()
        else:
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        # Add to generated tokens list (for repetition penalty)
        generated.append(next_token_id)
        
        # Stop if we hit EOS
        if next_token_id == tokenizer.token_to_id('[EOS]'):
            break
            
        # Add to input tensor
        input_tensor = torch.cat([input_tensor, 
                               torch.tensor([[next_token_id]]).to(device)], dim=1)
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(input_tensor[0].tolist())
    return generated_text

# Load model and start chat
def chat():
    # Load saved model
    checkpoint_path = Path(__file__).resolve().parent / 'weights' / 'best.pth'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['config']
    
    #load tokenizer
    tokenAtron = Tokenizer.from_file((Path(__file__).resolve().parent / 'redpajama' / config.tokenizer_file).as_posix())
    
      
    
    if not repmode:
        model = VImodel(vocabSize=tokenAtron.get_vocab_size(),
                                    seqLen=384,
                                    dModel=768,
                                    hidSize=3072,
                                    heads=12,
                                    layers=12,
                                    dropout=0.1)
    else:
        model = VImodelRepMode(vocabSize=tokenAtron.get_vocab_size(),
                        seqLen=config.seq_len,
                        dModel=config.d_model,
                        hidSize=config.hid_size,
                        heads=config.heads,
                        layers=config.layers,
                        dropout=config.dropout)
    
    model = model.to('cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Chat started! (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        # Add bos tkn
        prompt = f"[BOS] {user_input}"
        
        response = generate_response(model, tokenAtron, prompt,
                                     temperature=0.7,
                                     top_p=0.9,
                                     repetition_penalty=1.2)
        print("Avina:", response)

if __name__ == "__main__":
    chat()