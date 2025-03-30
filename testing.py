import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import nltk
from tqdm import tqdm
from pathlib import Path
from tokenizers import Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Download Penn Treebank if needed
nltk.download('ptb')
from nltk.corpus import ptb

from VImodel import VImodel, VImodelRepMode
repmode = True

def causalMask(size):
    """Create a causal mask for self-attention"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # Convert to boolean mask where True values are attended to

class PennTreebankDataset(Dataset):
    def __init__(self, split='train', context_length=1024, tokenizer=None):
        """
        Args:
            split: 'train', 'valid', or 'test'
            context_length: maximum sequence length
            tokenizer: tokenizer to use (GPT2Tokenizer by default)
        """
        # Map split names to PTB fileids
        split_map = {
            'train': ptb.fileids()[:85],  # First 85% for training
            'valid': ptb.fileids()[85:90],  # Next 5% for validation
            'test': ptb.fileids()[90:]  # Last 10% for testing
        }
        
        # Get raw text
        text = " ".join([ptb.raw(f) for f in split_map[split]])
        
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        # Set pad token ID
        if isinstance(tokenizer, Tokenizer):
            self.pad_token = tokenizer.token_to_id("[PAD]") if tokenizer.token_to_id("[PAD]") is not None else 0
        else:
            self.pad_token = tokenizer.pad_token_id
            
        # Check if we have a tokenizers.Tokenizer or a transformers tokenizer
        if isinstance(tokenizer, Tokenizer):
            # If it's a tokenizers.Tokenizer
            encoding = self.tokenizer.encode(text)
            self.input_ids = torch.tensor([encoding.ids], dtype=torch.long)
        else:
            # If it's a transformers tokenizer
            self.encodings = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                          max_length=context_length * 100)
            self.input_ids = torch.cat([self.bosToken,
                                  torch.tensor(self.encodings.input_ids,dtype=torch.int64),
                                  self.eosToken])
    
    def __len__(self):
        # We'll create as many samples as we can with the given context length
        return max(1, (self.input_ids.size(1) - 1) // self.context_length)
    
    def __getitem__(self, idx):
        # Calculate start and end indices
        start_idx = idx * self.context_length
        end_idx = start_idx + self.context_length + 1  # +1 to include the last token as a target
        
        # Extract the segment - ensure we don't exceed array bounds
        max_end = min(end_idx, self.input_ids.size(1))
        max_start = min(start_idx, max_end - 1)
        
        input_ids = self.input_ids[0, max_start:max_end-1].clone()
        labels = self.input_ids[0, max_start+1:max_end].clone()
        
        # Debug: print some sample values
        if idx == 0:
            print(f"Sample input_ids: {input_ids[:5]}")
            print(f"Sample labels: {labels[:5]}")
        
        # Pad if necessary
        if input_ids.size(0) < self.context_length:
            padding_input = torch.full((self.context_length - input_ids.size(0),), 
                                       self.pad_token, dtype=torch.long)
            padding_labels = torch.full((self.context_length - labels.size(0),), 
                                        self.pad_token, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding_input])
            labels = torch.cat([labels, padding_labels])
        
        # Create attention masks
        # Padding mask: 1 for tokens, 0 for padding
        mask_pad = (input_ids != self.pad_token).int()
        # Causal mask for autoregressive prediction
        mask_cau = causalMask(input_ids.size(0))
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'mask_pad': mask_pad,
            'mask_cau': mask_cau
        }

def evaluate_model(model, eval_dataloader, tokenizer, device='cuda', gptmod=False):
    """
    Evaluate a language model on Penn Treebank
    
    Args:
        model: the model to evaluate
        eval_dataloader: DataLoader for evaluation data
        tokenizer: the tokenizer used to tokenize the data
        device: device to run evaluation on
        gptmod: whether the model is a GPT model
    
    Returns:
        Dictionary of metrics including perplexity, bits per character, etc.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Set up criterion based on tokenizer type
    if isinstance(tokenizer, Tokenizer):
        pad_token_id = tokenizer.token_to_id('[PAD]') if tokenizer.token_to_id('[PAD]') is not None else 0
    else:
        pad_token_id = tokenizer.pad_token_id
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1).to(device)
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].long().to(device)
            labels = batch['labels'].long().to(device)
            
            # Get model outputs
            if not gptmod:
                mask_pad = batch['mask_pad'].to(device)
                mask_cau = batch['mask_cau'][0].to(device)
                logits = model(input_ids, mask_pad, mask_cau)  # [b, seqlen, vocabsize]
                
                # Reshape for loss calculation
                if isinstance(tokenizer, Tokenizer):
                    vocab_size = tokenizer.get_vocab_size()
                else:
                    vocab_size = len(tokenizer)
                
                logits = logits.view(-1, vocab_size)
                target = labels.view(-1)  # [b * seqlen]
                loss = criterion(logits, target)
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            
            # Accumulate loss
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_tokens += (labels != pad_token_id).sum().item()
    
    # Calculate metrics
    avg_loss = total_loss / len(eval_dataloader.dataset)
    perplexity = math.exp(avg_loss)
    bits_per_token = avg_loss / math.log(2)
    
    results = {
        'loss': avg_loss,
        'perplexity': perplexity,
        'bits_per_token': bits_per_token,
    }
    
    return results

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load saved model
    checkpoint_path = Path(__file__).resolve().parent / 'weights' / 'best.pth'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['config']
    
    # Load tokenizer
    tokenizer_path = (Path(__file__).resolve().parent / 'redpajama' / config.tokenizer_file).as_posix()
    tokenAtron = Tokenizer.from_file(tokenizer_path)
    
    # Initialize GPT-2 tokenizer and model
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model.to(device)
    
    # Load VI model
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
    
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataset and dataloader for GPT-2
    gpt2_test_dataset = PennTreebankDataset(split='test', tokenizer=gpt2_tokenizer)
    gpt2_test_dataloader = DataLoader(gpt2_test_dataset, batch_size=4)
    
    print("Evaluating GPT-2...")
    gpt2_results = evaluate_model(gpt2_model, gpt2_test_dataloader,
                                  gpt2_tokenizer, device=device, gptmod=True)
    
    # Create test dataset and dataloader for VI model
    vi_test_dataset = PennTreebankDataset(split='test', tokenizer=tokenAtron, context_length=385)
    vi_test_dataloader = DataLoader(vi_test_dataset, batch_size=4)
    
    print("Evaluating VI...")
    vi_results = evaluate_model(model, vi_test_dataloader, tokenAtron, device=device)
    
    print("\nGPT-2 Results:")
    for metric, value in gpt2_results.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nVI Model Results:")
    for metric, value in vi_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Print comparative analysis
    print("\nComparative Analysis:")
    for metric in gpt2_results.keys():
        diff = vi_results[metric] - gpt2_results[metric]
        if metric == 'perplexity' or metric == 'loss' or metric == 'bits_per_token':
            better = "better" if diff < 0 else "worse"
        else:
            better = "better" if diff > 0 else "worse"
        print(f"  {metric}: {abs(diff):.4f} points {better} than GPT-2")

if __name__ == "__main__":
    main()