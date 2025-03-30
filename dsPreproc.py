# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:59:46 2025

@author: Mateo-drr
"""

from collections import Counter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import ast
import re
import unicodedata
import json

def normalize_text(text):
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    text = text.lower()

    # Replace curly quotes with straight quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Left and right double quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Left and right single quotes
    
    # Normalize dashes
    text = text.replace('\u2014', '-').replace('\u2013', '-')  # Em dash and en dash
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def cutoff(raw_texts,freq=1):
    # Count all words
    word_counts = Counter()
    for text in raw_texts:
        words = text.split()
        word_counts.update(words)
    
    # Create replacement map for rare words
    rare_words = {word for word, count in word_counts.items() if count == freq}
    
    # Replace rare words with [UNK]
    # simplified_texts = []
    for i,text in enumerate(raw_texts):
        words = text.split()
        words = ['[UNK]' if word in rare_words else word for word in words]
        #simplified_texts.append(' '.join(words))
        raw_texts[i] = ' '.join(words)
        
    return raw_texts

def getLanguageAndPerpScores():
    import numpy as np
    ds = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample", streaming=False) 
    metrics=[]
    for sample in ds['train']:
        if ast.literal_eval(sample['meta'])['language'] == 'en':
            metrics.append((json.loads(sample['quality_signals'])['ccnet_language_score'][0][-1],
                            json.loads(sample['quality_signals'])['ccnet_perplexity'][0][-1]))
    metrics = np.array(metrics)
    print(np.mean(metrics,axis=0))        
    #array([  0.91798676, 360.4429851 ])
    #so now in the loadRedPajama ill filter samples with langscore> 0.91 and ppx < 360
        

def loadRedPajama():
    ds = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample", streaming=False) 
    rawTxt = []
    for sample in ds['train']:
        if ast.literal_eval(sample['meta'])['language'] == 'en':
            qsig = json.loads(sample['quality_signals'])
            #only use high quality texts
            if qsig['ccnet_language_score'][0][-1] >= 0.917 and qsig['ccnet_perplexity'][0][-1] <= 360:
                rawTxt.append(normalize_text(sample['raw_content']))
    return tuple(rawTxt)

def getSentences(ds):
    for item in ds:
        yield item

def makeTokenizer(config, ds, method='wordlevel'):
    tokenizerPath = Path(config.tokenizer_file)
    if not Path.exists(tokenizerPath):
        if method == 'wordlevel':
            tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'], min_frequency=2)
            tokenizer.train_from_iterator(getSentences(ds), trainer=trainer)
        elif method == 'bpe':
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'], min_frequency=2)
            tokenizer.train_from_iterator(getSentences(ds), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizerPath))
        
    return tokenizer

def buildDataset(config):
    print('Step 1/3 load data...')
    dsRaw = loadRedPajama() 
    # dsRaw = cutoff(dsRaw)
    
    #build tokenizer
    print('Step 2/3 build tokenizer...')
    tokenAtron = makeTokenizer(config, dsRaw, method='bpe')
    
    print('Step 3/3 find max sequence length...')
    cropped = []
    srcMax=0
    # Batch encode all items at once - much faster than individual encoding
    # encoded_batch = tokenAtron.encode_batch(dsRaw)
    print('Step 3/3 tokenized all data...')
    for item in dsRaw:
        srcId = tokenAtron.encode(item).ids
        
        if len(srcId) > config.seq_len:
            # Calculate splits once
            splits = [srcId[i:i + config.seq_len] for i in range(0, len(srcId), config.seq_len)]
            # Batch decode the splits
            for chunk in splits:
                cropped.append(tokenAtron.decode(chunk))
                srcMax = max(srcMax, len(chunk))
        else:
            cropped.append(tokenAtron.decode(srcId))
            srcMax = max(srcMax, len(srcId))
    
    print('Max len', srcMax)
    
    return cropped, tokenAtron


# import json

# def read_jsonl(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             # Parse each line as a JSON object
#             json_obj = json.loads(line.strip())
#             data.append(json_obj)
#     return data

# current_dir = Path(__file__).resolve().parent #base dir
# dspath = current_dir / 'movie-corpus_v1' 
# ds = read_jsonl(dspath / 'utterances.jsonl')

def organizeDataset(rawds):
    
    # #extract the ids of all the utters
    # idIndex = []
    # for utter in rawds:
    #     idIndex.append(int(utter['id'][1:])) #ids have name L####, remove L and turn into int
        
    # #sort and get the sorted indexes
    # sortedIdxs = sorted(range(len(idIndex)), key=lambda i: idIndex[i])

    # #sort the utters in order 
    # sortedConvs=[rawds[idx] for idx in sortedIdxs]
    # #and group them by conversation
    # groupped=[[sortedConvs[0]]]
    # convIdx=0
    # for i in range(1,len(sortedConvs)):
    #     currentIdx=int(sortedConvs[i]['id'][1:])
    #     if  currentIdx - int(groupped[convIdx][-1]['id'][1:]) == 1:
    #         groupped[convIdx].append(sortedConvs[i])
    #     else:
    #         convIdx+=1
    #         groupped.append([sortedConvs[i]])
        
    #Not the most efficient code but gets the job done :)
    #get the roots of all the conversations
    roots = []
    for utter in rawds:
        roots.append(int(utter['root'][1:]))
    #get the sorted indexes
    sortedIdxs = sorted(range(len(roots)), key=lambda i: roots[i])
    #sort the conversation based on the roots
    sortedConvs = [rawds[i] for i in sortedIdxs]
    #and group them by conversation
    groupped=[[sortedConvs[0]]]
    convIdx=0
    for i in range(1,len(sortedConvs)):
        currentIdx=int(sortedConvs[i]['root'][1:])
        if  currentIdx == int(groupped[convIdx][-1]['root'][1:]):
            groupped[convIdx].append(sortedConvs[i])
        else:
            convIdx+=1
            groupped.append([sortedConvs[i]])
            
    #Now with the sorted roots we can sort each utter in each conversation
    replySorted=[]
    for conv in groupped:
        replyIdx=[]
        for utter in conv:
            if utter['reply-to'] is None:
                replyIdx.append(utter) #find first utter in conversation
                break
        
        i=0
        while True:
            if replyIdx[-1]['id'] == conv[i]['reply-to']:
                replyIdx.append(conv[i])
                i=0
            else:
                i+=1
            
            if len(replyIdx) == len(conv):
                break
        
        replySorted.append(replyIdx)
        
    return replySorted



