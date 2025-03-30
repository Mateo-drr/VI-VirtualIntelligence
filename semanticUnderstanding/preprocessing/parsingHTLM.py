# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 23:15:11 2025

@author: Mateo-drr

https://www.gutenberg.org/ebooks/29765
"""

import re
import json
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import unicodedata
import numpy as np
from tokenizers.models import WordLevel, BPE, Unigram
from tokenizers.trainers import WordLevelTrainer, BpeTrainer, UnigramTrainer
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
import matplotlib.pyplot as plt
import pickle

current_dir = Path(__file__).resolve().parent

gtp = ['See Guide to pronunciation, §§ 43-74.',
       '(See Guide to Pronunciation, §§ 196,220.)',
       'Note: See Guide to Pronunciation, t\'c5 221-228',
       'See Guide to Pronunciation, sq. root178, 179, 229',
       'See Guide to Pronunciation, §§ 74-97',
       'Note: [See Guide to Pronunciation, sq. root 155-7, 184.]',
       'See Guide to Pronunciation, sq. root 178, 179, 188, 198, 230.',
       'See Guide to Pronunciation, §§ 231-6, 155, 176, 178, 179, 196, 211, 246.',
       '(see Guide to Pronunciation, §§ 19, 161, 162)',
       '(see Guide to Pronunciation, §§ 18, 97, 191)',
       'For voice-glide, see Guide to Pronunciation, §§ 17, 95.',
       'See Guide to Pronunciation, §§ 153, 179, 181-3, 237-8.',
       'See Guide to Pronunciation, §§ 10, 11.',
       'See Guide to Pronunciation, §§ 98-106.',
       '(see Guide to Pronunciation, §§159, 189)',
       'See Guide to Pronunciation, §§ 179, 211, 239.',
       'Note: See Guide to Pronunciation , §§ 240, 178, 179, 185.',
       'See Guide to Pronunciation, § 241.',
       'See Guide to Pronunciation,',
       'See Guide to Pronunciation, §§ 5, 10, 11.',
       'See Guide to Pronunciation, §§ 178-180, 242.',
       'See Guide to Pronunciation, §§ 10, 11.',
       'See Guide to Pronunciation, §§ 243-246.',
       'See Guide to Pronunciation, § 13.',
       '(see Guide to Pronunciation, §§ 20, 208)',
       'See Guide to Pronunciation, § 17.',
       'See Guide to Pronunciation, §§ 107-129.',
       'See Guide to Pronunciation, §§ 247, 248, and 184-195.',
       'See Guide to Pronunciation, § 249.',
       'See Guide to Pronunciation, §§ 178, 179, and 250-254.',
       'See Guide to Pronunciation, § 11.', #x3
       'See Guide to pronunciation, t\'c5 255-261.', #not working
       'See Guide to Pronunciation, §§ 197-208.',
       'See Guide to pronunciation, §§ 31-35.',
       'See Guide to Pronunciation, §§155, 199-202.',
       'See Guide to Pronunciation, §§169, 179, 180.',
       'See Guide to Pronunciation, §275.',
       'See Guide to Pronunciation, §§ 130-144.',
       'See Guide to Pronunciation, § 265; also §§ 155, 169, 178-179, etc.',
       'See Guide to Pronunciation, §§ 5, 146, 155.',
       'See Guide to Pronunciation, §§ 5, 146-149.',
       'See Guide to Pronunciation, §§ 266-268.',
       'See Guide to Pronunciation, § 13-15.',
       'See Guide to Pronunciation, §§ 217, 270, 271.',
       'See Guide to Pronunciation, §§ 145, 178-9, 272.',
       'See Guide to Pronunciation, §§ 273, 274.',
       #additional
       'See Short, a., 13, and Guide to Pronunciation, §§ 22, 30.',
       'See Neutral vowel, under Neutral and Guide to Pronunciation, § 17.',
       'See Quantity, and Guide to Pronunciation, §§22, 30.',
       'See Voice, n., 2, and Guide to Pronunciation, §§ 5, 153, 154.',
       'See Guide to pronunciation, t\\\'c5 255-261.',
       'See Voice, and Vowel, also Guide to Pronunciation, §§ 199-202.'
       ]

def normalize_text(text):
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    #remove useless parts
    # text = re.sub('See Guide to [Pp]ronunciation,?\s*§*\s*\d+[-\d]*\.?', '',text)
    # text = re.sub('See Guide to [Pp]ronunciation,','',text)

    # Replace curly quotes with straight quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Left and right double quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Left and right single quotes
    
    # Normalize dashes
    text = text.replace('\u2014', '-').replace('\u2013', '-')  # Em dash and en dash
    text = text.replace('--', '-')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Loop through the list and remove each text
    for text_to_remove in gtp:
        text = re.sub(re.escape(text_to_remove), '', text)
        
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

# Load the dictionary text file
with open(current_dir.parent / 'WebstersUnabridgedDictionary' / 'dictionary.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    
# List to store entries
entries_data = []

# This pattern looks for words that are capitalized and stand alone on a line
# followed by all text until the next capitalized word on its own line
entry_pattern = r'(?:^|\n)([A-Z][-A-Z\']*(?:\s+[-A-Z\']+)*)\n(.*?)(?=\n[A-Z][-A-Z\']*(?:\s+[-A-Z\']+)*\n|\Z)'
entries = re.findall(entry_pattern, content, re.DOTALL)

for word, entry_text in entries:
    word = word.strip()
    entry_text = entry_text.strip()
    
    word = normalize_text(word)
    entry_text = normalize_text(entry_text)
    
    # Split the full_entry into a list of words
    entry_parts = entry_text.split(' ')
    # Remove the first item (word breakdown)
    entry_text = ' '.join(entry_parts[1:])
    
    entries_data.append({
        'word': word,
        'full_entry': entry_text
    })
    
    
#%%    
    
#calcuate statistics on the lengths of definitions
lens=[]
for entry in entries_data:
    lens.append(len(entry['full_entry']))

print('mean', np.mean(lens), 'max', np.max(lens), 'min', np.min(lens),
      'median', np.median(lens))
print(f"75th Percentile: {np.percentile(lens, 75):.2f} chars")
print(f"90th Percentile: {np.percentile(lens, 90):.2f} chars")
print(f"95th Percentile: {np.percentile(lens, 95):.2f} chars")
    
    
#cropping dataset at 75th percentile chars to avoid excesive padding
redux=[]
for entry in entries_data:    
    if len(entry['full_entry']) <= np.percentile(lens, 75):
        redux.append(entry)
    
# Save the list of dictionaries to a JSON file
with open(current_dir.parent / 'WebstersUnabridgedDictionary' / 'redux.json', 'w') as f:
    json.dump(redux, f, indent=4)
    
'''
train tokenizers
'''    

def getSentences(ds):
    for item in ds:
        yield item['word'] + ' ' + item['full_entry']

def makeTokenizer(ds):
    
    #tokenize definition
    # tokenizerDef = Tokenizer(WordLevel(unk_token='[UNK]'))
    # tokenizerDef.pre_tokenizer = Whitespace()
    # trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'], min_frequency=1)
    # tokenizerDef.train_from_iterator(getSentences(ds), trainer=trainer)
    #tokenize word
    tokenizer = Tokenizer(Unigram()) #no need to give unk id here
    tokenizer.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(vocab_size=30000,
                             special_tokens=['[UNK]',
                                             '[PAD]',
                                             '[BOS]',
                                             '[EOS]'],
                             min_frequency=1)
    tokenizer.train_from_iterator(getSentences(ds), trainer=trainer)
        
    return tokenizer
    
tokenizer = makeTokenizer(redux)    
    
lenw,lend = [],[]
for data in redux:
    w = tokenizer.encode(data['word']).ids    
    lenw.append(len(w))
    d = tokenizer.encode(data['full_entry']).ids
    lend.append(len(d))
    
print(np.max(lenw), np.max(lend))

tokenizer.save((current_dir.parent / 'WebstersUnabridgedDictionary' / 'unitknz.json').as_posix())

#sanity check

# Phrase to search for
phrase = "Guide to Pronunciation"

# Check if the phrase is in any of the full_entry values
for entry in entries_data:
    full_entry = entry.get('full_entry', '')
    if phrase.lower() in full_entry.lower():  # case-insensitive check
        print(entry)