import numpy as np
import pandas as pd
import os
import tensorflow as tf
import seaborn as sns
from Bio import SeqIO
from tqdm import tqdm
import time
from rhabdo_utils import *
from rhabdo_lstm import LSTMLanguageModel

fname =  'all_rhabdo.fasta'
seqs = {}

for record in SeqIO.parse(fname, 'fasta'):
    if len(record.seq) < 1000:
        continue
    if str(record.seq).count('X') > 0:
        continue
    if record.seq not in seqs:
        seqs[record.seq] = []
        
    meta = parse_viprbr_r(record.description)
    
    seqs[record.seq].append(meta)
    
AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ]
vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }

seq_len = max([ len(seq) for seq in seqs ]) + 2
vocab_size = len(AAs) + 2

model = LSTMLanguageModel(seq_len,
        vocab_size, from_checkpoint = True,
        checkpoint = './checkpoints/lstm/lstm_128-01.hdf5',
        inference_batch_size = 100,
        verbose=True)

family_seqs = {}
for seq in seqs:
  for meta in seqs[seq]:
      if ('lyssa' in meta['strain'] and 'Rabies' not in meta['strain']) or 'vesicul' in meta['strain']:
        family_seqs[seq] = seqs[seq]
        break
    
seqs = analyze_embedding(model, family_seqs, vocabulary, 'rhabdo')
