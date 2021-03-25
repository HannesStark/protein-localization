import difflib
from collections import defaultdict

import h5py
import pandas as pd

# Using readlines()
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

file1 = open('../data/embeddings/subCell_all_pssmFile.txt', 'r')
Lines = file1.readlines()

fasta_sequences = []
for record in tqdm(SeqIO.parse(open('../data/embeddings/subCell_all.fasta'), 'fasta')):
    fasta_sequences.append(str(record.seq))

count = 0
# Strips the newline character
skipnext = False
with h5py.File('../data/embeddings/subCell_all_pssmFile.h5', 'w') as hf:
    pssms = {}
    lengths = defaultdict(list)
    sequence = []
    pssm = []
    for i, line in tqdm(enumerate(Lines[2:])):
        if not skipnext:
            string = line.strip().split()
            if string[0] == 'Query':
                skipnext = True
                pssm_sequence = ''.join(sequence)
                pssms[pssm_sequence] = torch.stack(pssm)
                lengths[len(pssm_sequence)].append(pssm_sequence)
                sequence = []
                pssm = []
            else:
                if string[1] != 'X':
                    sequence.append(string[1])
                pssm.append(torch.tensor(np.array(string[2:], dtype=int)))
        else:
            skipnext = False

    for i, fasta_sequence in tqdm(enumerate(fasta_sequences)):
        lookup_sequences = lengths[len(fasta_sequence)]
        sequence_id = difflib.get_close_matches(fasta_sequence, lookup_sequences, n=1, cutoff=0)[0]
        pssm = pssms[sequence_id]
        hf.create_dataset(fasta_sequence, data=pssms[sequence_id])

with h5py.File('../data/embeddings/subCell_all_pssmFile.h5', 'r') as hf:
    keys = list(hf.keys())
    sequences = []
    counter = 0
    for record in tqdm(SeqIO.parse(open('../data/embeddings/subCell_all.fasta'), 'fasta')):
        sequences.append(record.seq)
        if record.seq in keys:
            print(record.seq)
            counter += 1

    print(counter)
