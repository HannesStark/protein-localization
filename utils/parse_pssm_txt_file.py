import h5py
import pandas as pd

# Using readlines()
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

#file1 = open('../data/embeddings/subCell_all_pssmFile.txt', 'r')
#Lines = file1.readlines()
#
#count = 0
## Strips the newline character
#skipnext = False
#with h5py.File('../data/embeddings/subCell_all_pssmFile.h5', 'w') as hf:
#    sequences = []
#    sequence = []
#    pssm = []
#    for i, line in tqdm(enumerate(Lines[2:])):
#        if not skipnext:
#            string = line.strip().split()
#
#            if string[0] == 'Query':
#                skipnext = True
#                sequence_str = ''.join(sequence)
#                if sequence_str not in sequences:
#                    hf.create_dataset(sequence_str, data=torch.stack(pssm))
#                    sequences.append(sequence_str)
#                sequence = []
#                pssm = []
#            else:
#                if string[1] != 'X':
#                    sequence.append(string[1])
#                pssm.append(torch.tensor(np.array(string[2:], dtype=int)))
#        else:
#            skipnext = False
#    hf.create_dataset(''.join(sequence), data=torch.stack(pssm))

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