import os

import h5py
from Bio import SeqIO
from tqdm import tqdm

counter = 0
for record in tqdm(SeqIO.parse(open('data/fasta_files/duplicates_removed.fasta'), 'fasta')):
    if '-U' not in record.description:
        counter += 1

print(counter)