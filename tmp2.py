import os

import h5py
from Bio import SeqIO
from tqdm import tqdm

seqs = os.listdir('C:\\Users\\HannesStark\\tmp\\ppc')

for seq in seqs:
    print(seq)