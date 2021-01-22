import os

import h5py
from Bio import SeqIO
import numpy as np
from tqdm import tqdm

from utils.preprocess import reduce_embeddings
fasta_paths = [
    'data/fasta_files/hard_set.fasta',
    #'data/fasta_files/train.fasta',
    #'data/fasta_files/val_homreduced.fasta',
    #'data/fasta_files/test_as_per_deeploc.fasta'
]

save_name = [#'train_',
             #'val_',
             #'test_',
    'hard_set_'
    ]

for split_index, fasta_path in enumerate(fasta_paths):
    embeddings_file = h5py.File(os.path.join('data/embeddings/new_hard_set.h5'), 'r')
    print(os.path.join('data/embeddings/new_hard_set_reduced.h5'))
    reduced_embeddings = h5py.File(os.path.join('data/embeddings', save_name[split_index] + 'T5_reduced.h5'), 'w')
    for record in tqdm(SeqIO.parse(open(fasta_path), 'fasta')):
        if len(record.seq) < 13000:
            embedding = embeddings_file[str(str(record.description).replace('.','_').replace('/','_'))][:]
            reduced_embeddings.create_dataset(record.description,
                                              data=np.mean(embedding, axis=0))
        else:
            print(record.description)
    reduced_embeddings.flush()
    reduced_embeddings.close()
