# import os
#
#
# import numpy as np
#
# res = np.array([80.56, 80.39, 79.91, 80.28, 80.33])
#
# print(res.mean())
# print(res.std())
import os

import h5py
from Bio import SeqIO

base_path = '/mnt/home/mheinzinger/deepppi1tb/embedder/embeddings/hannes_embeddings'

appendix = ['hannes_deeploc_t5-encoderOnly.h5',
            'hannes_deeploc_t5-encoderOnlyHALF.h5']

save_appendix = ['t5-encoderOnly.h5',
                 't5-encoderOnlyHALF.h5']

fasta_paths = [
    'data/fasta_files/train.fasta',
    'data/fasta_files/val_homreduced.fasta',
    'data/fasta_files/test_as_per_deeploc.fasta'
]

save_name = ['train_',
             'val_',
             'test_']

paths = []
save_paths = []


for i, append in enumerate(appendix):
    embeddings_file = h5py.File(os.path.join(base_path, append), 'r')
    embeddings_file2 = h5py.File(os.path.join(base_path, 'hannes_deeploc_bertHALF.h5'), 'r')
    print(len(embeddings_file.keys()))
    for split_index, fasta_path in enumerate(fasta_paths):
        print(fasta_path)
        save_file = h5py.File(os.path.join('data/embeddings', save_name[split_index] + save_appendix[i]), 'w')
        for record in SeqIO.parse(open(fasta_path), 'fasta'):
            try:
                save_file.create_dataset(record.description,
                                         data=embeddings_file[
                                             str(record.description).replace('.', '_').replace('/', '_')])
            except:
                print(str(record.description).replace('.', '_').replace('/', '_'))
                save_file.create_dataset(record.description,
                                         data=embeddings_file2[
                                             str(record.description).replace('.', '_').replace('/', '_')])

