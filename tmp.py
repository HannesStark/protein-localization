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
import numpy as np
import h5py
from Bio import SeqIO
from tqdm import tqdm

from utils.preprocess import reduce_embeddings, remove_duplicates_full

base_path = '/mnt/home/mheinzinger/deepppi1tb/embedder/embeddings/hannes_embeddings'

appendix = ['hannes_deeploc_bertSECONDLAST.h5']

save_appendix = ['bertSECONDLAST.h5']

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

identifiers = []
descriptions = []
labels = []
sequences = []
solubility = []
for record in SeqIO.parse('data/fasta_files/duplicates_removedhard_set.fasta', "fasta"):
    identifiers.append(record.id)
    descriptions.append(record.description)
    labels.append(record.description.split(' ')[1].split('-')[0])
    sequences.append(str(record.seq))
    solubility.append(record.description.split(' ')[1].split('-')[-1])
print(descriptions)
print(len(descriptions))
loaded_embeddings = h5py.File('data/embeddings_raw/hannes_new_hard_test_set_PIDE20_t5-encoderOnly.h5', 'r')
print(loaded_embeddings.keys())
counter = 0
embeddings = h5py.File('data/embeddings/all_dupes_removed_hard_test_T5.h5', 'w')
keys = []
for key_raw in tqdm(loaded_embeddings.keys()):
    key = key_raw.replace('_','.').replace('Lysosome.Vacuole','Lysosome/Vacuole')
    keys.append(key)
    embedding = loaded_embeddings[key_raw][:]
    if key in descriptions:
        counter +=1
        embeddings.create_dataset(key, data=embedding)
embeddings.close()
print(set(descriptions) - set(keys))
print(counter)




#remove_duplicates_full('data/fasta_files/new_test_set.fasta', 'data/fasta_files')

reduce_embeddings(['data/embeddings/test_t5-encoderOnly.h5',
                   'data/embeddings/train_t5-encoderOnly.h5',
                   'data/embeddings/val_t5-encoderOnly.h5'],
                  'data/embeddings/reduced/',
                  ['test_t5_reduced.h5',
                   'train_t5_reduced.h5',
                   'val_t5_reduced.h5'])


#for i, append in enumerate(appendix):
#    embeddings_file = h5py.File(os.path.join(base_path, append), 'r')
#    embeddings_file2 = h5py.File(os.path.join(base_path, 'hannes_deeploc_bertHALF.h5'), 'r')
#    print(len(embeddings_file.keys()))
#    for split_index, fasta_path in enumerate(fasta_paths):
#        print(fasta_path)
#        save_file = h5py.File(os.path.join('data/embeddings', save_name[split_index] + save_appendix[i]), 'w')
#        for record in SeqIO.parse(open(fasta_path), 'fasta'):
#            try:
#                save_file.create_dataset(record.description,
#                                         data=embeddings_file[
#                                             str(record.description).replace('.', '_').replace('/', '_')])
#            except:
#                print(str(record.description).replace('.', '_').replace('/', '_'))
#                save_file.create_dataset(record.description,
#                                         data=embeddings_file2[
#                                             str(record.description).replace('.', '_').replace('/', '_')])

