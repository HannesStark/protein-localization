import os


import numpy as np
#
#res = np.array([80.56, 80.39, 79.91, 80.28, 80.33])
#
#print(res.mean())
#print(res.std())
from utils.preprocess import create_annotations_csv

create_annotations_csv('data/fasta_files/train.fasta', 'data/fasta_files/train_annotations.csv')

create_annotations_csv('data/fasta_files/model_sequences.fasta', 'data/fasta_files/test_annotations.csv')