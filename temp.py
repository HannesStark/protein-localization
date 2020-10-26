import inspect
import struct
from typing import List, Tuple

import h5py
import torch
from Bio import SeqIO
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import transforms
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from torchvision.transforms import transforms
from torch.nn.utils.rnn import pad_sequence
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
import pandas as pd
import torch
from torch.optim import *

# print(dataset[0])

# train_val_split('data/fasta_files/test_as_per_deeploc.fasta', train_size=0.5)

# identifiers = []
# labels = []
# for record in SeqIO.parse('fasta_files1/test.fasta', "fasta"):
#    identifiers.append(record.id)
#    labels.append(record.description.split(' ')[1].split('-')[0])
# df = pd.DataFrame(list(zip(identifiers, labels)), columns=['identifier', 'label'])
# print(df)

# create_annotations_csv('fasta_files1/test.fasta', '.')
# from models.conv_avg_pool import ConvAvgPool
#
# identifiers = []
# labels = []
# seqs = []
#
# for record in SeqIO.parse('data/split3_fasta_files/val.fasta', "fasta"):
#    identifiers.append(record.id)
#    seqs.append(record.seq)
#    labels.append(record.description.split(' ')[1].split('-')[0])
# df = pd.DataFrame(list(zip(identifiers, labels, seqs)), columns=['identifier', 'label', 'seq'])
# print(df)
# ids = df["seq"]
# print(df[ids.isin(ids[ids.duplicated()])].sort_values(by="seq"))
#


# train_val_split('data/split3_fasta_files/model_homreduced.fasta','data/split3_fasta_files',train_size=0.85)

# retrieve_by_id('data/split3_fasta_files/downloaded_without_annotations_model_homreduced.fasta', 'data/split3_fasta_files/model_sequences.fasta',
#               'data/split3_fasta_files/model_homreduced.fasta')
# sum_seqvec_embeddings(
#    ['data/seqvec_embeddings/3train.h5', 'data/seqvec_embeddings/3val.h5', 'data/seqvec_embeddings/3test.h5'],
#    'data/seqvec_embeddings', ['train.h5', 'val.h5', 'test.h5'])
# reduce_embeddings(['data/embeddings/train.h5', 'data/embeddings/val.h5','data/embeddings/test.h5'], 'data/embeddings', ['train_mean_max.h5','val_mean_max.h5','test_mean_max.h5'])
# id_labels_list = []
# for record in SeqIO.parse('data/embeddings/train_remapped.fasta', 'fasta'):
#    print(record.id)
#    localization, solubility  = record.description.split(' ')[2].split('-')[0]
#    if len(record.seq) <= 6000:
#        id_labels_list.append({'id': record.id, 'label': localization})

# def packed_padded_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[
#    torch.Tensor, torch.Tensor, torch.Tensor]:
#    """
#    Takes list of tuples with embeddings of variable sizes and pads them with zeros
#    Args:
#        batch: list of tuples with embeddings and the corresponding label
#
#    Returns: tuple of tensor of embeddings with [batchsize, length_of_longest_sequence, embeddings_dim]
#    and tensor of labels [batchsize, labels_dim]
#
#    """
#    embeddings = [item[0] for item in batch]
#    localization = torch.tensor([item[1] for item in batch])
#    solubility = torch.tensor([item[1] for item in batch]).float()
#    embeddings = pad_sequence(embeddings, batch_first=True)
#    return embeddings.permute(0, 2, 1), localization, solubility
#
# transform = transforms.Compose([LabelToInt(), ToTensor()])
# train_set = EmbeddingsLocalizationDataset('data/fasta_files/test_as_per_deeploc.fasta', 'data/fasta_files/test_remappings.fasta', 6000, transform)

from models import *

# class TryingDispatch():
#    def __init__(self, batchsize=3, n_layers=5, test=0):
#        print("initialized")
#        self.a = "teststring"
#        print(batchsize)
#        print(n_layers)
#        print(test)
#
#    def stringprint(self):
#        print(self.a)
#
#
# kwargs = {'batchsize': 5, 'n_layers': 17}
# instance = TryingDispatch()
# print(inspect.getfile(ConvAvgPool))
# classname = type(instance).__name__
# model = inspect.getsource(globals()[classname])
# print(model)
# import pywt
import numpy as np
import matplotlib.pyplot as plt
import cv2

# transform = transforms.Compose([LabelToInt()])
# dataset = EmbeddingsLocalizationDataset('data/embeddings/val_reduced.h5', 'data/embeddings/val_remapped.fasta', True, 6000,
#                                        transform)
##
# embedding, localization, solubility, meta = dataset[0]
# print(type(embedding))
#
# coeffs = pywt.wavedec(embedding, 'db1', axis=0)
# print(coeffs[0].shape)
# print(np.array(coeffs).shape)
# print(embedding.shape)

# for type in ['cat', 'sum', 'avg', 'max']:
#    for file in ['train_reduced.h5', 'val_reduced.h5', 'test_reduced.h5']:
#        combine_embeddings('data/embeddings/' + file,'data/seqvec_embeddings/' + file, type=type)


# for file in ['train.h5', 'val.h5', 'test.h5']:
#    position_cat_reduced('data/embeddings/' + file, 'data/combined_embeddings/' + 'cls_cat_reduced_' + file,
#                         position=0)

from utils.preprocess import remove_duplicates, deeploc_train_test, train_val_split, retrieve_by_id, reduce_embeddings, \
    sum_seqvec_embeddings, combine_embeddings, position_token_embeddings, position_cat_reduced

a = torch.tensor([[1, 2, 3, 0, 0, 0], [2, 3, 0, 0, 0, 0]]).double()

print(torch.softmax(a, dim=0))
