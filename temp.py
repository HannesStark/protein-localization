import torch
from Bio import SeqIO
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import ToTensor, LabelToInt
from torchvision.transforms import transforms
from torch.nn.utils.rnn import pad_sequence
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import ToTensor, LabelToInt
import pandas as pd

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
from models.conv_avg_pool import ConvAvgPool
from utils.preprocess import remove_duplicates, deeploc_train_test, train_val_split, retrieve_by_id, reduce_embeddings

# train_val_split('data/split3_fasta_files/model_homreduced.fasta','data/split3_fasta_files',train_size=0.85)

# retrieve_by_id('data/split3_fasta_files/downloaded_without_annotations_model_homreduced.fasta', 'data/split3_fasta_files/model_sequences.fasta',
#               'data/split3_fasta_files/model_homreduced.fasta')

reduce_embeddings(['data/embeddings/train.h5', 'data/embeddings/val.h5','data/embeddings/test.h5'], 'data/embeddings', ['train_mean_max.h5','val_mean_max.h5','test_mean_max.h5'])
