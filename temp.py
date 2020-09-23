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
#
# def my_collate(batch):
#    data = [item[0] for item in batch]
#    target = torch.tensor([item[1] for item in batch])
#    data = pad_sequence(data, batch_first=True)
#    return (data, target)
#
# model = ConvAvgPool()
#
# dataset = EmbeddingsLocalizationDataset('data/embeddings/train.h5', 'data/embeddings/train_remapped.fasta',
#                                        transform=transforms.Compose([LabelToInt(), ToTensor()]))
#
#
# trainset = DataLoader(dataset=dataset,
#                      batch_size=4,
#                      shuffle=True,
#                      collate_fn=my_collate, # use custom collate function here
#                      pin_memory=True)
#
# trainiter = iter(trainset)
# embeddings, labels = trainiter.next()
# print(embeddings.shape)
# print(labels.shape)
from utils.preprocess import remove_duplicates, deeploc_train_test, train_val_split, retrieve_by_id

#train_val_split('data/fasta_files4/model_homreduced.fasta','data/fasta_files4',train_size=0.85)

#retrieve_by_id('data/fasta_files4/downloaded_without_annotations_model_homreduced.fasta', 'data/fasta_files4/model_sequences.fasta',
#               'data/fasta_files4/model_homreduced.fasta')

identifiers = []
labels = []
seqs = []

for record in SeqIO.parse('data/fasta_files4/model_homreduced.fasta', "fasta"):
    identifiers.append(record.id)
    seqs.append(record.seq)
    labels.append(record.description.split(' ')[1].split('-')[0])
df = pd.DataFrame(list(zip(identifiers, labels, seqs)), columns=['identifier', 'label', 'seq'])
print(df)
ids = df["seq"]
print(df[ids.isin(ids[ids.duplicated()])].sort_values(by="seq"))
