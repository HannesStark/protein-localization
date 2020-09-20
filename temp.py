from Bio import SeqIO
from torchvision.transforms import transforms
import pandas as pd
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import ToTensor, LabelToInt, LabelOneHot
from models.simple_ffn import SimpleFFN
from utils.preprocess import deeploc_train_test, train_val_split, create_annotations_csv
import numpy as np


#dataset = EmbeddingsLocalizationDataset('embeddings/test_reduced.h5', 'embeddings/test_remapped.fasta',
#                                        transform=transforms.Compose([LabelOneHot(), ToTensor()]))

#print(dataset[0])

train_val_split('data/fasta_files/test_as_per_deeploc.fasta', train_size=0.5)

#identifiers = []
#labels = []
#for record in SeqIO.parse('old_fasta_files/test.fasta', "fasta"):
#    identifiers.append(record.id)
#    labels.append(record.description.split(' ')[1].split('-')[0])
#df = pd.DataFrame(list(zip(identifiers, labels)), columns=['identifier', 'label'])
#print(df)

#create_annotations_csv('old_fasta_files/test.fasta', '.')
