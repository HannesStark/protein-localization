import os

import h5py
from Bio import SeqIO
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams["figure.figsize"] = (4,20)
#df = pd.read_csv('data/results/paper_tables.CSV')
#df = df[['method', 'acc_deeploc', 'stdev_acc_deeploc', 'acc_hard', 'stdev_acc_hard']]
#df = df.melt(id_vars=['method', 'stdev_acc_deeploc', 'stdev_acc_hard'])
#print(df)
#sns.barplot(data=df, x='value', y='method', hue='variable', palette='gray')
#plt.errorbar(x=df['value'], y=df['method'], xerr=df['stdev_acc_deeploc'], fmt='none', c='black', capsize=3)
#plt.xlim(40,100)
#plt.show()
from utils.general import LOCALIZATION
fasta_path = 'data/embeddings/new_hard_set_t5_remapping.fasta'
localizations = []
for record in tqdm(SeqIO.parse(open(fasta_path), 'fasta')):
    localization = record.description.split(' ')[2].split('-')[0]
    localization = LOCALIZATION.index(localization)
    localizations.append(localization)

draws = 1000
accuracies = []
arr = np.copy(localizations)
shuffle = np.array(localizations)
for i in tqdm(range(draws)):
    counter = 0
    np.random.shuffle(shuffle)
    accuracies.append((shuffle == arr).sum()/len(arr))


print(100*np.array(accuracies).mean())