import os

from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
from utils.general import LOCALIZATION, LOCALIZATION_abbrev
import seaborn as sn

from utils.preprocess import remove_duplicates

sn.set_style('darkgrid')

fasta_path = '../data/fasta_files/new_hard_set.fasta'
fasta_path = '../data/fasta_files/val_homreduced.fasta'

filename = os.path.basename(fasta_path)
if 'test' in filename:
    color = 'orange'
else:
    color = 'lightskyblue'

identifiers = []
labels = []
sequences = []
solubility = []
for record in SeqIO.parse(fasta_path, "fasta"):
    identifiers.append(record.id)
    labels.append(record.description.split(' ')[1].split('-')[0])
    sequences.append(str(record.seq))
    solubility.append(record.description.split(' ')[1].split('-')[-1])
df = pd.DataFrame(list(zip(identifiers, labels, solubility, sequences)),
                  columns=['identifier', 'label', 'solubility', 'seq'])
# df = df[df['solubility'] != 'U']
df['length'] = df['seq'].apply(lambda x: len(x))

print('number sequences: ', len(df))

print(df['label'].value_counts()/len(df))

print('percentage of sequences larger than threshold AAs: {}'.format(100 * len(df[df['length'] > 1000]) / len(df)))
print('average length: ', df['length'].mean())



# visualize class prevalences
counts = df['label'].value_counts()
counts = pd.DataFrame({'Localization': counts.index,
                       "Number_Sequences": counts.array})
counts['ordering'] = counts['Localization'].apply(lambda x: LOCALIZATION.index(x))
counts['Localization'] = counts['Localization'].apply(lambda x: LOCALIZATION_abbrev[LOCALIZATION.index(x)])
counts = counts.sort_values(by=['ordering'])
barplot = sn.barplot(x='Number_Sequences', y='Localization', data=counts, ci=None)
barplot.set(xlabel='Number Sequences per Class', ylabel='')
plt.show()

cut_off = 2000
print(df['length'].max())
df[df['length'] < cut_off].hist(bins=50, ec='black', color=color)
plt.xlim((40, cut_off))  # there are only sequences longer than 40 in the datset
plt.title("Sequence lengths in {}".format(filename))
plt.xlabel("Sequence length")
plt.ylabel("Number sequences")
plt.show()

cut_off = 1500
axes = df[df['length'] < 1500].hist(by='label', ec='black', bins=30, color=color)
for rows in axes:
    for cell in rows:
        cell.set_xlim((40, cut_off))  # there are only sequences longer than 40 in the datset
        cell.tick_params(axis='x', which='both', labelbottom=False)
plt.show()
