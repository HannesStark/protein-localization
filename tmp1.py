import numpy as np

mapping = {"Endoplasmic.reticulum": "Endoplasmic reticulum",
           "Cell.membrane": 'Cell membrane',
           'Golgi.apparatus': 'Golgi apparatus',
           'Cytoplasm': 'Cytoplasm',
           'Lysosome/Vacuole': 'Lysosome/Vacuole',
           'Mitochondrion': 'Mitochondrion',
           'Nucleus': 'Nucleus',
           'Peroxisome': 'Peroxisome',
           'Plastid': 'Plastid',
           'Extracellular': 'Extracellular'
           }
switched_mapping = {y: x for x, y in mapping.items()}

import os
from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
from utils.general import LOCALIZATION, LOCALIZATION_abbrev
import seaborn as sn

from utils.preprocess import remove_duplicates

sn.set_style('darkgrid')

deeploc_preds = pd.read_csv('deep_loc_results_BLOSUM.csv')

deeploc_preds['true_label'] = deeploc_preds['true_label'].map(switched_mapping).map(lambda x: LOCALIZATION.index(x))
deeploc_preds['deep_loc_prediction'] = deeploc_preds['deep_loc_prediction'].map(switched_mapping).map(
    lambda x: LOCALIZATION.index(x))

deeploc_preds = np.array(
    [deeploc_preds['deep_loc_prediction'], deeploc_preds['true_label'], deeploc_preds['correct']]).T

results = np.load(
    'runs/..finalModels/FirstAttention_927_15-11_08-38-57/results_array_new_hard_set.npy')  # predictions first and true label second

# for stuff loaded from csv as produced by another tmp
results = deeploc_preds

fasta_path = 'data/fasta_files/deeploc_data.fasta'

filename = os.path.basename(fasta_path)

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
counts = df['label'].value_counts()
counts = pd.DataFrame({'Localization': counts.index,
                       "Number_Sequences": counts.array})
counts['ordering'] = counts['Localization'].apply(lambda x: LOCALIZATION.index(x))
counts['Localization'] = counts['Localization'].apply(lambda x: LOCALIZATION_abbrev[LOCALIZATION.index(x)])
counts = counts.sort_values(by=['ordering'])
deep_loc_distribution = np.array(counts)[:, 1] * 30

print(np.equal(results[:, 1], results[:, 0]).sum() / len(results))

print(results)
res_by_class = [results[results[:, 1] == k] for k in np.unique(results[:, 1])]

samples = []
for i, res in enumerate(res_by_class):
    corrects = np.equal(res[:, 1], res[:, 0])
    samples.append(np.random.choice(corrects, deep_loc_distribution[i]))

samples = np.concatenate(samples)
print(len(samples))
print(samples.sum() / len(samples))
