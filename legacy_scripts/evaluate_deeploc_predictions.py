import os

import pandas as pd
from sklearn.metrics import matthews_corrcoef

deep_loc_results_dfs = []
path = '../data/results/deeploc_hard_set'
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        csv = pd.read_csv(os.path.join(path, filename))
        deep_loc_results_dfs.append(csv)
results = pd.concat(deep_loc_results_dfs, axis=0, ignore_index=True)

results = pd.read_csv('../data/deeploc_predictions.csv')
annotations = pd.read_csv('../data/final_hard_set_annotations.csv')
annotations["location"] = annotations["location"].map({"Endoplasmic.reticulum": "Endoplasmic reticulum",
                                                       "Cell.membrane": 'Cell membrane',
                                                       'Golgi.apparatus': 'Golgi apparatus',
                                                       'Cytoplasm': 'Cytoplasm',
                                                       'Lysosome/Vacuole': 'Lysosome/Vacuole',
                                                       'Mitochondrion': 'Mitochondrion',
                                                       'Nucleus': 'Nucleus',
                                                       'Peroxisome': 'Peroxisome',
                                                       'Plastid': 'Plastid',
                                                       'Extracellular': 'Extracellular'
                                                       })
annotations = annotations.drop_duplicates(subset=['accession'])
# results = results.drop_duplicates(subset=['Entry ID'])

print(results['Localization'].value_counts())

correct = 0
counter = 0
li = []
for index, row in results.iterrows():

    id = row['Entry ID']
    li.append(annotations.loc[annotations['accession'] == id, 'location'].item())
    if row['Localization'] == str(annotations.loc[annotations['accession'] == id, 'location'].item()):
        correct += 1

print(len(li))
print('length: ', len(results))
print('correct: ', correct)
print(correct / len(results) * 100)
combined = pd.merge(annotations, results, left_on='accession', right_on='Entry ID')
combined = combined[['accession', 'location', 'Localization']]
combined['correct_prediction'] = combined['location'] == combined['Localization']
combined.columns = ['accession', 'true_label', 'deep_loc_prediction', 'correct']
print(combined)
print(matthews_corrcoef(combined['true_label'], combined['deep_loc_prediction']))
print(combined['correct'].value_counts(normalize=True).mul(100).astype(str) + '%')
combined.to_csv('deep_loc_results.csv')
