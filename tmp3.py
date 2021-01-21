import pandas as pd

annotations = pd.read_csv('data/final_hard_set_annotations.csv')
results = pd.read_csv('data/deeploc_predictions.csv')

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
#results = results.drop_duplicates(subset=['Entry ID'])

print(results['Localization'].value_counts())

correct = 0
for index, row in results.iterrows():
    id = row['Entry ID']
    if row['Localization'] == str(annotations.loc[annotations['accession'] == id, 'location'].item()):
        correct += 1

print(len(results))
print(correct)
print(90/len(results)*100)
print(correct/len(results)*100)