from Bio import SeqIO
import pandas as pd

fasta_path = 'fasta_files/model_sequences.fasta'

identifiers = []
labels = []
sequences = []
for record in SeqIO.parse(fasta_path, "fasta"):
    identifiers.append(record.id)
    labels.append(record.description.split(' ')[1].split('-')[0])
    sequences.append(str(record.seq))
df = pd.DataFrame(list(zip(identifiers, labels, sequences)), columns=['identifier', 'label', 'seq'])
df['length'] = df['seq'].apply(lambda x: len(x))

print(df.describe())