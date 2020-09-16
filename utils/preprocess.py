import os

from Bio import SeqIO
import pandas as pd
from tqdm import tqdm


def remove_duplicates(fasta_path: str, output_dir: str = 'data'):
    """removes duplicates from a fasta file and saves a new fasta file as "duplicates_removed + original_filename.fasta"

    Args:
        fasta_path: path to fasta file from which duplicates should be removed
        output_dir: where to save the fasta file with the duplicates removed. 'data' by default

    Returns:

    """
    record_seq = []
    records = []
    for record in tqdm(SeqIO.parse(fasta_path, "fasta")):
        if record.seq not in record_seq:
            record_seq.append(record.seq)
            records.append(record)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    SeqIO.write(records, os.path.join(output_dir, 'duplicates_removed' + os.path.basename(fasta_path)), 'fasta')


def create_annotations_csv(fasta_path: str, csv_path: str):
    """Save mapping csv between sequence ids and their labels

    Args:
        fasta_path: path to fasta file with the same annotation format as
        http://www.cbs.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta
        csv_path: where to save the csv, for intance annotations.csv

    Returns:

    """
    identifiers = []
    labels = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        identifiers.append(record.id)
        labels.append(record.description.split(' ')[1].split('-')[0])
    df = pd.DataFrame(list(zip(identifiers, labels)), columns=['identifier', 'label'])

    if not os.path.exists(os.path.dirname(os.path.abspath(csv_path))):
        os.mkdir(os.path.dirname(os.path.abspath(csv_path)))
    df.to_csv(csv_path)


def deeploc_train_test(deeploc_path: str, output_dir: str):
    """Splits the deeploc fasta http://www.cbs.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta
     into train and test set and saves it to the output_dir

    Args:
        deeploc_path: path to deeploc .fasta file http://www.cbs.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta
        output_dir: directory to save the train fasta and test fasta file
    """
    train = []
    test = []
    for record in SeqIO.parse(deeploc_path, "fasta"):
        info = record.description.split(' ')
        if info[-1] == 'test':
            test.append(record)
        else:
            train.append(record)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    SeqIO.write(train, os.path.join(output_dir, 'train.fasta'), 'fasta')
    SeqIO.write(test, os.path.join(output_dir, 'test.fasta'), 'fasta')
