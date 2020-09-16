import os
from typing import Tuple

from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
import numpy as np


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


def deeploc_train_test(deeploc_path: str, output_dir: str = 'fasta_files'):
    """Splits the deeploc fasta http://www.cbs.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta
     into train and test set and saves it to the output_dir

    Args:
        deeploc_path: path to deeploc .fasta file http://www.cbs.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta
        output_dir: directory to save the train fasta and test fasta file
    """
    model_sequences = []
    test = []
    for record in SeqIO.parse(deeploc_path, "fasta"):
        info = record.description.split(' ')
        if info[-1] == 'test':
            test.append(record)
        else:
            model_sequences.append(record)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    SeqIO.write(model_sequences, os.path.join(output_dir, 'model_sequences.fasta'), 'fasta')
    SeqIO.write(test, os.path.join(output_dir, 'test.fasta'), 'fasta')


def train_val_split(fasta_path: str, output_dir: str = 'fasta_files', train_size: float = 0.8):
    """
    Splits a fasta file into train and validation fasta files and saves them to the output_dir
    Args:
        fasta_path: path to .fasta file to split
        output_dir: directory to save the train.fasta and val.fasta file
        train_size: ratio between train and validation set

    Returns:

    """

    number_sequences = len([1 for line in open(fasta_path) if line.startswith(">")])
    train_indices, val_indices = disjoint_indices(number_sequences, train_size, random=True)

    train = []
    val = []
    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if i in train_indices:
            train.append(record)
        else:
            val.append(record)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    SeqIO.write(train, os.path.join(output_dir, 'train.fasta'), 'fasta')
    SeqIO.write(val, os.path.join(output_dir, 'val.fasta'), 'fasta')


def disjoint_indices(size: int, ratio: float, random=True) -> Tuple[np.ndarray, np.ndarray]:
    """Creates disjoint sets of indices where all indices together are size many indices. The first set of the returned
        tuple has size*ratio many indices and the second one has size*(ratio-1) many indices.

    Args:
        audio: total number of indices returned. First and second array together
        ratio: relative sizes between the returned index arrays
        random: should the indices be randomly sampled

    Returns:
        indices*ratio:
    """
    if random:
        train_indices = np.random.choice(np.arange(size), int(size * ratio), replace=False)
        val_indices = np.setdiff1d(np.arange(size), train_indices, assume_unique=True)
        return train_indices, val_indices

    indices = np.arange(size)
    split_index = int(size * ratio)
    return indices[:split_index], indices[split_index:]
