import os
from typing import Tuple, List

import h5py
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
import numpy as np


def reduce_embeddings(input_paths: List[str], output_dir: str, output_filenames: List[str]):
    """
    Pools along the length dimension of embeddings and saves them as h5
    Args:
        input_paths: paths to h5 files with arrays that should be pooled along their first dimension
        output_dir: directory where to save the files
        output_filenames: names as which the files should be saved

    Returns:

    """
    if len(input_paths) != len(output_filenames):
        raise ValueError('You cannot have more input files than output files')
    for i, input_path in enumerate(input_paths):
        output_path = os.path.join(output_dir, output_filenames[i])
        embeddings = h5py.File(input_path, 'r')
        reduced_embeddings = h5py.File(output_path, 'w')
        for key in tqdm(embeddings.keys()):
            embedding = embeddings[key][:]
            mean_pool = np.mean(embedding, axis=0)
            max_pool = np.max(embedding, axis=0)
            reduced = np.concatenate([mean_pool, max_pool], axis=-1)
            reduced_embeddings.create_dataset(key, data=reduced)


def combine_embeddings(file_1: str, file_2: str, output_dir: str = 'data/combined_embeddings', type: str = 'sum'):
    """
    Combine embeddings of the same size and save them to a file with the name of file1 and the combination type
     into the output_dir
    Args:
        file_1: string to .h5 file of first embeddings
        file_2: string to .h5 file of other embeddings
        output_dir: string to ouput_directory
        type: how to combine the embeddigns [cat, sum, avg, max]

    Returns:

    """
    output_path = os.path.join(output_dir, type + '_' + os.path.basename(file_1))
    embeddings_1 = h5py.File(file_1, 'r')
    embeddings_2 = h5py.File(file_2, 'r')
    combined_embeddings = h5py.File(output_path, 'w')
    print('combining ', file_1, ' with ', file_2, ' into ', output_path)
    for key in tqdm(embeddings_1.keys()):
        embedding1 = embeddings_1[key][:]
        embedding2 = embeddings_2[key][:]

        if type == 'cat':
            combined_embedding = np.concatenate([embedding1, embedding2], axis=-1)
        elif type == 'sum':
            combined_embedding = embedding1 + embedding2
        elif type == 'avg':
            combined_embedding = embedding1 + embedding2
            combined_embeddings /= 2
        elif type == 'max':
            combined_embedding = np.maximum(embedding1, embedding2)
        else:
            raise ValueError('Specified type does not exist')
        combined_embeddings.create_dataset(key, data=combined_embedding)


def sum_seqvec_embeddings(input_paths: List[str], output_dir: str, output_filenames: List[str]):
    """
    sums the layers of the seqvec embeddings and saves the resulting embeddings
    Args:
        input_paths: paths to h5 files with arrays that should summed
        output_dir: directory where to save the files
        output_filenames: names as which the files should be saved

    Returns:

    """
    if len(input_paths) != len(output_filenames):
        raise ValueError('You cannot have more input files than output files')
    for i, input_path in enumerate(input_paths):
        output_path = os.path.join(output_dir, output_filenames[i])
        embeddings = h5py.File(input_path, 'r')
        summed_embeddings = h5py.File(output_path, 'w')
        for key in tqdm(embeddings.keys()):
            embedding = embeddings[key][:]
            summed_embeddings.create_dataset(key, data=np.sum(embedding, axis=0))


def remove_duplicates(fasta_path: str, output_path: str):
    """removes duplicates from a fasta file and saves a new fasta file as "duplicates_removed + original_filename.fasta"

    Args:
        fasta_path: path to fasta file from which duplicates should be removed
        output_path: where to save the fasta file with the duplicates removed.

    Returns:

    """
    record_seq = []
    records = []
    for record in tqdm(SeqIO.parse(fasta_path, "fasta")):
        if record.seq not in record_seq:
            record_seq.append(record.seq)
            records.append(record)
    SeqIO.write(records, os.path.join(output_path, 'duplicates_removed' + os.path.basename(fasta_path)), 'fasta')


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


def deeploc_train_test(deeploc_path: str, output_dir: str = 'data'):
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
    SeqIO.write(test, os.path.join(output_dir, 'test_as_per_deeploc.fasta'), 'fasta')


def train_val_split(fasta_path: str, output_dir: str = 'data', train_size: float = 0.8):
    """
    Splits a fasta file into train and validation fasta files and saves them to the output_dir
    Args:
        fasta_path: path to .fasta file to split
        output_dir: directory to save the test.fasta and val.fasta file
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


def retrieve_by_id(ids_fasta: str, target_fasta: str, output_path: str):
    """
        pick sequences from target fasta that have the same ids as ids_fasta and save them to output path
    Args:
        ids_fasta: path to fasta file with ids that you want to pick from the target fasta
        target_fasta: path to fasta file from which records should be retrieved
        output_path: path to save the retrieved records as .fasta

    Returns:

    """

    ids = []
    for record in SeqIO.parse(ids_fasta, 'fasta'):
        ids.append(record.id)

    records = []
    for record in SeqIO.parse(target_fasta, 'fasta'):
        if record.id in ids:
            records.append(record)

    SeqIO.write(records, output_path, 'fasta')


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
