import os
import random
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_theme()
from models import *  # imports all classes in the models directory

LOCALIZATION = ['Cell.membrane', 'Cytoplasm', 'Endoplasmic.reticulum', 'Golgi.apparatus', 'Lysosome/Vacuole',
                'Mitochondrion', 'Nucleus', 'Peroxisome', 'Plastid', 'Extracellular']
LOCALIZATION_abbrev = ['Mem', 'Cyt', 'End', 'Gol', 'Lys', 'Mit', 'Nuc', 'Per', 'Pla', 'Ext']

SOLUBILITY = ['M', 'S', 'U']


def seed_all(seed):
    if not seed:
        seed = 0

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def annotation_transfer(evaluation_set: Dataset, lookup_set: Dataset, accuracy_threshold: float,
                        writer: SummaryWriter = None, filename: str = ''):
    '''
    Uses knn for embedding space similarity based annotation transfer
    Args:
        evaluation_set: dataset for which to make predictions.
        lookup_dataset: annotated dataset from which to transfer the annotations.
        accuracy_threshold: threshold to determine the disitance under which annotation transfer will be used.

    Returns:
        tuple of array[predictions, labels], and indices for which high confidence predictions were possible
    '''

    lookup_loader = DataLoader(lookup_set, batch_size=len(lookup_set), collate_fn=numpy_collate_reduced)
    evaluation_loader = DataLoader(evaluation_set, batch_size=len(evaluation_set), collate_fn=numpy_collate_reduced)

    lookup_data = next(iter(lookup_loader))  # tuple of embedding, localization, solubility, metadata
    evaluation_data = next(iter(evaluation_loader))  # tuple of embedding, localization, solubility, metadata

    # if we have per residue embeddings they have an additional length dim so we sum them up to get the reduced embeddings
    if len(evaluation_data[0][0].shape) == 2:
        evaluation_data[0] = evaluation_data[0].mean(axis=-2)  # average out the length dimension

    classifier = KNeighborsClassifier(n_neighbors=1, p=1) # use 1 neighbor and L1 distance
    classifier.fit(lookup_data[0], lookup_data[1])
    predictions = classifier.predict(evaluation_data[0])
    distances, _ = classifier.kneighbors(evaluation_data[0])

    # here we want to find out below which distance we still get an accuracy higher than accuracy_threshold
    cutoffs = np.linspace(distances.min(), distances.max(), 500)  # check 500 different cutoff possibilities
    results = np.array([predictions, evaluation_data[1], distances.squeeze()]).T
    accuracies = []
    number_sequences = []
    lower_accuracy_found = False
    high_accuracy_predictions = None
    low_accuracy_mask = None
    for cutoff in cutoffs:
        high_accuracy_mask = results[:, 2] <= cutoff
        below_cutoff = results[high_accuracy_mask]
        accuracy = np.equal(below_cutoff[:, 0], below_cutoff[:, 1]).sum() / len(below_cutoff)
        accuracies.append(accuracy * 100)
        if accuracy <= accuracy_threshold:
            lower_accuracy_found = True
        if accuracy >= accuracy_threshold and not lower_accuracy_found:
            high_accuracy_predictions = below_cutoff
            low_accuracy_mask = np.invert(high_accuracy_mask)
        number_sequences.append(len(below_cutoff))

    if writer:
        df = pd.DataFrame(np.array([cutoffs, accuracies, number_sequences]).T,
                          columns=["distance", "accuracy", 'number sequences'])
        sn.lineplot(data=df, x="distance", y="accuracy")
        plt.axhline(y=accuracy_threshold * 100, linewidth=1, color='black')
        plt.savefig(os.path.join(writer.log_dir, 'embedding_distances_' + filename + '.png'))
        plt.clf()
        sn.lineplot(data=df, x="number sequences", y="accuracy")
        plt.axhline(y=accuracy_threshold * 100, linewidth=1, color='black')
        plt.savefig(os.path.join(writer.log_dir, 'embedding_distances_num_sequences_' + filename + '.png'))

    return high_accuracy_predictions, np.where(low_accuracy_mask)[0]


def tensorboard_class_accuracies(train_results: np.ndarray, val_results: np.ndarray, writer: SummaryWriter, args,
                                 step: int):
    """
    Turns results into two confusion matrices, plots them side by side and writes them to tensorboard
    Args:
        train_results: [n_samples, 2] the first column is the prediction the second is the true label
        val_results: [n_samples, 2] the first column is the prediction the second is the true label
        writer: a pytorch summary writer
        step: the step at which the confusion matrix should be displayed

    Returns:

    """
    if args.target == 'sol':
        train_confusion = confusion_matrix(train_results[:, 3], train_results[:, 2])  # confusion matrix for train
        val_confusion = confusion_matrix(val_results[:, 3], val_results[:, 2])  # confusion matrix for validation
        labels = SOLUBILITY[:2]
    else:
        train_confusion = confusion_matrix(train_results[:, 1], train_results[:, 0])  # confusion matrix for train
        val_confusion = confusion_matrix(val_results[:, 1], val_results[:, 0])  # confusion matrix for validation
        labels = LOCALIZATION

    train_class_accuracies = np.diag(train_confusion) / train_confusion.sum(1)
    val_class_accuracies = np.diag(val_confusion) / val_confusion.sum(1)

    train_class_accuracies = pd.DataFrame({'Localization': labels,
                                           "Accuracy": train_class_accuracies})
    val_class_accuracies = pd.DataFrame({'Localization': labels,
                                         "Accuracy": val_class_accuracies})
    sn.set_style('darkgrid')
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].set_title('Training')
    ax[1].set_title('Validation')
    barplot1 = sn.barplot(x="Accuracy", y="Localization", ax=ax[0], data=train_class_accuracies, ci=None)
    barplot1.set(xlabel='Accuracy', ylabel='')
    barplot1.axvline(1)
    barplot2 = sn.barplot(x="Accuracy", y="Localization", ax=ax[1], data=val_class_accuracies, ci=None)
    barplot2.set(xlabel='Accuracy', ylabel='')
    barplot2.axvline(1)
    plt.tight_layout()
    writer.add_figure('Class accuracies ', fig, global_step=step)


def tensorboard_confusion_matrix(train_results: np.ndarray, val_results: np.ndarray, writer: SummaryWriter, args,
                                 step: int):
    """
    Turns results into two confusion matrices, plots them side by side and writes them to tensorboard
    Args:
        train_results: [n_samples, 2] the first column is the prediction the second is the true label
        val_results: [n_samples, 2] the first column is the prediction the second is the true label
        writer: a pytorch summary writer
        step: the step at which the confusion matrix should be displayed

    Returns:

    """
    if args.target == 'sol':
        train_confusion = confusion_matrix(train_results[:, 3], train_results[:, 2])  # confusion matrix for train
        val_confusion = confusion_matrix(val_results[:, 3], val_results[:, 2])  # confusion matrix for validation
        train_cm = pd.DataFrame(train_confusion, SOLUBILITY[:2], SOLUBILITY[:2])
        val_cm = pd.DataFrame(val_confusion, SOLUBILITY[:2], SOLUBILITY[:2])
    else:
        train_confusion = confusion_matrix(train_results[:, 1], train_results[:, 0])  # confusion matrix for train
        val_confusion = confusion_matrix(val_results[:, 1], val_results[:, 0])  # confusion matrix for validation
        train_cm = pd.DataFrame(train_confusion, LOCALIZATION_abbrev, LOCALIZATION_abbrev)
        val_cm = pd.DataFrame(val_confusion, LOCALIZATION_abbrev, LOCALIZATION_abbrev)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].set_title('Training')
    ax[1].set_title('Validation')
    sn.heatmap(train_cm, ax=ax[0], annot=True, cmap='Blues', fmt='g', rasterized=False)
    sn.heatmap(val_cm, ax=ax[1], annot=True, cmap='YlOrBr', fmt='g', rasterized=False)
    writer.add_figure('Confusion Matrix ', fig, global_step=step)


def plot_class_accuracies(accuracy, stderr, path, args=None):
    """
    Create seaborn plot and save it to path
    Args:
        accuracy: accuracies
        stderr: standard errors
        path: where to save the plot

    Returns:

    """
    labels = SOLUBILITY if args.target == 'sol' else LOCALIZATION
    df = pd.DataFrame({'Localization': labels,
                       "Accuracy": accuracy,
                       "std": stderr})
    sn.set_style('darkgrid')
    barplot = sn.barplot(x="Accuracy", y="Localization", data=df, ci=None)
    barplot.set(xlabel='Average accuracy over ' + str(args.n_draws) + ' draws', ylabel='')
    barplot.axvline(1)
    plt.errorbar(x=df['Accuracy'], y=labels, xerr=df['std'], fmt='none', c='black', capsize=3)
    plt.tight_layout()
    plt.savefig(path)


def padded_permuted_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings with [batchsize, length_of_longest_sequence, embeddings_dim]
    and tensor of labels [batchsize, labels_dim] and metadate collated according to default collate

    """
    embeddings = [item[0] for item in batch]
    localization = torch.tensor([item[1] for item in batch])
    solubility = torch.tensor([item[2] for item in batch])
    metadata = [item[3] for item in batch]
    metadata = torch.utils.data.dataloader.default_collate(metadata)
    embeddings = pad_sequence(embeddings, batch_first=True)
    return embeddings.permute(0, 2, 1), localization, solubility, metadata


def numpy_collate_reduced(batch: List[Tuple[np.array, np.array, np.array, dict]]) -> Tuple[
    np.array, np.array, np.array, dict]:
    """
    Collate function for reduced per protein embedding that returns numpy arrays intead of tensors
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of np.arrays of embeddings with [batchsize, embeddings_dim] and the rest in batched form

    """
    embeddings = [np.array(item[0]) for item in batch]
    localization = [np.array(item[1]) for item in batch]
    solubility = [item[2] for item in batch]
    metadata = [item[3] for item in batch]
    metadata = torch.utils.data.dataloader.default_collate(metadata)
    return embeddings, localization, solubility, metadata


def packed_padded_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings with [batchsize, length_of_longest_sequence, embeddings_dim]
    and tensor of labels [batchsize, labels_dim]

    """
    embeddings = [item[0] for item in batch]
    localization = torch.tensor([item[1] for item in batch])
    solubility = torch.tensor([item[1] for item in batch]).float()
    embeddings = pad_sequence(embeddings, batch_first=True)
    return embeddings.permute(0, 2, 1), localization, solubility


def normalize(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr
