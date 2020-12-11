import random
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
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
    barplot2 .set(xlabel='Accuracy', ylabel='')
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
    barplot.set(xlabel='Average accuracy over '+ str(args.n_draws) + ' draws', ylabel='')
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
