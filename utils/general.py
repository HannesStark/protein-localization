import os
import shutil
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

LOCALIZATION = ['Cell.membrane', 'Cytoplasm', 'Endoplasmic.reticulum', 'Golgi.apparatus', 'Lysosome/Vacuole',
                'Mitochondrion', 'Nucleus', 'Peroxisome', 'Plastid', 'Extracellular']
LOCALIZATION_abbrev = ['Mem', 'Cyt', 'End', 'Gol', 'Lys', 'Mit', 'Nuc', 'Per', 'Pla', 'Ext']

SOLUBILITY = ['M', 'S', 'U']


def tensorboard_confusion_matrix(train_results: np.ndarray, val_results: np.ndarray, writer: SummaryWriter, step: int):
    """
    Turns results into two confusion matrices, plots them side by side and writes them to tensorboard
    Args:
        train_results: [n_samples, 2] the first column is the prediction the second is the true label
        val_results: [n_samples, 2] the first column is the prediction the second is the true label
        writer: a pytorch summary writer
        step: the step at which the confusion matrix should be displayed

    Returns:

    """

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


def experiment_checkpoint(run_directory: str, model, optimizer, epoch: int, config_path: str):
    """
    Saves state dict of model and the used config file to the run_directory
    Args:
        run_directory: where to save
        model: pytorch nn.Module model
        optimizer:
        config_path: path to the config file that was used for this run

    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(run_directory, 'checkpoint.pt'))
    shutil.copyfile(config_path, os.path.join(run_directory, os.path.basename(config_path)))


def padded_permuted_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[
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
