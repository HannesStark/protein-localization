import os
import shutil

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

LOCALIZATION = ['Cell.membrane', 'Cytoplasm', 'Endoplasmic.reticulum', 'Golgi.apparatus', 'Lysosome/Vacuole',
                'Mitochondrion', 'Nucleus', 'Peroxisome', 'Plastid', 'Extracellular']
LOCALIZATION_abbrev = ['Mem', 'Cyt', 'End', 'Gol', 'Lys', 'Mit', 'Nuc', 'Per', 'Pla', 'Ext']


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


def experiment_checkpoint(run_directory: str, model, config_path: str):
    """
    Saves state dict of model and the used config file to the run_directory
    Args:
        run_directory: where to save
        model: pytorch nn.Module model
        config_path: path to the config file that was used for this run

    """
    torch.save(model.state_dict(), os.path.join(run_directory, 'model.pt'))
    shutil.copyfile(config_path, os.path.join(run_directory, os.path.basename(config_path)))
