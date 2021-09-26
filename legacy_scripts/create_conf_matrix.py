import os

import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Subset

from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from torchvision.transforms import transforms

from datasets.transforms import SolubilityToInt
from utils.general import numpy_collate_for_reduced
import matplotlib.pyplot as plt

os.chdir('C:/Users/HannesStark\\projects\\protein-localization')

denovo_predictions = np.load('data/results/results_array_inference.npy')

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from utils.general import plot_confusion_matrix, LOCALIZATION_abbrev

# order according to prevalence in deeploc
LOCALIZATION_abbrev_reordered = ['Nuc', 'Cyt', 'Ext', 'Mit', 'Mem', 'End', 'Pla', 'Gol', 'Lys', 'Per']
perm = []
for abbrev in LOCALIZATION_abbrev:
    perm.append(LOCALIZATION_abbrev_reordered.index(abbrev))
print(perm)
print(len(denovo_predictions))
print(denovo_predictions[:, 1].shape)
reordered_pred = np.zeros_like(denovo_predictions[:, 1])
reordered_true = np.zeros_like(reordered_pred)
for i in range(len(denovo_predictions[:, 1])):
    reordered_true[i] = perm[denovo_predictions[:, 1][i]]
    reordered_pred[i] = perm[denovo_predictions[:, 0][i]]
confusion = confusion_matrix(reordered_true, reordered_pred,
                             normalize='true')  # normalize='true' for relative freq

confusion = np.array(confusion, dtype=float)
confusion = np.concatenate([confusion.sum(0)[None, :], confusion])
confusion[confusion < 0.01] = np.nan
# confusion[confusion == 0.] = np.nan
confusion_df = pd.DataFrame(confusion, ['sum'] + LOCALIZATION_abbrev_reordered, LOCALIZATION_abbrev_reordered)
sn.set_style("whitegrid")

# fmt='.2f' for relative freq
# fmt='g' for absolute freq
hm = sn.heatmap(confusion_df, annot=True, cmap='gray_r', fmt='.2f', rasterized=False, cbar=False)
ax = hm.axes
plt.hlines([1], 0, 10, colors=['orange'],linewidths=[2.5])
for tick in ax.yaxis.get_majorticklabels():
    tick.set_verticalalignment("center")
plt.show()
