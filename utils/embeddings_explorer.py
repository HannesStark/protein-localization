import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import transforms

from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import LabelToInt

transform = transforms.Compose([LabelToInt()])
dataset = EmbeddingsLocalizationDataset('../data/embeddings/val.h5', '../data/embeddings/val_remapped.fasta', True,
                                        6000,
                                        transform)
embedding, localization, solubility, meta = dataset[0]

emb = pd.DataFrame(embedding)

ones = np.ones([200,200])*0.5
cv2.imshow('embedding', ones)
plt.hist(embedding, bins=20, histtype='bar', facecolor='blue')
plt.show()
