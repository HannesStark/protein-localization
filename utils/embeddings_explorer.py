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
embedding, localization, solubility, meta = dataset[100]

emb = pd.DataFrame(embedding)

image = embedding *2
image = image + 0.5
image /= 1

plt.plot(np.mean(embedding,axis=1))
plt.show()

plt.plot(np.max(embedding,axis=1))
plt.show()

plt.rcParams['figure.dpi'] = 600
plt.imshow(image.T)
plt.show()

print('done')
#plt.hist(embedding, bins=20, histtype='bar', facecolor='blue')
#plt.show()

    #model.conv1.register_forward_hook(visualize_activation)
    #output = model(data_set[100][0].T.unsqueeze(0))
#
    #return
