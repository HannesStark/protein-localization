import os

import h5py

embeddings_file = h5py.File(os.path.join('data/embeddings/test_t5.h5'), 'r')
print(len(embeddings_file.keys()))