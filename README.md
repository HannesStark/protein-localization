# Protein Subcellular Localization Prediction

PyTorch Implementation of different architectures for predicting the subcellular localization of proteins.
The architectures work on embeddings that are generated from single sequences of amino acids without additional evolutionary
information as in profile embeddings. Such embeddings can be generated from ``.fasta`` files using the 
[bio-embeddings](https://pypi.org/project/bio-embeddings/) library.

### Quickstart
Change the training and validation file paths in ``configs/simpleFFN.yaml`` to point to your embeddings
and remapping files obtained from [bio-embeddings](https://pypi.org/project/bio-embeddings/) .
Then start the training like this.
```
conda env create -f environment.yml
conda activate bio
python train.py --config configs/simpleFFN.yaml
tensorboard --logdir=runs --port=6006
```
If everything works without errors, you can now go to `localhost:6006` in your browser and the model training.
### Setup

Python 3 dependencies:

- torch <=1.4
- biopython
- h5py
- matplotlib
- seaborn
- pandas
- pyaml
- torchvision
- sklearn

You can use the conda environment file to install all of the above dependencies. Create the conda environment `bio` by running:
```
conda env create -f environment.yml
```
