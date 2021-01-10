 # Protein Subcellular Localization Prediction :microscope:

PyTorch Implementation for predicting the subcellular localization of proteins.
Achieves **83.37%** accuracy on the [DeepLoc](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857) test set
(previous SOTA is 78%). To reproduce just run ``train.py`` on the embedded DeepLoc 
[train set](http://www.cbs.dtu.dk/services/DeepLoc/data.php) and ``inference.py`` on 
the [test set](http://www.cbs.dtu.dk/services/DeepLoc/data.php).

The architecture works on embeddings that are generated from single sequences of amino acids without additional evolutionary
information as in profile embeddings. Such embeddings can be generated from ``.fasta`` files using the 
[bio-embeddings](https://pypi.org/project/bio-embeddings/) library.

### Quickstart
Change the training and validation file paths in ``configs/light_attention.yaml`` to point to your embeddings
and remapping files obtained from [bio-embeddings](https://pypi.org/project/bio-embeddings/) .
Then start the training like this.
```
conda env create -f environment.yml
conda activate bio
python train.py --config configs/light_attention.yaml
tensorboard --logdir=runs --port=6006
```
If everything works without errors, you can now go to `localhost:6006` in your browser and the model training.
### Architecture

![architecture](https://github.com/HannesStark/protein-localization/blob/master/architecture.png)

### Setup

Python 3 dependencies:

- pytorch (verified with version 1.6 and 1.7)
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

### Performance

The DeepLoc [data set](http://www.cbs.dtu.dk/services/DeepLoc/data.php) has 10 different subcellular localizations
that need to be classified.
Accuracy on the DeepLoc test set:

| Method | Accuracy |
| --- | --- |
| Ours | **83.37%** |
| DeepLoc | 77.97% |
| iLoc-Euk | 68.20% |
| YLoc | 61.22% |
| LocTree2 | 61.20% |
| SherLoc2 | 58.15% |

(Ours evaluated accross 10 different randomly chosen seeds)

(Numbers taken from the DeepLoc [paper](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857))
