 # Protein Subcellular Localization Prediction :microscope:

PyTorch Implementation for predicting the subcellular localization of proteins.
Achieves **86.01%** accuracy on the [DeepLoc](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857) test set
(previous SOTA is 78%). To reproduce just run ``train.py`` on the embedded DeepLoc 
[train set](http://www.cbs.dtu.dk/services/DeepLoc/data.php) and ``inference.py`` on 
the [test set](http://www.cbs.dtu.dk/services/DeepLoc/data.php).

The architecture works on embeddings that are generated from single sequences of amino acids without additional evolutionary
information as in profile embeddings. Such embeddings can be generated from ``.fasta`` files using the 
[bio-embeddings](https://pypi.org/project/bio-embeddings/) library.

### Quickstart
Change the training and validation file paths in ``configs/light_attention.yaml`` to point to your embeddings
and remapping files obtained from [bio-embeddings](https://pypi.org/project/bio-embeddings/).
The library has many jupyter [notebook examples](https://github.com/sacdallago/bio_embeddings/blob/develop/notebooks/extract_localization_from_ProtBert_using_light_attention.ipynb) on how to easily generate those embeddings that can simply be run in google colab.
Then start the training like this.
```
conda env create -f environment.yml
conda activate bio
python train.py --config configs/light_attention.yaml
tensorboard --logdir=runs --port=6006
```
If everything works without errors, you can now go to `localhost:6006` in your browser and watch the model train.

### Architecture

![architecture](https://github.com/HannesStark/protein-localization/blob/master/.architecture.png)

### Setup

Python 3 dependencies:

- pytorch 
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

### Reproduce exact results
You can use the respective configuration file to reproduce the results of the methods in the paper. The 10 randomly 
generated seeds with which we trained 10 models of each method to get standard errors are:
```
[921, 969, 309, 559, 303, 451, 279, 624, 657, 702]
```
Besides using the config files and `train.py` you can also use `inference.py` where you can specify a list of trained
checkpoints and evaluate them on different test sets. Again, have a look at `configs/inference.yaml` for an example
of how to use it.


### Performance

The DeepLoc [data set](http://www.cbs.dtu.dk/services/DeepLoc/data.php) has 10 different subcellular localizations
that need to be classified.
Meanwhile, `setHard` is a new Dataset with less redundancy and harder targets. The dataset details can be found in our paper.

![accuracyplot](https://github.com/HannesStark/protein-localization/blob/master/.accuracy.png)


Accuracy on the DeepLoc test set:

| Method | Accuracy |
| --- | --- |
| Ours | **86.01%** |
| DeepLoc | 77.97% |
| iLoc-Euk | 68.20% |
| YLoc | 61.22% |
| LocTree2 | 61.20% |
| SherLoc2 | 58.15% |

(Ours evaluated accross 10 different randomly chosen seeds)
(Numbers taken from the DeepLoc [paper](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857))
