# Protein Subcellular Localization Prediction :microscope:

PyTorch Implementation for predicting the subcellular localization of proteins. Achieves **86.01%** accuracy on
the [DeepLoc](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857) test set
(previous SOTA is 78%).

If you have any questions can't run some of the code, don't hesitate to open an issue or ask me via hannes.staerk@gmail.com
## Usage

Either train your own model or use the weights I provide in this repository and do only inference. We also provide
a webserver https://embed.protein.properties/ where you can just upload sequences and get the localization predictions (and more).

### 1. Get Protein Embeddings

The architecture works on embeddings that are generated from single sequences of amino acids without additional
evolutionary information as in profile embeddings.
Just [download](https://drive.google.com/drive/folders/1Qsu8uvPuWr7e0sOdjBAsWQW7KvHcSo1y?usp=sharing)
the embeddings and place them in `data_files`.

Alternatively you can generate the embedding files from ``.fasta`` files using the
[bio-embeddings](https://pypi.org/project/bio-embeddings/) library. For using the embeddings here, just replace the path
in the corresponding config file, such as `configs/light_attention.yaml` to point to your embeddings and remapping file
and set the parameter `key_format: hash`.

### 2. Setup environment

If you are using conda, you can install all dependencies like this. Otherwise, look at the setup below.

```
conda env create -f environment.yml
conda activate bio
```

### 3.1 Training

Make sure that you have
the [embeddings](https://drive.google.com/drive/folders/1Qsu8uvPuWr7e0sOdjBAsWQW7KvHcSo1y?usp=sharing)
placed in the files specified in `configs/light_attention.yaml` as described in step 1. Then start training:

```
python train.py --config configs/light_attention.yaml
tensorboard --logdir=runs --port=6006
```

You can now go to `localhost:6006` in your browser and watch the model train.

### 3.2 Inference

Either use your own tranined weights that were saved in the `runs` directory, or download
the [trained_model_weights](https://drive.google.com/drive/folders/13Ci6OmUaqUBpjWnG5nHLhg8HdJfbPi0p?usp=sharing)
and place the folder in the repository root. Running the following command will use the weights to generate the
predictions for the protein embeddings specified in `configs/inference.yaml` (setHARD currently).

```
python inference.py --config configs/inference.yaml
```

The predictions are then saved in the checkpoint in `trained_model_weights` as `predictions.txt` in the same order as
your input.

## Architecture

![architecture](https://github.com/HannesStark/protein-localization/blob/master/.architecture.png)

## Setup

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

You can use the conda environment file to install all of the above dependencies. Create the conda environment `bio` by
running:

```
conda env create -f environment.yml
```

### Reproduce exact results

You can use the respective configuration file to reproduce the results of the methods in the paper. The 10 randomly
generated seeds with which we trained 10 models of each method to get standard errors are:

```
[921, 969, 309, 559, 303, 451, 279, 624, 657, 702]
```

### Performance

The DeepLoc [data set](http://www.cbs.dtu.dk/services/DeepLoc/data.php) has 10 different subcellular localizations that
need to be classified. Meanwhile, `setHard` is a new Dataset with less redundancy and harder targets. The dataset
details can be found in our paper.

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
