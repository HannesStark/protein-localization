import os
from models import *  # required dont remove this
from torch.optim import *  # required dont remove this
from adabelief_pytorch import AdaBelief
import argparse
import yaml
import torch.nn as nn
from torchvision.transforms import transforms
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import *
from solvers.base_solver import BaseSolver


def inference(args):
    transform = transforms.Compose([SolubilityToInt(), ToTensor()])
    data_set = EmbeddingsLocalizationDataset(args.embeddings, args.remapping, unknown_solubility=False,
                                             transform=transform)

    # Needs "from models import *" to work
    model: nn.Module = globals()[args.model_type](embeddings_dim=data_set[0][0].shape[-1], **args.model_parameters)

    # Needs "from torch.optim import *" and "from models import *" to work
    solver = BaseSolver(model, args, globals()[args.optimizer], globals()[args.loss_function])
    solver.evaluation(data_set)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/inference.yaml')
    p.add_argument('--checkpoint', type=str, default='runs/FFN__30-10_10-44-51',
                   help='path to directory that contains a checkpoint')
    p.add_argument('--batch_size', type=int, default=16, help='samples that will be processed in parallel')
    p.add_argument('--n_draws', type=int, default=100,
                   help='how often to bootstrap from the dataset for variance estimation')
    p.add_argument('--log_iterations', type=int, default=100, help='log every log_iterations (-1 for no logging)')

    p.add_argument('--embeddings', type=str, default='data/embeddings/val_reduced.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--remapping', type=str, default='data/embeddings/val_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    args = p.parse_args()
    arg_dict = args.__dict__
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    # get the arguments from the yaml config file that is saved in the runs checkpoint
    data = yaml.load(open(os.path.join(args.checkpoint, 'train_arguments.yaml'), 'r'), Loader=yaml.FullLoader)
    for key, value in data.items():
        if key not in args.__dict__.keys():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


if __name__ == '__main__':
    inference(parse_arguments())
