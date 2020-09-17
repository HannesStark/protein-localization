import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from torchvision.transforms import transforms

from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import LabelOneHot, ToTensor
from models.simple_ffn import SimpleFFN


def train(args):
    transform = transforms.Compose([LabelOneHot(), ToTensor()])
    train_set = EmbeddingsLocalizationDataset(args.train_embeddings, args.train_remapping, transform)
    val_set = EmbeddingsLocalizationDataset(args.val_embeddings, args.val_remapping, transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    model = SimpleFFN()


    print(train_set[0])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/simpleFFN.yaml')
    parser.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')

    parser.add_argument('--train_embeddings', type=str, default='embeddings/train.h5',
                        help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    parser.add_argument('--train_remapping', type=str, default='embeddings/train_remapped.fasta',
                        help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    parser.add_argument('--val_embeddings', type=str, default='embeddings/val.h5',
                        help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    parser.add_argument('--val_remapping', type=str, default='embeddings/val_remapped.fasta',
                        help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    parser.add_argument('--test_embeddings', type=str, default='embeddings/test.h5',
                        help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    parser.add_argument('--test_remapping', type=str, default='embeddings/test_remapped.fasta',
                        help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    args = parser.parse_args()
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        delattr(args, 'config')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


if __name__ == '__main__':
    train(parse_arguments())
