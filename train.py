import argparse
import os

from torchvision.transforms import transforms

from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import LabelOneHot, ToTensor


def train(args):
    transform = transforms.Compose([LabelOneHot(), ToTensor()])
    train_set = EmbeddingsLocalizationDataset(args.train_embeddings, args.train_remapping, transform)
    val_set = EmbeddingsLocalizationDataset(args.val_embeddings, args.val_remapping, transform)
    print(train_set[0])

def parse_arguments():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    train(args)
