import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import *
from models.simple_ffn import SimpleFFN
from solvers.base_solver import BaseSolver


def train(args):
    transform = transforms.Compose([LabelToInt(), ToTensor()])
    train_set = EmbeddingsLocalizationDataset(args.train_embeddings, args.train_remapping, transform)
    val_set = EmbeddingsLocalizationDataset(args.val_embeddings, args.val_remapping, transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    model = SimpleFFN()

    solver = BaseSolver(model, args, torch.optim.Adam, torch.nn.CrossEntropyLoss())
    solver.train(train_loader, val_loader)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/simpleFFN.yaml')
    parser.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of times to iterate through all samples')
    parser.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    parser.add_argument('--lrate', type=float, default=1.0e-4, help='learning rate for training')
    parser.add_argument('--log_iterations', type=int, default=5, help='log every log_iterations iterations')

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
