import argparse
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import *
from models.conv_avg_pool import ConvAvgPool
from models.ffn import FFN
from solvers.base_solver import BaseSolver
from utils.general import padded_permuted_collate


def train(args):
    transform = transforms.Compose([LabelToInt(), ToTensor()])
    train_set = EmbeddingsLocalizationDataset(args.train_embeddings, args.train_remapping, args.max_length, transform)
    val_set = EmbeddingsLocalizationDataset(args.val_embeddings, args.val_remapping, transform=transform)
    if args.model_type == 'ffn':
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        model = FFN(2*train_set[0][0].shape[0], args.hidden_dim, 10, args.num_hidden_layers, args.dropout)
    elif args.model_type == 'var-length':
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=padded_permuted_collate)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=padded_permuted_collate)
        model = ConvAvgPool()
        print(sum(p.numel() for p in model.parameters() if p.requires_grad)
              )
    else:
        raise ValueError('given model_type does not exist')
    solver = BaseSolver(model, args, torch.optim.Adam, torch.nn.CrossEntropyLoss())
    solver.train(train_loader, val_loader)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/variable_length.yaml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--num_epochs', type=int, default=50, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--lrate', type=float, default=1.0e-4, help='learning rate for training')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')

    p.add_argument('--model_type', type=str, help='type of model [ffn, var-length]')
    p.add_argument('--hidden_dim', type=int, default=32, help='neurons in hidden layers of feed forward network')
    p.add_argument('--num_hidden_layers', type=int, default=0, help='hidden layers in feed forward network')
    p.add_argument('--dropout', type=float, default=0.25, help='dropout in feed forward network')
    p.add_argument('--max_length', type=int, default=6000, help='maximum lenght of sequences that will be used for '
                                                                'training when using embedddings of variable length')

    p.add_argument('--train_embeddings', type=str, default='data/embeddings/train.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--train_remapping', type=str, default='data/embeddings/train_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--val_embeddings', type=str, default='data/embeddings/val.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--val_remapping', type=str, default='data/embeddings/val_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--test_embeddings', type=str, default='data/embeddings/test.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--test_remapping', type=str, default='data/embeddings/test_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    args = p.parse_args()
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
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
