import argparse
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from models import *
from datasets.transforms import *
from models.loss_functions import cross_entropy_joint
from solvers.base_solver import BaseSolver
from utils.general import padded_permuted_collate


def train(args):
    transform = transforms.Compose([LabelToInt(), ToTensor()])
    train_set = EmbeddingsLocalizationDataset(args.train_embeddings, args.train_remapping, args.max_length, transform)
    val_set = EmbeddingsLocalizationDataset(args.val_embeddings, args.val_remapping, transform=transform)

    if len(train_set[0][0].shape) == 2: # if we have per residue embeddings they have an additional lenght dim
        collate_function = padded_permuted_collate
    else: # if we have reduced sequence wise embeddings use the default collate function by passing None
        collate_function = None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,collate_fn=collate_function)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_function)

    model = globals()[args.model_type](embeddings_dim=train_set[0][0].shape[-1], **args.model_parameters)
    solver = BaseSolver(model, args, torch.optim.Adam, cross_entropy_joint, checkpoint_dir=args.checkpoint)
    solver.train(train_loader, val_loader)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/self_attention.yaml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--num_epochs', type=int, default=50, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--lrate', type=float, default=1.0e-4, help='learning rate for training')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint')

    p.add_argument('--model_type', type=str, default='FFN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--solubility_loss', type=float, default=0,
                   help='how much the loss of the solubility will be weighted')
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
