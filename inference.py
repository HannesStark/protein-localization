import glob
import os

from models import *  # required dont remove this
from torch.optim import *  # required dont remove this
import argparse
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import *
from models.loss_functions import cross_entropy_joint
from solvers.base_solver import BaseSolver
from utils.general import padded_permuted_collate


def inference(args):
    transform = transforms.Compose([LabelToInt(), ToTensor()])
    data_set = EmbeddingsLocalizationDataset(args.embeddings, args.remapping, transform=transform)
    if len(data_set[0][0].shape) == 2:  # if we have per residue embeddings they have an additional lenght dim
        collate_function = padded_permuted_collate
    else:  # if we have reduced sequence wise embeddings use the default collate function by passing None
        collate_function = None

    data_loader = DataLoader(data_set, batch_size=args.batch_size, collate_fn=collate_function)

    # Needs "from models import *" to work
    model = globals()[args.model_type](embeddings_dim=data_set[0][0].shape[-1], **args.model_parameters)

    # Needs "from torch.optim import *" to work
    solver = BaseSolver(model, args, globals()[args.optimizer], cross_entropy_joint)

    with torch.no_grad():
        loc_loss, sol_loss, results = solver.predict(data_loader)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/inference.yaml')
    p.add_argument('--checkpoint', type=str, default='runs/FFN__14-10_10-25-11',
                   help='path to directory that contains a checkpoint')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--loops', type=int, default=1, help='how often to loop over the given data')
    p.add_argument('--log_iterations', type=int, default=-1, help='log every log_iterations (-1 for no logging)')

    p.add_argument('--embeddings', type=str, default='data/embeddings/val.h5',
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
