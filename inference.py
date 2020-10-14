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
    data_set = EmbeddingsLocalizationDataset(args.val_embeddings, args.val_remapping, transform=transform)
    if len(data_set[0][0].shape) == 2:  # if we have per residue embeddings they have an additional lenght dim
        collate_function = padded_permuted_collate
    else:  # if we have reduced sequence wise embeddings use the default collate function by passing None
        collate_function = None

    data_loader = DataLoader(data_set, batch_size=args.batch_size, collate_fn=collate_function)

    # Needs "from models import *" to work
    model = globals()[args.model_type](embeddings_dim=data_set[0][0].shape[-1], **args.model_parameters)

    solver = BaseSolver(model, args, loss_func=cross_entropy_joint)
    loc_loss, sol_loss, results = solver.predict(data_loader)




def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'),
                   default='configs/conv_max_avg_pool_9_sol007_momentum.yaml')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint')

    p.add_argument('--data_embeddings', type=str, default='data/embeddings/train.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--data_remapping', type=str, default='data/embeddings/train_remapped.fasta',
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
    inference(parse_arguments())
