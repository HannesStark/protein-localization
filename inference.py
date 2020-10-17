import glob
import os

from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

from models import *  # required dont remove this
from torch.optim import *  # required dont remove this
from adabelief_pytorch import AdaBelief
import argparse
import yaml
from torch.utils.data import DataLoader, RandomSampler
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

    sampler = RandomSampler(data_set, replacement=True)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_function)

    # Needs "from models import *" to work
    model = globals()[args.model_type](embeddings_dim=data_set[0][0].shape[-1], **args.model_parameters)
    model.eval()

    # Needs "from torch.optim import *" to work
    solver = BaseSolver(model, args, globals()[args.optimizer], cross_entropy_joint)

    mccs = []
    accuracies = []
    for i in tqdm(range(args.n_draws)):
        with torch.no_grad():
            loc_loss, sol_loss, results = solver.predict(data_loader)
        accuracies.append(100 * np.equal(results[:, 0], results[:, 1]).sum() / len(results))
        mccs.append(matthews_corrcoef(results[:, 1], results[:, 0]))

    accuracy = np.mean(accuracies)
    accuracy_stderr = np.std(accuracies)
    mcc = np.mean(mccs)
    mcc_stderr = np.std(mccs)
    results_string = 'Accuracy: {:.2f}% \n' \
                     'Accuracy stderr: {:.2f}\n' \
                     'MCC: {:.4f}\n' \
                     'MCC stderr: {:.4f}\n'.format(accuracy, accuracy_stderr, mcc, mcc_stderr)
    with open(os.path.join(args.checkpoint, 'evaluation.txt'), 'w') as file:
        file.write(results_string)
    print(results_string)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/inference.yaml')
    p.add_argument('--checkpoint', type=str, default='runs/.ex9/ConvMaxAvgPool_9_lr5-e6_14-10_11-18-55',
                   help='path to directory that contains a checkpoint')
    p.add_argument('--batch_size', type=int, default=32, help='samples that will be processed in parallel')
    p.add_argument('--n_draws', type=int, default=100,
                   help='how often to bootstrap from the dataset for variance estimation')
    p.add_argument('--log_iterations', type=int, default=10, help='log every log_iterations (-1 for no logging)')

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
