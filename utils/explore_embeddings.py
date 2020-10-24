import argparse
import os

import matplotlib.pyplot as plt
import torch
import yaml
from models import *  # This is necessary, do not remove it
from torchvision.transforms import transforms
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import LabelToInt, ToTensor
from utils.general import normalize

checkpoint = 'runs/.ex15/ConvMaxAvgPoolConv_9_NoBatchnorm_20-10_08-58-05'
embeddings = 'data/embeddings/val.h5'
remapping = 'data/embeddings/val_remapped.fasta'
batch_size = 8


def explore_embeddings(args):
    transform = transforms.Compose([LabelToInt(), ToTensor()])
    dataset = EmbeddingsLocalizationDataset(embeddings, remapping, unknown_solubility=False, transform=transform)

    # Needs "from models import *" to work
    model = globals()[args.model_type](embeddings_dim=dataset[0][0].shape[-1], **args.model_parameters)

    embedding, localization, solubility, metadata = dataset[100]

    # register defined hooks and run trough some example in the dataset
    model.conv1.register_forward_hook(visualize_activation_hook)
    model(embedding.T.unsqueeze(0))


    plt.plot(embedding[355])
    plt.show()

def visualize_activation_hook(self, input, output):
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams["image.cmap"] = 'viridis'
    inp = normalize(input[0].squeeze())
    out = normalize(output.data.squeeze())
    out_avg = torch.mean(out, dim=0)
    out_max = torch.max(out, dim=0)[0]
    plt.plot(out_avg)
    plt.show()
    plt.plot(out_max)
    plt.show()
    plt.imshow(out)
    plt.show()
    plt.imshow(inp)
    plt.show()


def parse_arguments():
    p = argparse.ArgumentParser()
    args = p.parse_args()
    arg_dict = args.__dict__
    data = yaml.load(open(os.path.join(checkpoint, 'train_arguments.yaml'), 'r'), Loader=yaml.FullLoader)
    for key, value in data.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value
    return args


if __name__ == '__main__':
    os.chdir('./..')
    explore_embeddings(parse_arguments())
