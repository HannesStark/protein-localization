import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import ToTensor, LabelToInt
from torchvision.transforms import transforms
from torch.nn.utils.rnn import pad_sequence
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import ToTensor, LabelToInt


# print(dataset[0])

# train_val_split('data/fasta_files/test_as_per_deeploc.fasta', train_size=0.5)

# identifiers = []
# labels = []
# for record in SeqIO.parse('old_fasta_files/test.fasta', "fasta"):
#    identifiers.append(record.id)
#    labels.append(record.description.split(' ')[1].split('-')[0])
# df = pd.DataFrame(list(zip(identifiers, labels)), columns=['identifier', 'label'])
# print(df)

# create_annotations_csv('old_fasta_files/test.fasta', '.')
from models.conv_avg_pool import ConvAvgPool


def my_collate(batch):
    data = [item[0] for item in batch]
    target = torch.tensor([item[1] for item in batch])
    data = pad_sequence(data, batch_first=True)
    return (data, target)

model = ConvAvgPool()

dataset = EmbeddingsLocalizationDataset('data/embeddings/train.h5', 'data/embeddings/train_remapped.fasta',
                                        transform=transforms.Compose([LabelToInt(), ToTensor()]))


trainset = DataLoader(dataset=dataset,
                      batch_size=4,
                      shuffle=True,
                      collate_fn=my_collate, # use custom collate function here
                      pin_memory=True)

trainiter = iter(trainset)
embeddings, labels = trainiter.next()
print(embeddings.shape)
print(labels.shape)

