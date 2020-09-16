from torchvision.transforms import transforms

from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import ToTensor, LabelToInt, LabelOneHot

dataset = EmbeddingsLocalizationDataset('embeddings/test_reduced.h5', 'embeddings/test_remapped.fasta',
                                        transform=transforms.Compose([LabelOneHot(), ToTensor()]))

print(dataset[0])
