from typing import Tuple, Union

import torch
import numpy as np

from utils.general import LOCALIZATION


class ToTensor():
    """
    Turn np.array into torch.Tensor.
    """

    def __init__(self):
        pass

    def __call__(self, sample: Tuple[np.ndarray, Union[int, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding, label = sample
        embedding = torch.from_numpy(embedding).float()
        label = torch.from_numpy(np.array(label)).long()

        return embedding, label


class AvgMaxPool():
    """
    Pools embeddings along dim and concatenates max and avg pool
    """

    def __init__(self, dim: int = -2):
        """

        Args:
            dim: dimension along which to pool
        """
        self.dim = dim

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            sample: ([sequence_length, embedding_size],[label_encoding_size]) tuple of embedding and label

        Returns:
            embedding: [2*embedding_size] the embedding tensor avg pooled and mean pooled along dim and concatenated
            label: the original label
        """
        embedding, label = sample
        avg_pool = torch.mean(embedding, dim=self.dim)
        max_pool, _ = torch.max(embedding, dim=self.dim)
        embedding = torch.cat([avg_pool, max_pool], dim=-1)
        return embedding, label


class LabelToInt():
    """
    Turn string label of localization into an integer
    """

    def __init__(self):
        pass

    def __call__(self, sample: Tuple[np.ndarray, str]) -> Tuple[np.ndarray, int]:
        embedding, label = sample
        label = LOCALIZATION.index(label)  # get label as integer

        return embedding, label


class LabelOneHot():
    """
    Turn string label of localization into a one hot np array
    """

    def __init__(self):
        pass

    def __call__(self, sample: Tuple[np.ndarray, str]) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            sample: tuple of embedding and label

        Returns:
            embedding: the original embedding
            label: [10] array with one hot encoding of label
        """
        embedding, label = sample
        label = LOCALIZATION.index(label)  # get label as integer
        one_hot_label = np.zeros(len(LOCALIZATION))
        one_hot_label[label] = 1
        return embedding, one_hot_label
