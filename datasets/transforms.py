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
        label = torch.from_numpy(np.array(label)).float()

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