from typing import Tuple, Union

import torch
import numpy as np

from utils.general import LOCALIZATION, SOLUBILITY


class ToTensor():
    """
    Turn np.array into torch.Tensor.
    """

    def __init__(self):
        pass

    def __call__(self, sample: Tuple[np.ndarray, int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, localization, solubility = sample
        embedding = torch.from_numpy(embedding).float()
        localization = torch.tensor(localization).long()
        solubility = torch.tensor(solubility).float()
        return embedding, localization, solubility


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

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            sample: ([sequence_length, embedding_size],[localization_encoding_size], [1]) tuple of embedding and localization

        Returns:
            embedding: [2*embedding_size] the embedding tensor avg pooled and mean pooled along dim and concatenated
            localization: the original localization
            solubility: the original solubility
        """
        embedding, localization, solubility = sample
        avg_pool = torch.mean(embedding, dim=self.dim)
        max_pool, _ = torch.max(embedding, dim=self.dim)
        embedding = torch.cat([avg_pool, max_pool], dim=-1)
        return embedding, localization, solubility


class LabelToInt():
    """
    Turn string localization of localization into an integer
    """

    def __init__(self):
        pass

    def __call__(self, sample: Tuple[np.ndarray, str, str]) -> Tuple[np.ndarray, int, int]:
        embedding, localization, solubility = sample
        localization = LOCALIZATION.index(localization)  # get localization as integer
        # solubility = SOLUBILITY.index(solubility)  # get solubility as integer
        solubility = 1
        return embedding, localization, solubility


class LabelOneHot():
    """
    Turn string localization of localization into a one hot np array
    """

    def __init__(self):
        pass

    def __call__(self, sample: Tuple[np.ndarray, str, str]) -> Tuple[np.ndarray, np.ndarray, int]:
        """

        Args:
            sample: tuple of embedding and localization

        Returns:
            embedding: the original embedding
            localization: [10] array with one hot encoding of localization
        """
        embedding, localization, solubility = sample
        localization = LOCALIZATION.index(localization)  # get localization as integer
        one_hot_localization = np.zeros(len(LOCALIZATION))
        one_hot_localization[localization] = 1
        solubility = SOLUBILITY.index(solubility)
        return embedding, one_hot_localization, solubility
