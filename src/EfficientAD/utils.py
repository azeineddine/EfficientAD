import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class SIZEADAPTER(Enum):

    """
    Enum class for different size adaptation methods in the EfficientAD model.
    Attributes:
        ADAPTIVE_MAX_POOLING (str): Represents the adaptive max pooling method.
        INTERPOLATION (str): Represents the interpolation method.
    """
    ADAPTIVE_MAX_POOLING = "AdaptiveMaxPooling",
    INTERPOLATION = "Interpolation",


def adapt_size(x: torch.Tensor, outpu_size: int = 384, by: SIZEADAPTER = SIZEADAPTER.ADAPTIVE_MAX_POOLING) -> torch.Tensor:  # noqa: E501
        """
        Adjusts the size of the input tensor `x` to the specified `output_size` using the specified method.

        Args:
            x (torch.Tensor): The input tensor to be resized.
            outpu_size (int, optional): The desired output size. Defaults to 384.
            by (SIZEADAPTER, optional): The method to use for resizing. Defaults to SIZEADAPTER.ADAPtIVE_MAX_POOLING.

        Returns:
            torch.Tensor: The resized tensor.
        """
        if by == SIZEADAPTER.ADAPTIVE_MAX_POOLING:
            return nn.AdaptiveMaxPool1d(outpu_size)(x, output_size=outpu_size)  # noqa: E501
        else:
            return F.interpolate(x, size=outpu_size, mode='bilinear')
    