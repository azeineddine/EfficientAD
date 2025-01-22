from hashlib import sha1
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import numpy as np
import cv2


class SIZEADAPTER(Enum):

    """
    Enum class for different size adaptation methods in the EfficientAD model.
    Attributes:
        ADAPTIVE_MAX_POOLING (str): Represents the adaptive max pooling method.
        INTERPOLATION (str): Represents the interpolation method.
    """
    ADAPTIVE_MAX_POOLING = "AdaptiveMaxPooling",
    INTERPOLATION = "Interpolation",


def adapt_size(x: torch.Tensor, output_size: int = 384, by: SIZEADAPTER = SIZEADAPTER.ADAPTIVE_MAX_POOLING) -> torch.Tensor:  # noqa: E501
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
            N, _, W, H = x.shape
            x = x.permute(0, 2, 3, 1)  # Change to N, W, H, C
            x = x.reshape(-1, x.shape[-1])  # Flatten to (N*W*H, C)
            x = nn.AdaptiveMaxPool1d(output_size)(x.unsqueeze(0)).squeeze(0)  # Apply AdaptiveMaxPool1d
            x = x.reshape(-1, N*W*H, output_size)  # Reshape back to (N*W*H, output_size)
            x = x.reshape(-1, W, H, output_size)  # Reshape to (N, W, H, output_size)
            x = x.permute(0, 3, 1, 2)  # Change back to N, output_size, W, H
            return x
        else:
            output = F.interpolate(x, size=(output_size,output_size), mode='bilinear')
            return output


def correct_dead_pixel(img):

    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0

    img_peaks = np.zeros(img.shape)

    img_erode = cv2.erode(img, kernel)
    diff = img_erode - img
    diff[img_erode < img] = 0
    img_peaks[diff > 4] = 255

    img_dilate = cv2.dilate(img, kernel)

    diff = img - img_dilate
    diff[img_dilate > img] = 0
    img_peaks[diff > 4] = 255

    [y, x] = np.nonzero(img_peaks)

    for px, py in zip(x, y):
        patch = img[
            max(0, py - 1): min(img.shape[0], py + 2),
            max(0, px - 1): min(img.shape[1], px + 2),
        ]
        img[py, px] = np.median(patch).astype(np.uint8)

    return img
