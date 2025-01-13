import math
from enum import Enum
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from abc import ABC, abstractmethod
from torchvision.models.feature_extraction import create_feature_extractor
import copy

import torchvision.transforms._functional_pil
from utils import adapt_size, SIZEADAPTER


class PDNTYPES(str, Enum):
    """
    PDNTYPES is an enumeration for different types of roles.
    Attributes:
        S (str): Represents a student.
        T (str): Represents a teacher.
    """
    S = "Student",
    T = "Teacher",


class PDNSIZE(str, Enum):
    """
    Enum representing different sizes for PDN (Power Delivery Network).
    Attributes:
        S (str): Represents a small size PDN.
        M (str): Represents a medium size PDN.
    """
    S = "Small",
    M = "Medium",


class PatchDescriptionNetwork(nn.Module, ABC):

    """
    PatchDescriptionNetwork is an abstract base class for a neural network model that describes patches.
    Attributes:
        input_size (int): The size of the input. Default is 3.
        output_size (int): The size of the output. It is set to 384 if PDN_type is PDNTYPES.T, otherwise it is set to 768.
    Args:
        input_size (int): The size of the input. Default is 3.
        PDN_type (PDNTYPES): The type of the Patch Description Network. Default is PDNTYPES.S.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Abstract method to define the forward pass of the network. Must be implemented by subclasses.
    """

    def __init__(self, input_size: int = 3, PDN_type: PDNTYPES = PDNTYPES.S) -> None:
        super().__init__()
        self.input_size = input_size
        if PDN_type == PDNTYPES.T:
            self.output_size = 384
        else:
            self.output_size = 768

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        """
        forward method
        """
        raise NotImplementedError


class PatchDescriptionNetworkSmall(PatchDescriptionNetwork):

    """
    A small variant of the Patch Description Network (PDN).
    This class defines a convolutional neural network (CNN) model for patch description tasks.
    It inherits from the PatchDescriptionNetwork base class and implements a smaller version
    of the network architecture.
    Attributes:
        model (nn.Sequential): The sequential container of the network layers.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the network.
        Initializes the PatchDescriptionNetworkSmall instance.
        Args:
            input_size (int): The number of input channels. Default is 3.
            PDN_type (PDNTYPES): The type of PDN. Default is PDNTYPES.S.
        # Model definition here
        Defines the forward pass of the network.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after passing through the network.
    """
    def __init__(self, input_size: int = 3, PDN_type: PDNTYPES = PDNTYPES.S) -> None:
        super().__init__(input_size, PDN_type)

        self.model = nn.Sequential(
            nn.Conv2d(self.input_size, 128, 4, 1, 3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 1),
            nn.Conv2d(128, 256, 4, 1, 3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, self.output_size, 4, 1, 0)
        )

    def forward(self, x) -> torch.Tensor:
        """
        forward method
        """
        return self.model(x)


class PatchDescriptionNetworkMedium(PatchDescriptionNetwork):

    """
    PatchDescriptionNetworkMedium is a neural network model designed for medium-sized patch description tasks.
    Attributes:
        input_size (int): The number of input channels. Default is 3.
        PDN_type (PDNTYPES): The type of Patch Description Network. Default is PDNTYPES.S.
        model (nn.Sequential): The sequential container of the neural network layers.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the network.
    """
    def __init__(self, input_size: int = 3, PDN_type: PDNTYPES = PDNTYPES.S) -> None:
        super().__init__(input_size, PDN_type)

        self.model = nn.Sequential(
            nn.Conv2d(self.input_size, 256, 4, 1, 3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 1),
            nn.Conv2d(256, 512, 4, 1, 3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 1),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, self.output_size, 4, 1, 0),
            nn.ReLU(),
            nn.Conv2d(self.output_size, self.output_size, 1, 1, 0)
        )

    def forward(self, x) -> torch.Tensor:
        """
        forward method
        """
        return self.model(x)


class Encoder(nn.Module):

    """
    Encoder class for a convolutional neural network.
    This class defines an encoder model using a series of convolutional layers with ReLU activations.
    Attributes:
        input_size (int): The number of input channels. Default is 3.
        output_size (int): The number of output channels. Default is 64.
        model (nn.Sequential): The sequential container of convolutional layers and activations.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the encoder model.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after passing through the encoder model.
    """
    def __init__(self, input_size: int = 3, output_size: int = 64) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Conv2d(self.input_size, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, self.output_size, 8, 1, 0),
        )

    def forward(self, x) -> torch.Tensor:
        """
        forward method
        """
        return self.model(x)
  
   
class Decoder(nn.Module):

    """
    A PyTorch neural network module for decoding an input tensor through a series of convolutional layers
    and upsampling operations.
    Attributes:
        input_size (int): The number of input channels.
        output_size (int): The number of output channels.
    Methods:
        forward(x: torch.Tensor, image_size: tuple[int, int] = (256, 256), reducing_size: int = 64) -> torch.Tensor:
            Forward pass of the decoder. Takes an input tensor `x` and returns the decoded output tensor.
            Args:
                x (torch.Tensor): The input tensor.
                image_size (tuple[int, int], optional): The size of the input image. Defaults to (256, 256).
                reducing_size (int, optional): The factor by which the image size is reduced at each step. Defaults to 64.
            Returns:
                torch.Tensor: The decoded output tensor.
    """

    def __init__(self, input_size: int = 64, output_size: int = 384) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.dec_conv1 = nn.Conv2d(self.input_size, 64, 4, 1, 2)
        self.dec_conv2 = nn.Conv2d(64, 64, 4, 1, 2)
        self.dec_conv3 = nn.Conv2d(64, 64, 4, 1, 2)
        self.dec_conv4 = nn.Conv2d(64, 64, 4, 1, 2)
        self.dec_conv5 = nn.Conv2d(64, 64, 4, 1, 2)
        self.dec_conv6 = nn.Conv2d(64, 64, 4, 1, 2)
        self.dec_conv7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.dec_conv8 = nn.Conv2d(64, self.output_size, 3, 1, 1)

    def forward(
            self, x: torch.Tensor,
            image_size: tuple[int, int] = (256, 256),
            reducing_size: int = 64
    ) -> torch.Tensor:
        """
         Forward pass for the model.
            Args:
                x (torch.Tensor): Input tensor.
                image_size (tuple[int, int], optional): The size of the input image. Defaults to (256, 256).
                reducing_size (int, optional): The factor by which the image size is reduced at each step. Defaults to 64.
            Returns:
                torch.Tensor: Output tensor after applying the forward pass transformations.
        """

        last_upsample = (
            math.ceil(image_size[0] / 4),
            math.ceil(image_size[1] / 4)
        )

        x = F.interpolate(
            x,
            size=(
                image_size[0] // reducing_size - 1,
                image_size[1] // reducing_size - 1
            ),
            mode='bilinear'
        )

        x = F.dropout2d(F.relu(self.dec_conv1(x)), 0.2)

        x = F.interpolate(
            x,
            size=(
                image_size[0] // (reducing_size // 2) - 1,
                image_size[1] // (reducing_size // 2) - 1
            ),
            mode='bilinear'
        )

        x = F.dropout2d(F.relu(self.dec_conv2(x)), 0.2)

        x = F.interpolate(
            x,
            size=(
                image_size[0] // (reducing_size // 4) - 1,
                image_size[1] // (reducing_size // 4) - 1
            ),
            mode='bilinear'
        )
        x = F.dropout2d(F.relu(self.dec_conv3(x)), 0.2)

        x = F.interpolate(
            x,
            size=(
                image_size[0] // (reducing_size // 8) - 1,
                image_size[1] // (reducing_size // 8) - 1
            ),
            mode='bilinear'
        )

        x = F.dropout2d(F.relu(self.dec_conv4(x)), 0.2)

        x = F.interpolate(
            x,
            size=(
                image_size[0] // (reducing_size // 16) - 1,
                image_size[1] // (reducing_size // 16) - 1
            ),
            mode='bilinear'

        )

        x = F.dropout2d(F.relu(self.dec_conv5(x)), 0.2)

        x = F.interpolate(
            x,
            size=(
                image_size[0] // (reducing_size // 32) - 1,
                image_size[1] // (reducing_size // 32) - 1
                ),
            mode='bilinear'
        )

        x = F.dropout2d(F.relu(self.dec_conv6(x)), 0.2)

        x = F.interpolate(x, size=last_upsample, mode='bilinear')

        x = F.relu(self.dec_conv7(x))

        x = F.relu(self.dec_conv8(x))

        return x


class AutoEncoder(nn.Module):
    """
    AutoEncoder model consisting of an encoder and a decoder.
    Attributes:
        input_size (int): The size of the input features.
        encoder_output_size (int): The size of the output features from the encoder.
        decoder_output_size (int): The size of the output features from the decoder.
    Methods:
        forward(x: torch.Tensor, image_size: tuple[int, int] = (256, 256), reducing_size: int = 64) -> torch.Tensor:
            Performs a forward pass through the autoencoder.
            Args:
                x (torch.Tensor): The input tensor.
                image_size (tuple[int, int], optional): The size of the image. Defaults to (256, 256).
                reducing_size (int, optional): The size to reduce the image to. Defaults to 64.
            Returns:
                torch.Tensor: The reconstructed tensor after passing through the autoencoder.
    """

    def __init__(self, input_size: int = 3, encoder_output_size: int = 64,
                 decoder_output_size: int = 384) -> None:
        
        super().__init__()
        self.input_size = input_size
        self.encoder_output_size = encoder_output_size
        self.decoder_output_size = decoder_output_size

        self.encoder = Encoder(input_size, encoder_output_size)
        self.decoder = Decoder(encoder_output_size, decoder_output_size)

    def forward(
            self, x: torch.Tensor,
            image_size: tuple[int, int] = (256, 256),
            reducing_size: int = 64
    ) -> torch.Tensor:
        """
        forward method
        """
        x = self.encoder(x)
        x = self.decoder(x, image_size, reducing_size)
        return x


class Pretraining:
    
    """
    A class used to handle the pretraining of a model using a pretrained Wide ResNet101_2 as a feature extractor.
    Attributes
    ----------
    pretrained_model : torch.nn.Module
        The pretrained Wide ResNet101_2 model.
    feature_extractor : torch.nn.Module
        The feature extractor created from the pretrained model.
    model_to_train : PatchDescriptionNetwork
        The model that will be trained.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training data.
    mean_feature_extractor_channels : list[int]
        List to store the mean of feature extractor channels.
    std_feature_extractor_channels : list[int]
        List to store the standard deviation of feature extractor channels.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    criterion : torch.nn.Module
        The loss function used for training the model.
    Methods
    -------
    calculate_feature_extractor_channel_normalization_parameters() -> None
        Calculates and stores the mean and standard deviation for each channel of the feature extractor.
    get_feature_extractor(pretrained_model: torch.nn.Module) -> torch.nn.Module
        Creates and returns a feature extractor from the pretrained model.
    train_step(img: torch.Tensor) -> torch.Tensor
        Performs a single training step and returns the loss.
    pretrain(epochs: int = 100) -> None
        Pretrains the model for the specified number of epochs.
    """
    def __init__(
            self,
            model_to_train: PatchDescriptionNetwork,
            train_loader: torch.utils.data.DataLoader
    ) -> None:

        # pylint: disable=E1121
        self.pretrained_model = torchvision.models.wide_resnet101_2('Wide_ResNet101_2_Weights.IMAGENET1K_V2').to('cuda')  # noqa: E501
        self. feature_extractor = self.get_feature_extractor(self.pretrained_model)  # noqa: E501
        self.model_to_train = model_to_train
        self.model_to_train.train()
        # self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.train_loader = train_loader
        self.mean_feature_extractor_channels: list[int] = []
        self.std_feature_extractor_channels: list[int] = []
        self.calculate_feature_extractor_channel_normalization_parameters()
        self.optimizer = torch.optim.Adam(self.model_to_train.parameters(), lr=0.001)  # noqa: E501
        self.criterion = nn.MSELoss()

    def calculate_feature_extractor_channel_normalization_parameters(self) -> None:  # noqa: E501   
        """
        Calculate and store the mean and standard deviation for each channel of the feature extractor's output.
        This method iterates over the training dataset, passing each image through the feature extractor
        and collecting the output of a specific layer. It then computes the mean and standard deviation
        for each channel across all images and stores these values in the corresponding attributes.
        """
        for c in range(self.model_to_train.output_size):
            X = []
            for img in self.train_loader:
                img = img.to('cuda')
                output_feature_extractor = adapt_size(self.feature_extractor(img).get('layer2'), outpu_size=self.model_to_train.output_size)  # noqa: E501
                X.append(output_feature_extractor[:, c, :].flatten().cpu().detach().numpy())  # noqa: E501
            x = np.array(X).flatten()
            self.mean_feature_extractor_channels.append(np.mean(x))
            self.std_feature_extractor_channels.append(np.std(x))

    def get_feature_extractor(
            self,
            pretrained_model: torch.nn.Module,
    ) -> torch.nn.Module:
        """
        Extracts features from a pretrained model using specified return nodes.
            Args:
                pretrained_model (torch.nn.Module): The pretrained model from which to extract features.
            Returns:
                torch.nn.Module: A feature extractor model that outputs the specified layers.
        """

        return_nodes = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        }

        fe = create_feature_extractor(pretrained_model, return_nodes=return_nodes)  # noqa: E501
        return fe

    def train_step(self, img) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Computed loss for the training step.
        """
        self.optimizer.zero_grad()
        output_feature_extractor = adapt_size(self.feature_extractor(img).get('layer2'), outpu_size=self.model_to_train.output_size)  # noqa: E501 adapt the size of the feature extraxctor output to the model_to_train output size on the channel dim.
        output_feature_extractor = (output_feature_extractor - torch.tensor(self.mean_feature_extractor_channels).to('cuda')) / torch.tensor(self.std_feature_extractor_channels).to('cuda')  # noqa: E501
        img = adapt_size(img, outpu_size=512, by=SIZEADAPTER.INTERPOLATION)  # noqa: E501 adapt the size of the input image to the model_to_train input size using interpolation on the width and height dims.
        outputs_model_to_train = self.model_to_train(img)
        loss = self.criterion(outputs_model_to_train, output_feature_extractor)
        loss.backward()
        self.optimizer.step()
        return loss

    def pretrain(self, epochs: int = 100) -> None:
        """
        Pretrains the model for a specified number of epochs.
        Args:
            epochs (int): The number of epochs to pretrain the model. Default is 100.
        Returns:
            None
        """
        for epoch in range(epochs):
            loss_batch = 0.0
            for img in self.train_loader:
                img = img.to('cuda')
                loss = self.train_step(img)
                loss_batch += loss.item()
                print(f'epoch: {epoch}, loss: {loss.item()}')
        print('Finished Training the PDN Teacher model!')
        torch.save(self.model_to_train.state_dict(), 'PDN_Teacher_Weights.pth')
