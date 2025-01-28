import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms._functional_pil
import torchvision.transforms.functional
from utils import adapt_size, SIZEADAPTER
from models import (
    PatchDescriptionNetworkSmall,
    PatchDescriptionNetworkMedium,
    PDNSIZE,
    PDNTYPES,
    AutoEncoder,
    Pretraining
)
from Dataset import DirectoryDataset


class EfficientADClass(nn.Module):
    
    """
    EfficientAD is a neural network model designed for anomaly detection using 
    patch description networks and an autoencoder.
    Attributes:
        input_size (int): The size of the input channels.
        PDN_size (PDNSIZE): The size of the Patch Description Network (PDN).
    Methods:
        efficientAD_train():
            Sets the student PDN to evaluation mode and the teacher PDN and 
            autoencoder to training mode.
        forward(x: torch.Tensor, image_size: tuple[int, int] = (256, 256), 
                reducing_size: int = 64) -> torch.Tensor:
            Performs a forward pass through the autoencoder with the given 
            input tensor, image size, and reducing size.
    """
    def __init__(
        self, train_loader,
        val_loader,
        input_size: int = 3,
        PDN_size: PDNSIZE = PDNSIZE.S
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.PDN_size = PDN_size
        self.mean_teacher_channels: list[int] = []
        self.std_teacher_channels: list[int] = []
        self.image_net_data_loader = self.get_images_from_image_net_data_loader()  # noqa: E501
        self.train_loader = train_loader
        self.val_loader = val_loader
        if self.PDN_size == PDNSIZE.S: 
            self.pdn_student = PatchDescriptionNetworkSmall(
                self.input_size, PDNTYPES.S
            ).to('cuda')

            self.pdn_teacher = PatchDescriptionNetworkSmall(
                self.input_size, PDNTYPES.T
            ).to('cuda')
        else:

            self.pdn_student = PatchDescriptionNetworkMedium(
                self.input_size, PDNTYPES.S
            ).to('cuda')  # type: ignore

            self.pdn_teacher = PatchDescriptionNetworkMedium(
                self.input_size, PDNTYPES.T
            ).to('cuda')  # type: ignore

        self.auto_encoder = AutoEncoder()

        self.pdn_teacher.eval()
        self.pdn_student.train()
        self.auto_encoder.train()
        params = list(self.pdn_student.parameters()) + list(self.auto_encoder.parameters())  # noqa: E501
        self.optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)  # noqa: E501
        self.criterion = nn.MSELoss()

    def pretrain_teacher(self) -> None:
        """
        Pretrains the teacher PDN model using the given training data.
        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
        Returns:
            None
        """
        pretraining = Pretraining(self.pdn_teacher, self.image_net_data_loader)
        self.pdn_teacher=pretraining.pretrain()
        #DEL: pretraining object
        del pretraining
        self.pdn_teacher.eval()
        with torch.no_grad():
            self.calculate_teacher_channel_normalization_parameters()

    def calculate_teacher_channel_normalization_parameters(self) -> None:  # noqa: E501   
        """
        Calculate and store the mean and standard deviation for each channel of the teacher's output.
        This method iterates over the training dataset, passing each image through the teacher
        and collecting the output of a specific layer. It then computes the mean and standard deviation
        for each channel across all images and stores these values in the corresponding attributes.
        """

        X = []
        for _, img in self.train_loader:
            img = img.to('cuda')
            output_teacher = self.pdn_teacher(img)  # noqa: E501
            batch_mean= torch.mean(output_teacher,dim=[0,2,3])
            X.append(batch_mean.cpu().detach())  # noqa: E501
        self.mean_teacher_channels = torch.mean(torch.stack(X), dim=0)[None,:,None,None] 
        mean_diff_list = []
        for _, img in self.train_loader:
            img = img.to('cuda')
            teacher_output = self.pdn_teacher(img)
            distance = (teacher_output - self.mean_teacher_channels.to('cuda')) ** 2
            mean_diff = torch.mean(distance, dim=[0, 2, 3])
            mean_diff_list.append(mean_diff)
        channel_var = torch.mean(torch.stack(mean_diff_list), dim=0)
        channel_var = channel_var[None, :, None, None]
        self.std_teacher_channels = torch.sqrt(channel_var)


    def get_images_from_image_net_data_loader(self, num_images: int = 10) -> torch.utils.data.DataLoader:  # noqa: E501
        """
        Selects a specified number of images from the ImageNet dataset.
        Args:
            num_images (int): The number of images to select. Default is 1000.
        Returns:
            torch.utils.data.DataLoader: A DataLoader containing the selected images.
        """
        dataset = DirectoryDataset(
            img_dir='C:/Users/DeepHawk/Documents/Abbass_Tests/datasets/image_net/',
            ext='jpeg',
            with_subfolders=True,
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor()
            ]),
        )
        indices = np.random.choice(len(dataset), num_images, replace=False)
        sampler = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(sampler, batch_size=20, shuffle=True)
        return loader
    
    def random_augmentation(self, img: torch.Tensor) -> torch.Tensor:
        """
        Randomly augments the input image tensor.

        """
        rand_int = np.random.randint(0, 3)
        sampled_lambda = torch.rand(1).item() * 0.4 + 0.8
        
        if rand_int == 0:
            augmented_img = torchvision.transforms.functional.adjust_brightness(img, sampled_lambda)  # noqa: E501
        elif rand_int == 1:
            augmented_img = torchvision.transforms.functional.adjust_contrast(img, sampled_lambda)  # noqa: E501
        else:
            augmented_img = torchvision.transforms.functional.adjust_saturation(img, sampled_lambda)  # noqa: E501
        return augmented_img
    
    def efficientAD_train_step(self, img: torch.Tensor) -> torch.Tensor:
        """
        Perform a single training step for the EfficientAD model.
        Args:
            img (torch.Tensor): The input image tensor.
        Returns:
            None
        """
        self.optimizer.zero_grad()
        img_resized = adapt_size(img, output_size=512, by=SIZEADAPTER.INTERPOLATION)  # noqa: E501
        output_teacher = self.pdn_teacher(img_resized)
        output_teacher = (output_teacher - torch.tensor(self.mean_teacher_channels).to('cuda')) / torch.tensor(self.std_teacher_channels).to('cuda')  # noqa: E501
        output_student = self.pdn_student(img_resized)
        teacher_ae_output_dimension = output_student.shape[1] // 2
        y_students_teacher_part = output_student[:, :teacher_ae_output_dimension, :, :]  # noqa: E501. the student output dimension is equalt ot he output dimension of the teacher plus that of the AE.
        difference_student_teacher = torch.square(y_students_teacher_part - output_teacher)  # noqa: E501
        d_hard =np.quantile(difference_student_teacher.detach().cpu().numpy(), 0.999)
        l_hard = torch.mean(difference_student_teacher[difference_student_teacher >= d_hard])  # noqa: E501
        _, image_net_sample = next(iter(self.image_net_data_loader))
        l_student_teacher = l_hard + 1/(teacher_ae_output_dimension * output_student.shape[2] * output_student.shape[3]) * torch.sum(torch.square(self.pdn_student(image_net_sample.to('cuda'))))  # noqa: E501
        augmented_img = self.random_augmentation(img_resized)
        output_ae = self.auto_encoder(augmented_img)
        output_teacher_ae = self.pdn_teacher(augmented_img)
        output_teacher_ae = (output_teacher_ae - torch.tensor(self.mean_teacher_channels).to('cuda')) / torch.tensor(self.std_teacher_channels).to('cuda')  # noqa: E501
        output_student_ae = self.pdn_student(augmented_img)
        output_student_ae = output_student_ae[:, teacher_ae_output_dimension:, :, :]  # noqa: E501
        difference_teacher_ae = torch.square(output_teacher_ae - output_ae)
        difference_student_ae = torch.square(output_student_ae - output_ae)
        l_ae = torch.mean(difference_teacher_ae) 
        l_st_ae = torch.mean(difference_student_ae)
        l_total = l_student_teacher + l_ae + l_st_ae
        l_total.backward()
        self.optimizer.step()
        return l_total

    def efficientAD_eval(self) -> None:
        """
        Perform a single evaluation step for the EfficientAD model.
        Args:
            img (torch.Tensor): The input image tensor.
        Returns:
            None
        """
        self.pdn_student.eval()
        self.pdn_teacher.eval()
        self.auto_encoder.eval()
        x_st = []
        x_ae = []
        for label, img in self.val_loader:
            img=img.to('cuda')
            img_resized = adapt_size(img, output_size=512, by=SIZEADAPTER.INTERPOLATION) # noqa: E501
            output_teacher = self.pdn_teacher(img_resized.to('cuda'))
            output_teacher = (output_teacher - torch.tensor(self.mean_teacher_channels).to('cuda')) / torch.tensor(self.std_teacher_channels).to('cuda')  # noqa: E501
            output_student = self.pdn_student(img_resized)
            output_ae = self.auto_encoder(img)
            teacher_ae_output_dimension = output_student.shape[1] // 2
            y_students_teacher_part = output_student[:, :teacher_ae_output_dimension, :, :]  # noqa: E501
            difference_student_teacher = torch.square(y_students_teacher_part - output_teacher)  # noqa: E501
            output_student_ae = output_student[:, teacher_ae_output_dimension:, :, :]  # noqa: E501
            difference_student_ae = torch.square(output_student_ae - output_ae)
            anomaly_map_st = (1/teacher_ae_output_dimension) * torch.sum(difference_student_teacher, dim=1)  # noqa: E501
            anomaly_map_ae = (1/teacher_ae_output_dimension) * torch.sum(difference_student_ae, dim=1)  # noqa: E501
            resized_anomaly_map_st = adapt_size(anomaly_map_st[:,None,...], output_size=1024, by=SIZEADAPTER.INTERPOLATION)  # noqa: E501
            resized_anomaly_map_ae = adapt_size(anomaly_map_ae[:,None,...], output_size=1024, by=SIZEADAPTER.INTERPOLATION)  # noqa: E501
            x_st.append(resized_anomaly_map_st)
            x_ae.append(resized_anomaly_map_ae)

        self.q_a_st = np.quantile(torch.cat(x_st,dim=0).detach().cpu().numpy(), 0.9)
        self.q_b_st = np.quantile(torch.cat(x_st,dim=0).detach().cpu().numpy(), 0.995)
        self.q_a_ae = np.quantile(torch.cat(x_ae,dim=0).detach().cpu().numpy(), 0.9)
        self.q_b_ae = np.quantile(torch.cat(x_ae,dim=0).detach().cpu().numpy(), 0.995)

    # pylint: disable=C0103
    def efficientAD_train(self) -> None:

        """
        Trains the EfficientAD model by setting the appropriate modes for the student, teacher, 
        and autoencoder networks.
        This method performs the following steps:
        1. Sets the student network (pdn_student) to evaluation mode.
        2. Sets the teacher network (pdn_teacher) to training mode.
        3. Sets the autoencoder network (auto_encoder) to training mode.
        Returns:
            None
        """
        for epoch in range(150):
            loss_batch = 0.0
            for _, img in self.train_loader:
                img = img.to('cuda')
                loss = self.efficientAD_train_step(img)
                loss_batch += loss.item()
                print(f'epoch: {epoch}, loss: {loss.item()}')
            if epoch > 1000 == 0:
                # decay the learning rate to 10-5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-5
        torch.save(self.pdn_student.state_dict(), 'PDN_Student_Weights.pth')
        torch.save(self.pdn_teacher.state_dict(), 'PDN_Teacher_Weights.pth')
        torch.save(self.auto_encoder.state_dict(), 'AutoEncoder_Weights.pth')
        torch.save(self.state_dict(), 'EfficientAD_Model.pth')
        self.efficientAD_eval()
        print('Finished Training the EfficientAD model!')
        # torch.save(self.pdn_student.state_dict(), 'PDN_Student_Weights.pth')l
        # torch.save(self.pdn_teacher.state_dict(), 'PDN_Teacher_Weights.pth')
        # torch.save(self.auto_encoder.state_dict(), 'AutoEncoder_Weights.pth')
        # torch.save(self.state_dict(), 'EfficientAD_Model.pth')
    
    def inference(self, img: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the input image tensor.
        Args:
            img (torch.Tensor): The input image tensor.
        Returns:
            torch.Tensor: The output tensor after passing through the EfficientAD model.
        """
        img=torch.Tensor(img)/255.0
        img=img[None,...].permute(0,3,1,2)
        img=img.to('cuda')
        img_resized = adapt_size(img, output_size=512, by=SIZEADAPTER.INTERPOLATION)  # noqa: E501
        output_teacher = self.pdn_teacher(img_resized.to('cuda'))
        output_teacher = (output_teacher.to('cuda') - torch.tensor(self.mean_teacher_channels.to('cuda'))) / torch.tensor(self.std_teacher_channels.to('cuda'))  # noqa: E501
        output_student = self.pdn_student(img_resized)
        output_ae = self.auto_encoder(img)
        teacher_ae_output_dimension = output_student.shape[1] // 2
        y_students_teacher_part = output_student[:, :teacher_ae_output_dimension, :, :]  # noqa: E501
        difference_student_teacher = torch.square(y_students_teacher_part - output_teacher)  # noqa: E501
        output_student_ae = output_student[:, teacher_ae_output_dimension:, :, :]  # noqa: E501
        difference_student_ae = torch.square(output_student_ae - output_ae)
        anomaly_map_st = (1/teacher_ae_output_dimension) * torch.sum(difference_student_teacher, dim=1)  # noqa: E501
        anomaly_map_ae = (1/teacher_ae_output_dimension) * torch.sum(difference_student_ae, dim=1)  # noqa: E501
        resized_anomaly_map_st = adapt_size(anomaly_map_st[None,...], output_size=1024, by=SIZEADAPTER.INTERPOLATION)  # noqa: E501
        resized_anomaly_map_ae = adapt_size(anomaly_map_ae[None,...], output_size=1024, by=SIZEADAPTER.INTERPOLATION)  # noqa: E501
        resized_anomaly_map_st = 0.1 * (resized_anomaly_map_st - self.q_a_st) / (self.q_b_st - self.q_a_st)  # noqa: E501
        resized_anomaly_map_ae = 0.1 * (resized_anomaly_map_ae - self.q_a_ae) / (self.q_b_ae - self.q_a_ae)  # noqa: E501
        anomaly_map = 0.5 * resized_anomaly_map_st + 0.5 * resized_anomaly_map_ae  # noqa: E501
        return anomaly_map
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inference(x)
