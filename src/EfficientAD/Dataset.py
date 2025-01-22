
from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import os
import torchvision.transforms as v2
from utils import correct_dead_pixel


class DirectoryDataset(Dataset):
    def __init__(self, img_dir, ext='bmp', transforms=None, is_training=True, with_subfolders=False):
        self.img_dir = img_dir
        if with_subfolders:
            self.img_list = glob.glob(img_dir + "/*/*." + ext)
        else:
            self.img_list = glob.glob(img_dir + "//*." + ext)
        self._idx = -1
        self.is_training = is_training
        self.val_transforms = v2.Compose([
            ToTensor(),
        ])

        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        self._idx = idx
        image = Image.open(img_path)
        image_name = str(self.img_list[idx].split("\\")[-1].split(".")[0])
        if image.mode == "L":
            image = np.array(image).astype(np.uint8)
            image = Image.fromarray(correct_dead_pixel(image), mode="L")
            image = Image.merge("RGB", (image, image, image))
        else:
            image = np.array(image).astype(np.uint8)
            image = Image.fromarray(correct_dead_pixel(image[:, :, 0]), mode="L")
            image = Image.merge("RGB", (image, image, image))         
        if self.transforms:
            image = self.transforms(image)
            return image_name, image
        return image_name, self.val_transforms(image)

    def next(self):
        self._idx += 1
        if self._idx >= self.__len__():
            self._idx = 0

        return self.__getitem__(self._idx)
