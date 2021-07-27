import torch
import torchvision.transforms
from PIL import Image
import os
import numpy as np


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        base_path = "D:/Data/Dataset/archive"
        mask_path = os.path.join(base_path, "with_mask")
        no_mask_path = os.path.join(base_path, "without_mask")
        num_images = 200
        mask_image_path = list(map(lambda x: os.path.join(mask_path, x), os.listdir(mask_path)))[:num_images]
        no_mask_image_path = list(map(lambda x: os.path.join(no_mask_path, x), os.listdir(no_mask_path)))[:num_images]
        mask_image_path = [(i, 1) for i in mask_image_path]
        no_mask_image_path = [(i, 0) for i in no_mask_image_path]
        self.image_path = mask_image_path + no_mask_image_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index][0])
        image = image.resize((448, 448))
        # image = np.array(image)
        label = torch.tensor(self.image_path[index][1])
        if self.transform:
            image = self.transform(image)
        return image, label
