import torch
import os
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.img_paths = list()
        self.transform = transform
        for (root, dirs, files) in os.walk(base_path):
            for file in files:
                self.img_paths.append(os.path.join(root, file))
        self.img_paths = sorted(self.img_paths)
        self.labels = sorted(os.listdir(self.base_path))
        self.label_ohe = torch.zeros(len(os.listdir(self.base_path))).to(device)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        index_image_class = self.labels.index("_".join(os.path.basename(self.img_paths[index]).split("_")[:-1]))
        index_image_class = torch.tensor(index_image_class).to(device)
        # self.label_ohe[index_image_class] = 1

        image = Image.open(self.img_paths[index], mode="r")
        image = image.resize((224, 224))
        if self.transform:
            image = self.transform(image)

        return image, index_image_class
