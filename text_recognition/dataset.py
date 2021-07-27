import albumentations as A
import torch
import numpy as np
import os
import config
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Classification:
    def __init__(self, image_paths, resize=None):
        self.image_paths = sorted([os.path.join(image_paths, i) for i in os.listdir(image_paths) if i.endswith("png")])
        self.targets = [os.path.basename(i).split(".")[0] for i in self.image_paths]
        self.resize = resize
        self.aug = A.Compose([A.Normalize(always_apply=True)])
        self.characters = config.characters
        self.targets = [[self.characters.index(w) for w in words] for words in self.targets]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("L")  # FROM RGBA to GRAY SCALE
        targets = self.targets[index]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)
        image = image.reshape((1,) + image.shape)
        # augmented = self.aug(image=image)
        # image = augmented["image"]
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)


if __name__ == "__main__":
    res = Classification(image_paths="C:/Users/power/PycharmProjects/torchdeleteme/src/captcha_images_v2")
    print(res.__getitem__(0))
