import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import os
from PIL import Image
from pandas import read_csv
import warnings
warnings.simplefilter("ignore", UserWarning)

# Tuple: (in_channel, out_channel, kernel_size, stride, padding)
architecture = [
    (3, 64, 7, 2, 3),
    "M",
    (64, 192, 3, 1, 1),
    "M",
    (192, 128, 1, 1, 0),
    (128, 256, 3, 1, 1),
    (256, 256, 1, 1, 0),
    (256, 512, 3, 1, 1),
    "M",
    [(512, 256, 1, 1, 0), (256, 512, 3, 1, 1), 4],
    (512, 512, 1, 1, 0),
    (512, 1024, 3, 1, 1),
    "M",
    [(1024, 512, 1, 1, 0), (512, 1024, 3, 1, 1), 2],
    (1024, 1024, 3, 1, 1),
    (1024, 1024, 3, 2, 1),
    (1024, 1024, 3, 1, 1),
    (1024, 1024, 3, 1, 1)
]


class YoloModel(nn.Module):
    def __init__(self):
        super(YoloModel, self).__init__()
        self.darknet_layer = self._create_darknet_layers(architecture)
        self.fc_layer = self._create_fc_layers(S=7, B=2, C=20)

    def forward(self, x):
        x = self.darknet_layer(x)
        x = self.fc_layer(x)
        return x

    def _create_darknet_layers(self, architecture):
        layers = list()
        for x in architecture:
            if type(x) == tuple:
                layers += [self._conv(x[0], x[1], x[2], x[3], x[4])]
            elif type(x) == str:
                layers += [nn.MaxPool2d(2, 2)]
            elif type(x) == list:
                for _ in range(x[-1]):
                    layers += [self._conv(x[0][0], x[0][1], x[0][2], x[0][3], x[0][4])]
                    layers += [self._conv(x[1][0], x[1][1], x[1][2], x[1][3], x[1][4])]

        return nn.Sequential(*layers)

    def _conv(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1)
        )

    def _create_fc_layers(self, S, B, C):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024*7*7, out_features=4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=S * S * (B * 5 + C))
        )


class VOCDataset(Dataset):
    def __init__(self):



class VOCDataset(Dataset):
    def __init__(self, csv_file_path, image_dir_path, label_dir_path, S=7, B=2, C=20, transform=None):
        self.S = S
        self.B = B
        self.C = C
        self.annotations = read_csv(csv_file_path, header=None)
        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir_path, self.annotations.iloc[index, 0])
        label_path = os.path.join(self.label_dir_path, self.annotations.iloc[index, 1])
        image = Image.open(image_path)

        bbox = list()
        with open(label_path, "r", encoding="utf-8") as f:
            labels = f.readlines()
            for label in labels:
                label = label.replace("\n", "").split()
                label_class, label_cx, label_cy, label_w, label_h = label
                bbox.append([int(label_class), float(label_cx), float(label_cy), float(label_w), float(label_h)])
        bbox = torch.tensor(bbox)
        if self.transform:
            image, bbox = self.transform(image), bbox

        # predictions.shape: (batch_size, S, S, B * 5 + C)   [-1, 7, 7, 30 in paper]
        label_matrix = torch.zeros(self.S, self.S, self.C + 5)
        for box in bbox:
            _label_class, _cx, _cy, _w, _h = box.tolist()
            _label_class = int(_label_class)
            obj_grid_x_num, obj_grid_y_num = int(self.S * _cx), int(self.S * _cy)
            obj_grid_cx, obj_grid_cy = self.S * _cx - obj_grid_x_num, self.S * _cy - obj_grid_y_num
            obj_grid_w, obj_grid_h = self.S * _w, self.S * _h
            box_coordinates = torch.tensor([obj_grid_cx, obj_grid_cy, obj_grid_w, obj_grid_h])
            if label_matrix[obj_grid_x_num, obj_grid_y_num, 24] == 0:
                label_matrix[obj_grid_x_num, obj_grid_y_num, 24] = 1
                label_matrix[obj_grid_x_num, obj_grid_y_num, 20:24] = box_coordinates
                label_matrix[obj_grid_x_num, obj_grid_y_num, _label_class] = 1

        return image, label_matrix


if __name__ == "__main__":
    model = YoloModel().to("cuda")
    X = torch.randn(4, 3, 448, 448).to("cuda")
    print(model(X).shape)
