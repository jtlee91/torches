import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Conv2d, MaxPool2d, LeakyReLU
from torchsummary import summary
import warnings
warnings.simplefilter("ignore", UserWarning)

# model
architecture_config = [
    # Tuple: (kernel_size, channels, stride, padding)
    (7, 64, 2, 3),
    # String: Max pooling
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: Tuples and the last integer stands for num of repeats
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = list()
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(
                    in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]

                in_channels = x[1]

            elif type(x) == str:
                layers += [MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0]  # Tuple
                conv2 = x[1]  # Tuple
                num_repeats = x[2]  # Integer

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ]

                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]

        return Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 128),  # Original Paper 4096 -> 128
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(128, S * S * (C + B * 5)),  # (S, S, 30) C + B * 5 = 30
        )


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lamda_noobj = 0.5
        self.lamda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 =

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
# X = torch.randn(2, 3, 448, 448).to(device)
# print(model(X).shape)
summary(model, input_size=(3, 448, 448))
