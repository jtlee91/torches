import torch
import torch.nn as nn

architecture_config = [
    # Tuple: Convolution Layer (kernel_size, channel, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
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


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fc = self._create_fc_layers(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = self.fc(x)
        return x

    def _create_conv_layers(self, architecture):
        layers = list()
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [ConvLayer(
                    in_channel=in_channels,
                    out_channel=x[1],
                    kernel_size=x[0],
                    stride=x[2],
                    padding=x[3]
                )]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [ConvLayer(
                        in_channel=in_channels,
                        out_channel=conv1[1],
                        kernel_size=conv1[0],
                        stride=conv1[2],
                        padding=conv1[3]
                    )]

                    layers += [ConvLayer(
                        in_channel=conv1[1],
                        out_channel=conv2[1],
                        kernel_size=conv2[0],
                        stride=conv2[2],
                        padding=conv2[3]
                    )]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fc_layers(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5))
        )

class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, bias=False, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


device = torch.device("cuda")
model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
X = torch.randn(1, 3, 448, 448).to(device)
print(model(X))
