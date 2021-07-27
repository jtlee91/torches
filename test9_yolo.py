import torch
import torch.nn as nn
from iou import intersection_over_union
import warnings
warnings.simplefilter("ignore", UserWarning)

architecture_config = [
    # Tuple: Convolution Layer (kernel_size, channel, stride, padding)
    (7, 64, 2, 3),
    # Str: Max Pooling kernel_size=2, stride=2
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: Tuples and an Integer represents num of repeats
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
    def __init__(self, in_channel=3, **kwargs):
        super(Yolov1, self).__init__()
        self.in_channel = in_channel
        self.architecture = architecture_config
        self.darknet = self._create_darknet_layers(self.architecture)
        self.fc = self._create_fc_layers(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = self.fc(x)
        return x

    def _create_darknet_layers(self, architecture):
        layers = list()
        in_channel = self.in_channel
        for x in architecture:
            if type(x) == tuple:
                # Tuple: Convolution Layer
                # (kernel_size, channel, stride, padding)
                # (7, 64, 2, 3)
                layers += [ConvLayer(
                    in_channels=in_channel,
                    out_channels=x[1],
                    kernel_size=x[0],
                    stride=x[2],
                    padding=x[3]
                )]
                in_channel = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                # List: Tuples and an Integer represents num of repeats
                # [(1, 256, 1, 0), (3, 512, 1, 1), 4]
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [ConvLayer(
                        in_channels=in_channel,
                        out_channels=conv1[1],
                        kernel_size=conv1[0],
                        stride=conv1[2],
                        padding=conv1[3]
                    )]

                    layers += [ConvLayer(
                        in_channels=conv1[1],
                        out_channels=conv2[1],
                        kernel_size=conv2[0],
                        stride=conv2[2],
                        padding=conv2[3]
                    )]
                    in_channel = conv2[1]

        return nn.Sequential(*layers)

    def _create_fc_layers(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        x = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * S * S, out_features=4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=4096, out_features=S * S * (B * 5 + C))
        )
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.fc = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # Iobj_i

        # Box Coordinates Loss
        box_predictions = exists_box * (
            (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25])
        )
        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) ==> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # Object Loss
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )
        # (N*S*S, 1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # No Object Loss
        # (N, S, S, 1) ==> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # Class Loss
        # (N, S, S, 20) ==> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss +
            self.lambda_noobj * no_object_loss +
            object_loss +
            class_loss
        )

        return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
X = torch.randn(4, 3, 448, 448).to(device)
print(model(X).shape)
