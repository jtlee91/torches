import torch
import torch.nn as nn
import torchvision
from yolo_dataset import VOCDataset
import warnings
warnings.simplefilter("ignore", UserWarning)

manual_seed = 100
torch.manual_seed(manual_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (input_channel, out_channel, kernel_size, stride, padding)
architecture = [
    (3, 64, 7, 2, 3), "M",
    (64, 192, 3, 1, 1), "M",
    (192, 128, 1, 1, 0),
    (128, 256, 3, 1, 1),
    (256, 256, 1, 1, 0),
    (256, 512, 3, 1, 1), "M",
    [(512, 256, 1, 1, 0), (256, 512, 3, 1, 1), 4],
    (512, 512, 1, 1, 0),
    (512, 1024, 3, 1, 1), "M",
    [(1024, 512, 1, 1, 0), (512, 1024, 3, 1, 1), 2],
    (1024, 1024, 3, 1, 1),
    (1024, 1024, 3, 2, 1),
    (1024, 1024, 3, 1, 1),
    (1024, 1024, 3, 1, 1)
]


class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        self.conv_layers = self._create_conv_layer(architecture)
        self.fc_layers = self._create_fc_layer(num_splits=7, num_bboxes=2, num_classes=20)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def _create_conv_layer(self, architecture):
        layers = list()
        for x in architecture:
            if type(x) == tuple:
                layers += [Conv(input_channel=x[0], out_channel=x[1], kernel_size=x[2], stride=x[3], padding=x[4])]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                for _ in range(x[-1]):
                    layers += [Conv(input_channel=x[0][0], out_channel=x[0][1], kernel_size=x[0][2], stride=x[0][3], padding=x[0][4])]
                    layers += [Conv(input_channel=x[1][0], out_channel=x[1][1], kernel_size=x[1][2], stride=x[1][3], padding=x[1][4])]

        return nn.Sequential(*layers)

    def _create_fc_layer(self, num_splits, num_bboxes, num_classes):
        S, B, C = num_splits, num_bboxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * 7 * 7, out_features=4096),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=S * S * (B * 5 + C))
        )


class Conv(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C

    def forward(self, preds, targets=None):
        print(preds.shape)
        preds = preds.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        print(preds.shape)
        box_class = preds[..., 00:20]
        box1_bbox = preds[..., 20:24]
        box1_conf = preds[..., 24:25]
        box2_bbox = preds[..., 25:29]
        box2_conf = preds[..., 29:30]
        iou_box1 = intersection_over_union(box1_bbox, targets[..., 20:24])
        iou_box2 = intersection_over_union(box2_bbox, targets[..., 20:24])
        print(iou_box1.shape)
        print(iou_box2.shape)
        print(iou_box1.unsqueeze(0).shape)
        print(iou_box2.unsqueeze(0).shape)
        ious = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)
        print(ious.shape)
        iou_maxes, best_box = torch.max(ious, dim=0)
        print(iou_maxes.shape, best_box.shape)
        exists_box = targets[..., 24].unsqueeze(3)  # Iobj_i
        print(exists_box.shape)


def intersection_over_union(boxes_preds, boxes_labels):
    """
    :param boxes_preds: (BATCH_SIZE, 4)
    :param boxes_labels: (BATCH_SIZE, 4)
    :return:
    """
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]
    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def main():
    model = Yolo().to(device)
    X = torch.randn(8, 3, 448, 448).to(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((448, 448)),
        torchvision.transforms.ToTensor()
    ])

    train_dataset = VOCDataset(
        "D:/Data/PascalVOC_YOLO/train.csv",
        "D:/Data/PascalVOC_YOLO/images",
        "D:/Data/PascalVOC_YOLO/labels",
        S=7, B=2, C=20, transform=transform
    )
    test_dataset = VOCDataset(
        "D:/Data/PascalVOC_YOLO/test.csv",
        "D:/Data/PascalVOC_YOLO/images",
        "D:/Data/PascalVOC_YOLO/labels",
        S=7, B=2, C=20, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=8
    )

    for image, targets in train_loader:
        image = image.to(device)
        targets = targets.to(device)
        print(targets.shape)
        break

    loss_fn = YoloLoss(S=7, B=2, C=20).to(device)
    loss_fn(model(X), targets)


if __name__ == "__main__":
    main()
