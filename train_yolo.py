import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from test9_yolo import Yolov1, YoloLoss
from dataset1 import VOCDataset


seed = 123
torch.manual_seed(seed)

# Hyper Parameters
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
weight_decay = 0
epochs = 100
num_workers = 2
img_dir = "D:/Data/PascalVOC_YOLO/images"
label_dir = "D:/Data/PascalVOC_YOLO/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader)
    mean_loss = list()

    for batch_idx, (X, y) in enumerate(loop):
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss : {sum(mean_loss) / len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = YoloLoss()

    train_dataset = VOCDataset(
        "D:/Data/PascalVOC_YOLO/8examples.csv",
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )
    test_dataset = VOCDataset(
        "D:/Data/PascalVOC_YOLO/test.csv",
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(epochs):
        train(train_loader, model, optimizer, loss_fn)


main()