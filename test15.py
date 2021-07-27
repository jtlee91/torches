import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
torch.manual_seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        num_embedding_feature = 512
        self.architecture = [(3, 128), "M", (128, 64), "M", (64, 32)]
        self.conv_layer = self._create_conv_layer(self.architecture)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*56*56, out_features=num_embedding_feature)
        self.fc2 = nn.Linear(num_embedding_feature, num_classes)

    def forward(self, x, label=None):
        x = self.conv_layer(x)
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        embedding_feature = self.fc1(x)
        # print(x.shape)
        if label is not None:
            class_feature = self.fc2(embedding_feature)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(class_feature, label)
            return class_feature, embedding_feature, loss
        return embedding_feature

    def _create_conv_layer(self, architecture):
        layers = list()
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    nn.Conv2d(in_channels=x[0], out_channels=x[1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(x[1]),
                    nn.LeakyReLU(0.1)
                ]

            elif type(x) == str:
                layers += [
                    nn.MaxPool2d(kernel_size=2)
                ]

        return nn.Sequential(*layers)


def save_checkpoint(state, filename):
    print("=> Saving check point")
    torch.save(state, filename)


def train(model, data_loader, epochs, optimizer):
    writer = SummaryWriter(f"logs/face")
    model.train()
    num_correct = 0
    num_size = 0
    step = 1
    class_labels = list(range(136))

    for epoch in range(epochs):
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            preds, embedding_features, loss = model(X, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += (preds.max(1)[1] == label).sum()
            num_size += preds.size(0)

            accuracy = round(float(num_correct / num_size * 100), 1)

            loop.set_description(f"Epochs: {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item(), accuracy=accuracy)

            classes = [class_labels[i] for i in preds.max(1)[1]]
            writer.add_scalar("loss", loss.item(), global_step=step)
            if batch_idx % 1300 == 0:
                writer.add_embedding(mat=embedding_features, metadata=classes, label_img=X, global_step=step)

            step += 1

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, f"my_checkpoint_{epoch}_{accuracy}.pth.tar")

def test(model, data_loader):
    model.eval()
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    num_correct = 0
    num_size = 0
    with torch.no_grad():
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            preds, _, _ = model(X, label)

            num_correct += (preds.max(1)[1] == label).sum()
            num_size += preds.size(0)

            accuracy = round(float(num_correct / num_size * 100), 1)

            loop.set_postfix(accuracy=f"{accuracy}%")


batch_size = 8
learning_rate = 3e-4
epochs = 50

model = CNN(num_classes=136).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor())
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor())
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
from test18_dataset import FaceDataset
dataset = FaceDataset(base_path="C:/Users/power/PycharmProjects/CreateFaceData/dest", transform=torchvision.transforms.ToTensor())
num_dataset = len(dataset)
train_set, test_set = torch.utils.data.random_split(dataset, [int(num_dataset*0.8), num_dataset - int(num_dataset*0.8)])
train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# for X, label in train_dataloader:
#     X, label = X.to(device), label.to(device)
#     break
#
# print(model(X))
# print(model(X).shape)
train(model, train_dataloader, epochs, optimizer)
test(model, test_dataloader)
