import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchsummaryX import summary
from tqdm import tqdm
import numpy as np
import random
import warnings
warnings.simplefilter("ignore", UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_seed = 100
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
cudnn.benchmark = True
cudnn.deterministic = True


class CNN(nn.Module):
    def __init__(self, num_classes, embedding_size=64):
        super(CNN, self).__init__()
        self.architecture = [(1, 128), "M", (128, 64), "M", (64, 32), "M", (32, 16)]
        self.conv_layer = self._crate_conv_layer()
        self.flat = nn.Flatten()
        self.embedding_feature = nn.Linear(in_features=16*3*3, out_features=embedding_size)

        self.fc = nn.Linear(in_features=embedding_size, out_features=num_classes, bias=False)

    def forward(self, x, label=None):
        x = self.conv_layer(x)
        x = self.flat(x)
        embedding_feature = self.embedding_feature(x)
        if label is not None:
            embedding_feature = F.normalize(embedding_feature, dim=1)  # (N, embedding_features)
            W = self.fc.weight
            W = F.normalize(W, dim=1)
            W = torch.transpose(W, 1, 0)  # (embedding_features, n)
            cos_theta = torch.matmul(embedding_feature, W)  # (N, n)
            # class_feature = self.fc(embedding_feature)

            loss_fn = CrossEntropyLoss().to(device)
            class_feature, loss = loss_fn(cos_theta, label)
            return embedding_feature, class_feature, loss
        return embedding_feature

    def _crate_conv_layer(self):
        layers = list()
        for x in self.architecture:
            if type(x) == tuple:
                layers += [
                    nn.Conv2d(in_channels=x[0], out_channels=x[1], kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(num_features=x[1]),
                    nn.LeakyReLU(negative_slope=0.1)
                ]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2))]

        return nn.Sequential(*layers)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.s = 1
        self.m = 0.5  # radian

    def forward(self, cos_theta, label):
        label_size = cos_theta.shape[0]
        label_ohe = torch.zeros(label_size, num_classes).to(device)
        label_ohe[range(label_size), label] = 1

        cos_theta = cos_theta * self.s

        x_softmax = torch.divide(torch.exp(cos_theta), torch.sum(torch.exp(cos_theta), dim=1).unsqueeze(1))
        x_softmax_log = torch.log10(x_softmax)
        ce_loss = torch.sum(torch.multiply(label_ohe, x_softmax_log), dim=1)
        loss = -torch.mean(ce_loss)
        return cos_theta, loss


class CNN2(nn.Module):
    def __init__(self, num_classes, embedding_size=64):
        super(CNN2, self).__init__()
        self.architecture = [(1, 128), "M", (128, 64), "M", (64, 32), "M", (32, 16)]
        self.conv_layer = self._crate_conv_layer()
        self.flat = nn.Flatten()
        self.embedding_feature = nn.Linear(in_features=16*3*3, out_features=embedding_size)

        self.fc = nn.Linear(in_features=embedding_size, out_features=num_classes)

    def forward(self, x, label=None):
        x = self.conv_layer(x)
        x = self.flat(x)
        embedding_feature = self.embedding_feature(x)
        if label is not None:
            class_feature = self.fc(embedding_feature)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(class_feature, label)
            return embedding_feature, class_feature, loss
        return embedding_feature

    def _crate_conv_layer(self):
        layers = list()
        for x in self.architecture:
            if type(x) == tuple:
                layers += [
                    nn.Conv2d(in_channels=x[0], out_channels=x[1], kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(num_features=x[1]),
                    nn.LeakyReLU(negative_slope=0.1)
                ]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2))]

        return nn.Sequential(*layers)


def train(model, dataloader, epochs, optimizer):
    model.train()
    num_correct =0
    num_size = 0
    for epoch in range(epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_dix, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            embedding_feature, class_feature, loss = model(X, label)

            num_correct += (class_feature.max(1)[1] == label).sum()
            num_size += class_feature.shape[0]
            accuracy = round(float(num_correct / num_size * 100), 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch: {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item(), accuracy=f"{accuracy}%")


def test(model, dataloader):
    model.eval()
    num_correct = 0
    num_size = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for batch_dix, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            embedding_feature, class_feature, loss = model(X, label)

            num_correct += (class_feature.max(1)[1] == label).sum()
            num_size += class_feature.shape[0]
            accuracy = round(float(num_correct / num_size * 100), 1)

            loop.set_postfix(accuracy=f"{accuracy}%")


num_classes = 10
embedding_size = 64
batch_size = 128
learning_rate = 3e-4
epochs = 3

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("Angular Margin Loss")
model = CNN(num_classes=num_classes, embedding_size=embedding_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model, train_dataloader, epochs, optimizer)
test(model, test_dataloader)


print("Softmax Loss")
model2 = CNN2(num_classes=num_classes, embedding_size=embedding_size).to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)

train(model2, train_dataloader, epochs, optimizer)
test(model2, test_dataloader)
