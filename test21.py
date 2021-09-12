import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import warnings
from tqdm import tqdm

torch.manual_seed(100)
warnings.simplefilter("ignore", UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, num_classes, embedding_size=64):
        super(CNN, self).__init__()
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
            return class_feature, embedding_feature, loss
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


model = CNN(num_classes=10, embedding_size=64).to(device)


def train(model, dataloader, epochs, optimizer):
    model.train()
    num_correct =0
    num_size = 0
    for epoch in range(epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_dix, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            class_feature, embedding_feature, loss = model(X, label)

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

            class_feature, embedding_feature, loss = model(X, label)

            num_correct += (class_feature.max(1)[1] == label).sum()
            num_size += class_feature.shape[0]
            accuracy = round(float(num_correct / num_size * 100), 1)

            loop.set_postfix(accuracy=f"{accuracy}%")


num_classes = 10
batch_size = 32
learning_rate = 3e-4
epochs = 3

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = CNN(num_classes=10, embedding_size=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model, train_dataloader, epochs, optimizer)
test(model, test_dataloader)
