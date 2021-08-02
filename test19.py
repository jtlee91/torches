import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
torch.manual_seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.architecture = [(1, 128), "M", (128, 64), "M", (64, 32), "M", (32, 16)]
        self.conv_layers = self._create_conv_layers()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_features=16*3*3, out_features=num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

    def _create_conv_layers(self):
        layers = list()
        for x in self.architecture:
            if type(x) == tuple:
                layers += [nn.Sequential(
                    nn.Conv2d(in_channels=x[0], out_channels=x[1], kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(x[1]),
                    nn.LeakyReLU(negative_slope=0.1)
                )]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2))]

        return nn.Sequential(*layers)


def train(model, data_loader, epochs, loss_fn, optimizer):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            preds = model(X)
            loss = loss_fn(preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch : {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item())


def test(model, data_loader):
    model.eval()
    num_correct = 0
    num_size = 0
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            preds = model(X)

            num_correct += (preds.max(1)[1] == label).sum()
            num_size += preds.size(0)

            acc = round(float(num_correct / num_size * 100), 1)

            loop.set_postfix(accuracy=f"{acc}%")


num_classes = 10
batch_size = 32
learning_rate = 3e-4
epochs = 10

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = CNN(num_classes=num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model, train_dataloader, epochs, loss_fn, optimizer)
test(model, test_dataloader)
