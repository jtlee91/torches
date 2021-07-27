import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from dataset2 import MaskDataset
import warnings
warnings.simplefilter("ignore", UserWarning)


class AN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3))
        self.mp = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_features=390728, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp(x)
        x = F.relu(self.conv2(x))
        x = self.flat(x)
        x = self.fc(x)
        return x


class ResnetMnist(nn.Module):
    def __init__(self, num_classes):
        super(ResnetMnist, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.avgpool = Identity()
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.resnet(x)
        x = self.flat(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
input_size = 448 * 448
num_classes = 2
learning_rate = 3e-4  # For Adam
batch_size = 16
epochs = 5

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=transform, download=True)
train_dataset, test_dataset = torch.utils.data.random_split(MaskDataset(transform=transform), [320, 80])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=transform, download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def train(model, dataloader, epochs, loss_fn, optimizer):
    for epoch in range(epochs):
        loop = tqdm(dataloader)
        for X, y in loop:
            X, y = X.to(device), y.to(device)

            inference = model(X)
            loss = loss_fn(inference, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch: {epoch+1} / {epochs}")
            loop.set_postfix(loss=loss.item())


def test(model, dataloader):
    loop = tqdm(dataloader)
    num_correct = 0
    num_size = 0
    model.eval()
    with torch.no_grad():
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            inference = model(X)

            num_correct += (inference.max(1)[1] == y).sum()
            num_size += inference.size(0)
            acc = round(float(num_correct / num_size * 100), 2)

            loop.set_postfix(acc=acc)


# model = AN(input_size=input_size, num_classes=num_classes).to(device)
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# test(model, test_dataloader)
# train(model, train_dataloader, epochs, loss_fn, optimizer)
# test(model, test_dataloader)
#
# model = CNN(in_channels=3, num_classes=num_classes).to(device)
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# test(model, test_dataloader)
# train(model, train_dataloader, epochs, loss_fn, optimizer)
# test(model, test_dataloader)

model = ResnetMnist(num_classes=num_classes).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# test(model, test_dataloader)
train(model, train_dataloader, epochs, loss_fn, optimizer)
# test(model, test_dataloader)
