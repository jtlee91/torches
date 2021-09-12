import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import numpy as np
import random
import warnings
warnings.simplefilter("ignore", UserWarning)

manual_seed = 100
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AN(nn.Module):
    def __init__(self, num_classes):
        super(AN, self).__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28*28, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


num_classes = 10
batch_size = 64
learning_rate = 3e-4
epochs = 10

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def train(model, dataloader, epochs, loss_fn, optimizer):
    for epoch in range(epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        num_correct = 0
        num_size = 0
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            preds = model(X)
            loss = loss_fn(preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += (preds.max(1)[1] == label).sum()
            num_size += preds.shape[0]
            epoch_accuracy = float(num_correct / num_size)

            loop.set_description(f"Epoch: {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item(), epoch_accuracy=f"{epoch_accuracy:.3f}")


def test(model, dataloader):
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    num_correct = 0
    num_size = 0
    for batch_idx, (X, label) in loop:
        X, label = X.to(device), label.to(device)

        preds = model(X)

        num_correct += (preds.max(1)[1] == label).sum()
        num_size += preds.shape[0]
        epoch_accuracy = float(num_correct / num_size)

        loop.set_postfix(epoch_accuracy=f"{epoch_accuracy:.3f}")


model = AN(num_classes=num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

train(model, train_dataloader, epochs, loss_fn, optimizer)
test(model, test_dataloader)
