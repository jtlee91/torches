import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import warnings
warnings.simplefilter("ignore", UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MnistModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=128, bias=True)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes, bias=True)
        self.relu = nn.LeakyReLU(0.1)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(model, dataloader, epochs, loss_fn, optimizer):
    model.train()
    for epoch in range(epochs):
        num_correct = 0
        num_size = 0
        for X, label in dataloader:
            X, label = X.to(device), label.to(device)

            preds = model(X)
            loss = loss_fn(preds, label)

            num_correct += (preds.max(1)[1] == label).sum()
            num_size += preds.shape[0]
            epoch_accuracy = float(num_correct / num_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch + 1} / {epochs}", end="\t")
        print(f"Accuracy: {epoch_accuracy}")


input_size = 28 * 28
num_classes = 10
batch_size = 32
learning_rate = 3e-4
epochs = 10

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = MnistModel(input_size, num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

train(model, train_dataloader, epochs, loss_fn, optimizer)
