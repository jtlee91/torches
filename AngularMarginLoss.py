import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.simplefilter("ignore", UserWarning)
torch.manual_seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AN(nn.Module):
    def __init__(self, input_size, out_feature):
        super(AN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=out_feature)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MSELoss(nn.Module):
    def __init__(self, feature, num_classes):
        super(MSELoss, self).__init__()
        self.fc = nn.Linear(in_features=feature, out_features=num_classes)

    def forward(self, X, ground_truth):
        # in_feature = 4
        # ground_truth = 10
        X = self.fc(X)
        ground_truth_ohe = torch.zeros(X.shape[0], 10).to(device)
        ground_truth_ohe[range(X.shape[0]), ground_truth] = 1
        loss = torch.sqrt(torch.sum((X - ground_truth_ohe) ** 2, dim=1))
        loss = loss.mean()
        return loss


class LossWrapper(nn.Module):
    def __init__(self):
        super(LossWrapper, self).__init__()
        num_feature = 1
        self.feature = AN(input_size=28*28, out_feature=num_feature)
        self.loss = MSELoss(feature=num_feature, num_classes=10)

    def forward(self, X, label, embed=False):
        X = self.feature(X)  # X.shape: num_feature = 3
        if embed:
            return X
        loss = self.loss(X, label)
        return loss


batch_size = 64
learning_rate = 3e-4
epochs = 10

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = LossWrapper().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    loop = tqdm(enumerate(train_dataloader))
    for batch_idx, (X, label) in loop:
        X, label = X.to(device), label.to(device)

        loss = model(X, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epochs: {epoch + 1} / {epochs}")
        loop.set_postfix(loss=loss.item())


for X, label in test_dataloader:
    X, label = X.to(device), label.to(device)

    inference = model(X, None, embed=True)
    print(inference)
    print(label)
    break