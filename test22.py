import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
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


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.avgpool = nn.Flatten()
        self.backbone.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = self.backbone(x)
        return x


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

    def forward(self, x, label):
        label_ohe = torch.zeros(x.shape[0], num_classes).to(device)
        label_ohe[range(x.shape[0]), label] = 1
        x_softmax = torch.divide(torch.exp(x), torch.sum(torch.exp(x), dim=1).unsqueeze(1))
        x_softmax_log = torch.log10(x_softmax)
        ce_loss = torch.sum(torch.multiply(label_ohe, x_softmax_log), dim=1)
        loss = -torch.mean(ce_loss)
        return loss


class SoftmaxLoss2(nn.Module):
    def __init__(self):
        super(SoftmaxLoss2, self).__init__()

    def forward(self, x, label):
        label_ohe = torch.zeros(x.shape[0], num_classes).to(device)
        label_ohe[range(x.shape[0]), label] = 1
        x_softmax = torch.divide(torch.exp(x), torch.sum(torch.exp(x), dim=1).unsqueeze(1))
        x_softmax_log = torch.log(x_softmax)
        ce_loss = torch.sum(torch.multiply(label_ohe, x_softmax_log), dim=1)
        loss = -torch.mean(ce_loss)
        return loss


class SoftmaxLoss3(nn.Module):
    def __init__(self):
        super(SoftmaxLoss3, self).__init__()

    def forward(self, x, label):
        one_hot_encoding = torch.zeros(x.shape[0], num_classes).to(device)
        one_hot_encoding[range(x.shape[0]), label] = 1

        to_softmax = torch.div(torch.exp(x), torch.sum(torch.exp(x), dim=1).unsqueeze(1))
        cross_entropy = torch.sum(torch.multiply(one_hot_encoding, torch.log(to_softmax)), dim=1)
        loss = -cross_entropy.mean()
        return loss



def train(model, dataloader, epochs, loss_fn, optimizer):
    model.train()
    num_correct = 0
    num_size = 0
    for epoch in range(epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        loss_sum = 0
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            preds = model(X)
            loss = loss_fn(preds, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += (preds.max(1)[1] == label).sum()
            num_size += preds.shape[0]
            accuracy = float(num_correct / num_size * 100)
            loss_sum += loss

            loop.set_description(f"Epoch: {epoch + 1} / {epochs}")
            loop.set_postfix(loss_sum=f"{loss_sum.item():.2f}", accuracy=f"{accuracy:.2f}")


def test(model, dataloader):
    model.eval()
    num_correct = 0
    num_size = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            preds = model(X)

            num_correct += (preds.max(1)[1] == label).sum()
            num_size += preds.shape[0]
            accuracy = float(num_correct / num_size * 100)

            loop.set_postfix(accuracy=f"{accuracy:.2f}")


num_classes = 10
batch_size = 128
epochs = 5
learning_rate = 3e-4

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = EmbeddingNet().to(device)
loss_fn = SoftmaxLoss2().to(device)

X = torch.randn(4, 10).to(device)
label = torch.tensor([0, 1, 2, 3]).to(device)
print(loss_fn(X, label))

loss_fn2 = nn.CrossEntropyLoss()
print(loss_fn2(X, label))

loss_fn3 = SoftmaxLoss3().to(device)
print(loss_fn3(X, label))


# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# train(model, train_dataloader, epochs, loss_fn, optimizer)
# test(model, test_dataloader)

# model = EmbeddingNet().to(device)
# loss_fn = SoftmaxLoss2().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# train(model, train_dataloader, epochs, loss_fn, optimizer)
# test(model, test_dataloader)
#
# model = EmbeddingNet().to(device)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# train(model, train_dataloader, epochs, loss_fn, optimizer)
# test(model, test_dataloader)
