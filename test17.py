import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        num_embedding_features = 2
        self.conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 4)  # Identity()
        self.embedding = nn.utils.weight_norm(nn.Linear(4, num_embedding_features, bias=False), dim=1)
        self.fc = nn.Linear(num_embedding_features, 10)

    def forward(self, x, label=None):
        x = self.conv(x)
        x = self.resnet(x)
        for W in self.embedding.parameters():
            print(torch.sum(torch.multiply(W, W), dim=1))
            W = F.normalize(W, p=2, dim=1)
            print(torch.sum(torch.multiply(W, W), dim=1))

        x = F.normalize(x, p=2, dim=1)
        embedding_features = self.embedding(x)


        if label is not None:
            class_features = self.fc(embedding_features)
            loss_fn = CELoss()  # nn.CrossEntropyLoss()
            loss = loss_fn(class_features, label)
            return class_features, embedding_features, loss
        return embedding_features


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, x, label):
        one_hot_encoding = torch.zeros(x.shape[0], 10).to(device)
        one_hot_encoding[range(x.shape[0]), label] = 1
        to_softmax = torch.div(torch.exp(x), torch.sum(torch.exp(x), dim=1).unsqueeze(1))
        cross_entropy = torch.multiply(one_hot_encoding, torch.log(to_softmax))
        loss = -cross_entropy.mean()
        return loss


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def train(model, data_loader, epochs, optimizer):
    num_correct = 0
    num_size = 0
    writer = SummaryWriter(f"logs/mnist")
    class_labels = list("0123456789")
    step = 1
    model.train()
    for epoch in range(epochs):
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            class_features, embedding_features, loss = model(X, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += (class_features.max(1)[1] == label).sum()
            num_size += class_features.size(0)
            accuracy = round(float(num_correct / num_size * 100), 1)

            loop.set_description(f"Epoch: {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item(), accuracy=f"{accuracy}%")

            writer.add_scalar("accuracy", accuracy, global_step=step)
            writer.add_scalar("loss", loss.item(), global_step=step)
            if batch_idx % 467 == 0:
                classes = [class_labels[i] for i in class_features.max(1)[1]]
                writer.add_embedding(embedding_features, classes, X, global_step=step)
            step += 1


def test(model, data_loader):
    num_correct = 0
    num_size = 0
    model.eval()
    with torch.no_grad():
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            class_features, embedding_features, loss = model(X, label)

            num_correct += (class_features.max(1)[1] == label).sum()
            num_size += class_features.size(0)
            accuracy = round(float(num_correct / num_size * 100), 1)

            loop.set_postfix(accuracy=f"{accuracy}%")


batch_size = 2
learning_rate = 3e-4
epochs = 5

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = MnistModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model, train_dataloader, epochs, optimizer)
test(model, test_dataloader)
