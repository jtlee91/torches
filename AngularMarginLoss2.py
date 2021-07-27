import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
torch.manual_seed(100)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.architecture = [(1, 128), "M", (128, 64), "M", (64, 32)]
        self.conv = self._conv_layer()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=128)

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

    def _conv_layer(self):
        layers = list()
        for x in self.architecture:
            if type(x) == tuple:
                layers += [
                    nn.Conv2d(in_channels=x[0], out_channels=x[1], kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(num_features=x[1]),
                    nn.LeakyReLU(negative_slope=0.1)
                ]

            elif type(x) == str:
                layers += [nn.MaxPool2d(2, 2)]

        return nn.Sequential(*layers)


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        self.feature = CNN()  # out : 128 features
        self.loss = SoftMaxLoss()
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, in_data, label, embed=False, val=False):
        feature = self.feature(in_data)  # X : 128 features
        if embed:
            return feature
        X = self.fc(feature)  # X : 10 features
        # if val:
        #     print(f"\nLabel: {label[:5]}\nFeature: {X[:5]}")
        num_correct = (X.max(1)[1] == label).sum()
        loss = self.loss(X, label)
        return loss, num_correct, feature


class SoftMaxLoss(nn.Module):
    def __init__(self):
        super(SoftMaxLoss, self).__init__()

    def forward(self, in_feature, label):
        label_ohe = torch.zeros(in_feature.shape[0], 10).to(device)
        label_ohe[range(in_feature.shape[0]), label] = 1

        loss = torch.sum(torch.multiply(
            label_ohe,
            torch.log(torch.div(torch.exp(in_feature), torch.sum(torch.exp(in_feature), dim=1).unsqueeze(1)))
        ), dim=1)
        return -loss.mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
learning_rate = 3e-4
epochs = 10

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = Trainer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter(f"tensorboard/MINST128feature")


def train(model, data, epochs, optimizer):
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    step = 0
    corrects = 0
    size = 0
    for epoch in range(epochs):
        loop = tqdm(enumerate(data))
        for batch_idx, (X, y) in loop:
            X, y = X.to(device), y.to(device)

            if batch_idx == 0:
                loss, num_correct, feature = model(X, y, val=True)
            else:
                loss, num_correct, feature = model(X, y)

            corrects += num_correct
            size += X.shape[0]
            acc = round(float(corrects / size * 100), 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch: {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item(), accuracy=acc)

            class_labels = [classes[label] for label in y]
            writer.add_scalar("Training Loss", loss, global_step=step)
            writer.add_scalar("Training Accuracy", acc, global_step=step)
            if batch_idx == 230:
                writer.add_embedding(feature, metadata=class_labels, label_img=X, global_step=step)
            step += 1


train(model, train_dataloader, epochs, optimizer)

for idx, (X, y) in enumerate(test_dataloader):
    X, y = X.to(device), y.to(device)

    print(model(X, None, embed=True))
    break
