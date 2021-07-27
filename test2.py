import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, MaxPool2d
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)


class AN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(AN, self).__init__()
        self.fc1 = Linear(in_features=input_size, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3))
        self.conv2 = Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3))
        self.conv3 = Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3))
        self.mp = MaxPool2d(kernel_size=(2, 2))
        self.fc = Linear(in_features=16 * 3 * 3, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 28 - 3 + 1 = 26
        x = self.mp(x) # 26 / 2 = 13
        x = F.relu(self.conv2(x)) # 13 - 3 + 1 = 11
        x = self.mp(x) # 11 / 2 = 5
        x = self.conv3(x) # 5 - 3 + 1 = 3  ==> (16, 3, 3)
        x = x.reshape(x.shape[0], -1) # 16 * 3 * 3
        x = self.fc(x)
        return x


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResnetMNIST(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResnetMNIST, self).__init__()
        self.conv = Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.avgpool = Identity()
        self.resnet18.fc = Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.resnet18(x)
        return x


class Resnet50MNIST(torch.nn.Module):
    def __init__(self, num_classes):
        super(Resnet50MNIST, self).__init__()
        self.conv = Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.avgpool = Identity()
        self.resnet50.fc = Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.resnet50(x)
        return x


class Resnet152MNIST(torch.nn.Module):
    def __init__(self, num_classes):
        super(Resnet152MNIST, self).__init__()
        self.conv = Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.resnet152 = torchvision.models.resnet152(pretrained=True)
        self.resnet152.avgpool = Identity()
        self.resnet152.fc = Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.resnet152(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

input_size = 28 * 28
num_classes = 10
batch_size = 64
learning_rate = 1e-3
epochs = 10

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=data_transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=data_transform, download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


def train(model, dataloader, loss_fn, optimizer, epochs, is_cnn=False):
    for epoch in range(epochs):
        loop = tqdm(dataloader, total=len(dataloader))
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            if not is_cnn:
                X = X.reshape(X.shape[0], -1)

            inference = model(X)
            loss = loss_fn(inference, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"epoch : {epoch + 1} / {epochs}")
            loop.set_postfix(Loss=loss.item())


def test(model, dataloader, is_cnn=False):
    num_correct = 0
    num_total = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        if not is_cnn:
            X = X.reshape(X.shape[0], -1)

        inference = model(X)

        num_correct += (inference.max(1)[1] == y).sum()
        num_total += inference.size(0)
        accuracy = round(float(num_correct / num_total * 100), 2)

        loop.set_description("testing accuracy")
        loop.set_postfix(accuracy=f"{accuracy}%")


print("AN")
model = AN(input_size=input_size, num_classes=num_classes).to(device)

loss_fn = torch.nn.CrossEntropyLoss()  # softmax + CEL
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs)
test(model, train_dataloader)
test(model, test_dataloader)

print("CNN")
model = CNN(num_classes=num_classes).to(device)

loss_fn = torch.nn.CrossEntropyLoss()  # softmax + CEL
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, is_cnn=True)
test(model, train_dataloader, is_cnn=True)
test(model, test_dataloader, is_cnn=True)

print("Resnet18")
model = ResnetMNIST(num_classes=num_classes).to(device)

loss_fn = torch.nn.CrossEntropyLoss()  # softmax + CEL
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, is_cnn=True)
test(model, train_dataloader, is_cnn=True)
test(model, test_dataloader, is_cnn=True)

print("Resnet50")
model = Resnet50MNIST(num_classes=num_classes).to(device)

loss_fn = torch.nn.CrossEntropyLoss()  # softmax + CEL
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, is_cnn=True)
test(model, train_dataloader, is_cnn=True)
test(model, test_dataloader, is_cnn=True)

print("Resnet152")
model = Resnet152MNIST(num_classes=num_classes).to(device)

loss_fn = torch.nn.CrossEntropyLoss()  # softmax + CEL
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, is_cnn=True)
test(model, train_dataloader, is_cnn=True)
test(model, test_dataloader, is_cnn=True)
