import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)


class AN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28 * 28
num_classes = 10
batch_size = 128
learning_rate = 3e-4
epochs = 5

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


def train(model, dataloader, epochs, loss_fn, optimizer):
    for epoch in range(epochs):
        loop = tqdm(dataloader)
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            learning_correct = 0

            inference = model(X)
            loss = loss_fn(inference, y)

            learning_correct += (inference.max(1)[1] == y).sum()
            batch_acc = round(float(learning_correct / inference.size(0) * 100), 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch: {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item(), batch_acc=batch_acc)


def test(model, dataloader):
    loop = tqdm(dataloader)
    num_correct = 0
    num_size = 0
    for X, y in loop:
        X, y = X.to(device), y.to(device)

        inference = model(X)
        num_correct += (inference.max(1)[1] == y).sum()
        num_size += inference.size(0)
        acc = round(float(num_correct / num_size * 100), 2)
        loop.set_postfix(acc=acc)


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, inference, ground_truth):
        ground_truth_ohe = torch.zeros((inference.shape[0], num_classes)).to(device)
        ground_truth_ohe[range(inference.shape[0]), ground_truth] = 1
        return torch.sqrt(((inference - ground_truth_ohe) ** 2).sum()) / inference.shape[0]


class SOFTMAX(nn.Module):
    def __init__(self):
        super(SOFTMAX, self).__init__()

    def forward(self, inference, ground_truth):
        inference_softmax = torch.exp(inference) / torch.exp(inference).sum()
        ground_truth_ohe = torch.zeros((inference.shape[0], num_classes)).to(device)
        ground_truth_ohe[range(inference.shape[0]), ground_truth] = 1
        # return torch.sqrt(((inference_softmax - ground_truth_ohe) ** 2).sum()) / inference.shape[0]


# model = AN(input_size=input_size, num_classes=num_classes).to(device)
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
#
# train(model, train_dataloader, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)
# test(model, test_dataloader)
#
# model = AN(input_size=input_size, num_classes=num_classes).to(device)
# loss_fn = MSE().to(device)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
#
# train(model, train_dataloader, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)
# test(model, test_dataloader)

model = AN(input_size=input_size, num_classes=num_classes).to(device)
loss_fn = SOFTMAX().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

train(model, train_dataloader, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)
test(model, test_dataloader)