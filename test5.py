import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, MaxPool2d
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # N X time_seq X features
        self.fc = nn.Linear(hidden_size*seq_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward Prop
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # N X time_seq X features
        self.fc = nn.Linear(hidden_size*seq_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward Prop
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # N X time_seq X features
        self.fc = nn.Linear(hidden_size*seq_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward Prop
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28
seq_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 1e-3
batch_size = 64
epochs = 10

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


def train(model, dataloader, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        loop = tqdm(dataloader, total=len(dataloader))
        for X, y in loop:
            X, y = X.to(device).squeeze(1), y.to(device)

            inference = model(X)
            loss = loss_fn(inference, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"epoch : {epoch + 1} / {epochs}")
            loop.set_postfix(Loss=loss.item())


def test(model, dataloader):
    num_correct = 0
    num_total = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in loop:
            X, y = X.to(device).squeeze(1), y.to(device)

            inference = model(X)

            num_correct += (inference.max(1)[1] == y).sum()
            num_total += inference.size(0)
            accuracy = round(float(num_correct / num_total * 100), 2)

            loop.set_description("testing accuracy")
            loop.set_postfix(accuracy=f"{accuracy}%")


print("RNN")
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
X = torch.randn(64, 1, 28, 28).to(device)
print(model(X).shape)
# loss_fn = torch.nn.CrossEntropyLoss()  # softmax + CEL
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs)
# test(model, train_dataloader)
# test(model, test_dataloader)
#
#
# print("GRU")
# model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)
#
# loss_fn = torch.nn.CrossEntropyLoss()  # softmax + CEL
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs)
# test(model, train_dataloader)
# test(model, test_dataloader)
#
# print("LSTM")
# model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
#
# loss_fn = torch.nn.CrossEntropyLoss()  # softmax + CEL
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs)
# test(model, train_dataloader)
# test(model, test_dataloader)