import torch
from torch.nn import Linear, Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)


# Define Model
class AN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(AN, self).__init__()
        self.fc1 = Linear(in_features=input_size, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3))
        self.maxpool = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3))
        self.fc = Linear(in_features=32 * 5 * 5, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


# Training
def train(model, dataloader, loss_fn, optimizer, epochs, device, is_cnn=False):
    step = 0
    writer = SummaryWriter(f"logs/MNIST/tensorboard")
    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (X, y) in loop:
            X, y = X.to(device), y.to(device)  # Set GPU/CPU to use
            # Flatten X shape from (batch_size, 1, 28, 28) to (batch_size, 1*28*28)
            # -1 means adjust the rest dimension automatically
            if not is_cnn:
                X = X.reshape(X.shape[0], -1)

            infer = model(X)  # inference value of train data
            loss = loss_fn(infer, y)  # Get the loss by comparing inference value and ground truths

            optimizer.zero_grad()  # Initialize gradient on every epochs
            loss.backward()  # back propagation to minimize the loss
            optimizer.step()  # Apply the back propagation result to optimizer

            num_correct = (infer.max(1)[1] == y).sum()
            training_acc = float(num_correct) / float(X.shape[0])

            writer.add_scalar("Training Loss ", loss, global_step=step)
            writer.add_scalar("Training Accuracy ", training_acc, global_step=step)
            step += 1

            loop.set_description(f"Epochs: {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item())


# Testing
def test(model, dataloader, device, is_cnn=False):
    model.eval()
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        loop = tqdm(dataloader, total=len(dataloader))
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            if not is_cnn:
                X = X.reshape(X.shape[0], -1)

            infer = model(X)
            num_correct += (infer.max(1)[1] == y).sum()
            num_total += infer.size(0)  # index 0 means batch_size and 1 means num of classes
            loop.set_description("Testing Accuray")
            loop.set_postfix(accuray=f"{round(float(num_correct / num_total * 100), 2)}%")
    # print(f"{num_correct} / {num_total} Accuracy: {round(float(num_correct / num_total * 100), 2)}%")


# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} System")

# Hyper parameters
input_size = 28 * 28
num_classes = 10
batch_size = 64
learning_rate = 3e-4  # for ADAM
epochs = 10

# Define Dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=transform, download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# # Initialize Model
# model = AN(input_size=input_size, num_classes=num_classes).to(device)
#
# # loss function & optimizer
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, device=device)
# test(model, train_dataloader, device=device)
# test(model, test_dataloader, device=device)
#
# # Initialize Model
# model = CNN(num_classes=num_classes).to(device)
#
# # loss function & optimizer
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, device=device, is_cnn=True)
# test(model, train_dataloader, device=device, is_cnn=True)
# test(model, test_dataloader, device=device, is_cnn=True)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResnetMNIST(torch.nn.Module):
    def __init__(self):
        super(ResnetMNIST, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3))
        self.resnet = resnet.resnet152(pretrained=True)
        self.resnet.avgpool = Identity()
        self.resnet.fc = Linear(2048, 512)
        self.fc = Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet(x)
        x = self.fc(x)
        return x


# Initialize Model
model = ResnetMNIST().to(device)

# loss function & optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, device=device, is_cnn=True)
test(model, train_dataloader, device=device, is_cnn=True)
test(model, test_dataloader, device=device, is_cnn=True)
