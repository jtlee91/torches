import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
torch.manual_seed(100)


class AN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer):
        super(AN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=2 ** hidden_layer)
        self.fc2 = nn.Linear(in_features=2 ** hidden_layer, out_features=num_classes)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, hidden_layer, out_features=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2 ** hidden_layer, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=2 ** hidden_layer, out_channels=2 ** hidden_layer // 2, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=2 ** hidden_layer // 2, out_channels=2 ** hidden_layer // 4, kernel_size=(3, 3))
        self.mp = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=2 ** hidden_layer // 4 * 3 * 3, out_features=out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.batch_norm1 = nn.BatchNorm2d(2 ** hidden_layer)
        self.batch_norm2 = nn.BatchNorm2d(2 ** hidden_layer // 2)
        self.batch_norm3 = nn.BatchNorm2d(2 ** hidden_layer // 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ResNetMNIST(nn.Module):
    def __init__(self, hidden_layer):
        super(ResNetMNIST, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.avgpool = Identity()
        self.resnet18.fc = nn.Linear(in_features=512, out_features=10)
        # self.bn = nn.BatchNorm1d(2 ** hidden_layer)
        # self.relu = nn.LeakyReLU(negative_slope=0.1)
        # self.fc = nn.Linear(in_features=2 ** hidden_layer, out_features=10)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def train(model, data, epochs, loss_fn, optimizer):
    for epoch in range(epochs):
        loop = tqdm(data)
        for X, y in loop:
            X, y = X.to(device), y.to(device)

            num_correct = 0
            num_size = 0

            inference = model(X)
            loss = loss_fn(inference, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += (inference.max(1)[1] == y).sum()
            num_size += inference.size(0)
            accuracy = f"{round(float(num_correct / num_size * 100), 1)}%"

            loop.set_description(f"Epoch: {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item(), batch_accuracy=accuracy)


def test(model, data):
    loop = tqdm(data)
    num_correct = 0
    num_size = 0
    model.eval()
    with torch.no_grad():
        for X, y in loop:
            X, y = X.to(device), y.to(device)

            inference = model(X)

            num_correct += (inference.max(1)[1] == y).sum()
            num_size += inference.size(0)
            accuracy = f"{round(float(num_correct / num_size * 100), 1)}%"

            loop.set_postfix(accuracy=accuracy)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28 * 28
num_classes = 10
batch_size = 64
learning_rate = 3e-4
epochs = 3

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inference, ground_truth):
        ground_truth_ohe = torch.zeros(inference.shape[0], 10).to(device)
        ground_truth_ohe[range(inference.shape[0]), ground_truth] = 1
        loss = torch.sqrt(torch.sum((inference - ground_truth_ohe) ** 2, dim=1))
        loss = loss.mean()
        return loss


class CELossLN(nn.Module):
    def __init__(self):
        super(CELossLN, self).__init__()

    def forward(self, inference, ground_truth):
        ground_truth_ohe = torch.zeros(inference.shape[0], 10).to(device)
        ground_truth_ohe[range(inference.shape[0]), ground_truth] = 1
        inference_exp = torch.exp(inference)
        inference_exp_sum = torch.sum(torch.exp(inference), dim=1).unsqueeze(1)
        inference_softmax = torch.div(inference_exp, inference_exp_sum)
        loss = torch.sum(ground_truth_ohe * torch.log(inference_softmax), dim=1) * -1
        loss = loss.mean()
        return loss


class CELossLOG10(nn.Module):
    def __init__(self):
        super(CELossLOG10, self).__init__()

    def forward(self, inference, ground_truth):
        ground_truth_ohe = torch.zeros(inference.shape[0], 10).to(device)
        ground_truth_ohe[range(inference.shape[0]), ground_truth] = 1
        inference_exp = torch.exp(inference)
        inference_exp_sum = torch.sum(torch.exp(inference), dim=1).unsqueeze(1)
        inference_softmax = torch.div(inference_exp, inference_exp_sum)
        loss = torch.sum(ground_truth_ohe * torch.log10(inference_softmax), dim=1) * -1
        loss = loss.mean()
        return loss


class AngularMarginLossWrapper(nn.Module):
    def __init__(self):
        super(AngularMarginLossWrapper, self).__init__()
        in_feature = 128
        num_classes = 10
        self.feature = CNN(hidden_layer=7, out_features=in_feature).to(device)
        self.arcface = ArcFaceLoss(in_features=in_feature, out_features=num_classes)

    def forward(self, X, ground_truth):
        inference = self.feature(X)
        L = self.arcface(inference, ground_truth)
        return L


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 3. fc7 = mx.sym.FullyConnected (data = x, weight = W, no bias = True, num hidden = n)
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=False)  # fix the bias bj = 0
        self.margin = margin

    def forward(self, X, ground_truth):
        ground_truth_ohe = torch.zeros(X.shape[0], 10).to(device)
        ground_truth_ohe[range(X.shape[0]), ground_truth] = 1

        # Input: Feature Scale s, Margin Parameter m in Eq. 3, Class Number n, Ground-Truth ID gt.
        # Output: Class-wise affinity score fc7.

        # 1. x = mx.symbol.L2Normalization (x, mode = ’instance’)
        X = F.normalize(X, p=2, dim=1)

        # 2. W = mx.symbol.L2Normalization (W, mode = ’instance’)
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1).to(device)

        # 4. original target logit = mx.sym.pick (fc7, gt, axis = 1)
        # 5. theta = mx.sym.arccos (original target logit)
        # 6. marginal target logit = mx.sym.cos (theta + m)
        # 7. one hot = mx.sym.one hot (gt, depth = n, on value = 1.0, off value = 0.0)
        # 8. fc7 = fc7 + mx.sym.broadcast mul (one hot, mx.sym.expand dims (marginal target logit - original target logit, 1))
        # 9. fc7 = fc7 * s

        print(X)
        print(W)
        theta = torch.rad2deg(torch.arccos(torch.matmul(X, W)))
        theta_margin = theta + self.margin
        print(theta_margin)


        # W : (1, 2)  X : (2 X 1)  =>  (1, 1)
        # print(torch.matmul(W, torch.tensor([0.5, 0.1])))


        # loss = ...
        # return loss.mean()


X = torch.tensor([0.9999, 0.0, 0.0, 0.0]).to(device)
label = 1

model = ArcFaceLoss(in_features=4, out_features=10).to(device)



# model = AngularMarginLossWrapper().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# for epoch in range(epochs):
#     loop = tqdm(train_dataloader)
#     for X, y in loop:
#         X, y = X.to(device), y.to(device)
#
#         loss = model(X, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
#         loop.set_postfix(loss=loss.item())


# num_layer = 7
# print("Cross Entropy Loss(Log e =ln)")
# model = CNN(hidden_layer=num_layer).to(device)
# loss_fn = CELossLN()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# train(model=model, data=train_dataloader, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)
# test(model=model, data=test_dataloader)
#
# print("Cross Entropy Loss(Log 10)")
# model = CNN(hidden_layer=num_layer).to(device)
# loss_fn = CELossLOG10()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# train(model=model, data=train_dataloader, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)
# test(model=model, data=test_dataloader)
#
# print("Cross Entropy Loss Pytorch")
# model = CNN(hidden_layer=num_layer).to(device)
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# train(model=model, data=train_dataloader, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)
# test(model=model, data=test_dataloader)
#
# print("Mean Squared Error Loss")
# model = CNN(hidden_layer=num_layer).to(device)
# loss_fn = MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# train(model=model, data=train_dataloader, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)
# test(model=model, data=test_dataloader)
