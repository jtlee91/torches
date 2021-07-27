import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, LSTM, MaxPool2d
import torchvision
from tqdm import tqdm
import os
import cv2
from glob import glob
import warnings
warnings.simplefilter("ignore", UserWarning)


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.num_each_image = 250
        self.transform = transform
        mask_path = "D:/Data/Dataset/archive/with_mask"
        no_mask_path = "D:/Data/Dataset/archive/without_mask"
        mask_image_path = [(i, 1) for i in glob(os.path.join(mask_path, "*.png"))[:self.num_each_image]]
        no_mask_image_path = [(i, 0) for i in glob(os.path.join(no_mask_path, "*.png"))[:self.num_each_image]]
        self.image_path = mask_image_path + no_mask_image_path

    def __len__(self):
        return self.num_each_image * 2

    def __getitem__(self, index):
        img, lable = self.image_path[index]
        img = cv2.imread(img, 1)
        img = cv2.resize(img, dsize=(224, 224))
        if self.transform:
            img = self.transform(img)
        return img, lable


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))
        self.conv2 = Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3))
        self.conv3 = Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3))
        self.mp = MaxPool2d(kernel_size=(2, 2))
        self.fc = Linear(in_features=4*26*26, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp(x)
        x = F.relu(self.conv2(x))
        x = self.mp(x)
        x = F.relu(self.conv3(x))
        x = self.mp(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


batch_size = 32
epochs = 10
learning_rate = 3e-4

dataset = MaskDataset(transform=torchvision.transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [400, 100])
train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    loop = tqdm(train_dataloader)
    for X, lable in loop:
        X, lable = X.to(device), lable.to(device)

        inference = model(X)
        loss = loss_fn(inference, lable)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epochs : {epoch + 1} / {epochs}")
        loop.set_postfix(Loss=loss.item())

num_correct = 0
num_total = 0
loop = tqdm(test_dataloader)
model.eval()
with torch.no_grad():
    for X, lable in loop:
        X, lable = X.to(device), lable.to(device)

        inference = model(X)
        num_correct += (inference.max(1)[1] == lable).sum()
        num_total += inference.size(0)

        accuracy = round(float(num_correct/num_total*100), 2)
        loop.set_description("Testing Accuracy")
        loop.set_postfix(Accuracy=accuracy)
