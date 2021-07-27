import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from test18_dataset import FaceDataset
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceSoftmax(nn.Module):
    def __init__(self):
        super(FaceSoftmax, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.avgpool = Identity()
        self.resnet.fc = nn.Linear(512, 512)

        num_embedding_features = 512
        self.embedding = nn.Linear(512, num_embedding_features)
        self.fc = nn.Linear(num_embedding_features, 5749)

    def forward(self, x, label=None):
        x = self.resnet(x)
        embedding_features = self.embedding(x)

        if label is not None:
            class_features = self.fc(embedding_features)
            loss_fn = CELoss()  # nn.CrossEntropyLoss()
            loss = loss_fn(class_features, label)
            return class_features, embedding_features, loss
        return embedding_features


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, x, label):
        one_hot_encoding = torch.zeros(x.shape[0], 5749).to(device)
        one_hot_encoding[range(x.shape[0]), label] = 1
        to_softmax = torch.div(torch.exp(x), torch.sum(torch.exp(x), dim=1).unsqueeze(1))
        cross_entropy = torch.sum(torch.multiply(one_hot_encoding, torch.log(to_softmax)), dim=1)
        loss = -cross_entropy.mean()
        return torch.tensor([loss]).to(device)


batch_size = 2
learning_rate = 3e-4
epochs = 5

model = FaceSoftmax().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = FaceDataset(base_path="D:/Data/Dataset/lfw", transform=torchvision.transforms.ToTensor())
num_dataset = len(dataset)
train_set, test_set = torch.utils.data.random_split(dataset, [int(num_dataset*0.8), num_dataset - int(num_dataset*0.8)])
train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


def train(model, data_loader, epochs, optimizer):
    num_correct = 0
    num_size = 0
    writer = SummaryWriter(f"logs/face")
    class_labels = list(range(5749))

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
            if batch_idx % 200 == 0:
                classes = [class_labels[i] for i in class_features.max(1)[1]]
                writer.add_embedding(embedding_features, classes, X, global_step=step)
            step += 1


# train(model, train_dataloader, epochs, optimizer)

for X, label in train_dataloader:
    X, label = X.to(device), label.to(device)
    break

print(model(X, label)[0], model(X, label)[0].shape)
print(model(X, label)[1], model(X, label)[1].shape)
print(model(X, label)[2].item, model(X, label)[2].shape)
