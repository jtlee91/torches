import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import numpy as np
import random
import warnings
warnings.simplefilter("ignore", UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(CNN, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1)
        self.backbone.fc = nn.Identity()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.embedding_features = nn.Linear(in_features=512, out_features=self.embedding_size)
        self.angular_softmax = AngularMarginLoss(self.embedding_size, self.num_classes).to(device)
        self.class_features = nn.Linear(in_features=self.embedding_size, out_features=self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        x = self.backbone(x)
        embedding_feature = self.embedding_features(x)
        if labels is not None:
            class_feature, loss = self.angular_softmax(embedding_feature, labels)
            # class_feature = self.class_features(embedding_feature)
            # loss = self.loss_fn(class_feature, labels)
            return embedding_feature, class_feature, loss
        return embedding_feature


class AngularMarginLoss(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(AngularMarginLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.class_features = nn.Linear(in_features=self.embedding_size, out_features=self.num_classes, bias=False)
        self.eps = 1e-12
        self.m = 0.5
        self.s = 64

    def forward(self, embedding_feature, label):
        label_ohe = torch.zeros(embedding_feature.shape[0], self.num_classes).to(device)
        label_ohe[range(embedding_feature.shape[0]), label] = 1

        embedding_feature_norm = F.normalize(embedding_feature, dim=1)  # (32, 512)
        weight_feature = self.class_features.weight  # (10, 512)
        weight_feature_norm = F.normalize(weight_feature, dim=1)  # (10, 512)
        weight_feature_norm_t = torch.t(weight_feature_norm)  # (512, 10)

        cos_theta = torch.mm(embedding_feature_norm, weight_feature_norm_t)  # (32, 10)
        cos_theta = torch.clamp(cos_theta, -1 + self.eps, 1 - self.eps)  # (32, 10)
        theta = torch.acos(cos_theta)  # (32, 10)
        theta_target = theta[torch.arange(embedding_feature.shape[0]), label]  # (32)
        theta_target_margin = theta_target + self.m
        theta_target_margin = torch.clamp(theta_target_margin, self.eps, math.pi - self.eps)  # (32)
        theta[torch.arange(embedding_feature.shape[0]), label] = theta_target_margin  # (32, 10)
        cos_theta_margin = torch.cos(theta)  # (32, 10)
        cos_theta_margin = cos_theta_margin * self.s  # (32, 10)
        to_softmax = torch.div(torch.exp(cos_theta_margin), torch.sum(torch.exp(cos_theta_margin), dim=1).unsqueeze(1))
        to_softmax_log = torch.log(to_softmax)
        ce_loss = torch.sum(torch.multiply(label_ohe, to_softmax_log), dim=1)
        loss = -torch.mean(ce_loss)

        return self.class_features(embedding_feature), loss


def train(model, dataloader, epochs, optimizer):
    writer = SummaryWriter(f"logs/ArcFace")
    class_label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    for epoch in range(epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        num_correct = 0
        num_size = 0
        for batch_idx, (X, label) in loop:
            X, label = X.to(device), label.to(device)

            embedding_feature, class_feature, loss = model(X, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += (class_feature.max(1)[1] == label).sum()
            num_size += class_feature.shape[0]
            epoch_accuracy = float(num_correct / num_size)

            loop.set_description(f"Epoch: {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item(), epoch_accuracy=f"{epoch_accuracy:.3f}")

            classes = [class_label[i] for i in class_feature.max(1)[1]]
            writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=batch_idx)
            if batch_idx % 50 == 49:
                writer.add_embedding(mat=embedding_feature, metadata=classes, label_img=X, global_step=batch_idx)


def test(model, dataloader):
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    num_correct = 0
    num_size = 0
    for batch_idx, (X, label) in loop:
        X, label = X.to(device), label.to(device)

        embedding_feature, class_feature, loss = model(X, label)

        num_correct += (class_feature.max(1)[1] == label).sum()
        num_size += class_feature.shape[0]
        epoch_accuracy = float(num_correct / num_size)

        loop.set_postfix(epoch_accuracy=f"{epoch_accuracy:.3f}")


if __name__ == "__main__":
    num_classes = 10
    batch_size = 256
    learning_rate = 3e-4
    epochs = 1

    random_seed = 100
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN(embedding_size=512, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    train(model, train_dataloader, epochs, optimizer)
    test(model, test_dataloader)

    # angular_margin = AngularMarginLoss(num_classes=3).to(device)
    # class_feature = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.1, 0.9], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]).to(device)
    # print("LOSS: ", angular_margin(class_feature, [0, 2, 1, 2, 2]))
