import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import pytorch_lightning as pl
import warnings
warnings.simplefilter("ignore", UserWarning)


class MnistModel(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=128, bias=True)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes, bias=True)
        self.relu = nn.LeakyReLU(0.1)
        self.flat = nn.Flatten()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        X, label = batch
        loss = self.loss(self(X), label)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


input_size = 28 * 28
num_classes = 10

dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataset, valid_dataset = random_split(dataset, [55000, 5000])
train_dataloader, valid_dataloader = DataLoader(dataset=train_dataset), DataLoader(dataset=valid_dataset)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset)

model = MnistModel(input_size, num_classes)
trainer = pl.Trainer(gpus=1, limit_train_batches=512)
trainer.fit(model, train_dataloader, valid_dataloader)
