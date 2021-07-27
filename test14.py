import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
torch.manual_seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetMNIST(nn.Module):
    def __init__(self):
        super(ResNetMNIST, self).__init__()
        # in_feature: 3 out_feature: 512
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), padding=(2, 2))
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(in_features=512, out_features=512)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        return x


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        self.in_embedding_features = ResNetMNIST()
        self.fc = nn.Linear(in_features=512, out_features=10)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, input_x_data, label, embed=False):
        embedding_features = self.in_embedding_features(input_x_data)
        if embed:
            return embedding_features
        class_10_features = self.fc(embedding_features)
        loss = self.ce_loss(class_10_features, label)
        return loss, class_10_features, embedding_features


def train(model, data, epochs, optimizer):
    writer = SummaryWriter(log_dir=f"logs/mnist")
    class_label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    step = 1
    for epoch in range(epochs):
        loop = tqdm(enumerate(data), total=len(data))
        for batch_idx, (X, y) in loop:
            X, y = X.to(device), y.to(device)

            loss, class_10_features, embedding_features = model(X, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch : {epoch + 1} / {epochs}")
            loop.set_postfix(loss=loss.item())
            classes = [class_label[i] for i in class_10_features.max(1)[1]]

            writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=step)
            if batch_idx % 100 == 99:
                writer.add_embedding(mat=embedding_features, metadata=classes, label_img=X, global_step=step)
            step += 1


batch_size = 256
learning_rate = 3e-4
epochs = 10

model = Trainer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

train(model, train_dataloader, epochs, optimizer)
