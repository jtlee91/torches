import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
torch.manual_seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptchaRecognizer(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaRecognizer, self).__init__()
        self.feature_extractor = self._create_conv_layer()
        self.fc = nn.Linear(in_features=32 * 25, out_features=256)
        self.sequence = nn.LSTM(input_size=256, hidden_size=256, bidirectional=True, num_layers=2, batch_first=True)
        self.output = nn.Linear(in_features=512, out_features=num_chars + 1)

    def forward(self, x, label=None):
        print(x.shape)
        x = self.feature_extractor(x)
        print(x.shape)
        x = x.permute(0, 3, 1, 2)
        print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        x, _ = self.sequence(x)
        print(x.shape)
        x = self.output(x)
        print(x.shape)
        x = x.permute(1, 0, 2)
        print(x.shape)
        return ...

    def _create_conv_layer(self):
        layers = list()
        architecture = [(1, 64), "M", (64, 32), "M", (32, 16)]
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    nn.Conv2d(in_channels=x[0], out_channels=x[1], kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(x[1]),
                    nn.LeakyReLU(0.1)
                ]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2))]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = CaptchaRecognizer(num_chars=27).to(device)
    X = torch.randn(1, 1, 200, 64).to(device)  # Batch_size, Channel, Width, Height
    print(model(X))
