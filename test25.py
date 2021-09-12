import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models._utils as _utils
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)

manual_seed = 100
torch.manual_seed(manual_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(self._conv_block(in_channel=1, out_channel=128), nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(self._conv_block(in_channel=128, out_channel=64), nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(self._conv_block(in_channel=64, out_channel=32), nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def _conv_block(self, in_channel, out_channel):
        layers = list()
        layers += [
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.LeakyReLU(negative_slope=0.1)
        ]
        return nn.Sequential(*layers)


X = torch.randn(8, 1, 28, 28).to(device)
model = CNN().to(device)
# print(model)
# print(model(X).shape)

return_layers = {"conv1": "conv1", "conv2": "conv2", "conv3": "conv3"}
mid_features = _utils.IntermediateLayerGetter(model, return_layers)
