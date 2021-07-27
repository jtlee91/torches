import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.simplefilter("ignore", UserWarning)


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features=z_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=img_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.tanh(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

optimizer_disc = optim.Adam(disc.parameters(), lr=learning_rate)
optimizer_gen = optim.Adam(gen.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train Discriminator: max log(D(real)) + log(1 - D(G(z))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_disc.step()

        # Train Generator min log(1 - D(G(z))) <--> max log(D(G(z))
        output = disc(fake).view(-1)
        lossG = loss_fn(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist real Images", img_grid_real, global_step=step
                )

                step += 1
