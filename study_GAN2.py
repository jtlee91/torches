import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
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
z_dim = 128
image_dim = 28 * 28 * 1
batch_size = 64
epochs = 50

discriminator = Discriminator(input_size=image_dim).to(device)
generator = Generator(z_dim=z_dim, img_dim=image_dim).to(device)
fixed_noise = torch.randn(batch_size, z_dim).to(device)

train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor())
train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

loss_fn = torch.nn.BCELoss()  # Binary Cross Entropy
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate)

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

step = 0

for epoch in range(epochs):
    loop = tqdm(enumerate(train_dataset_loader))
    for batch_idx, (real, y) in loop:
        real = real.view(-1, 784).to(device)

        noise = torch.randn(batch_size, z_dim).to(device)
        fake = generator(noise)
        disc_real = discriminator(real).view(-1)
        lossD_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake).view(-1)
        lossD_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        discriminator.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_discriminator.step()

        output = discriminator(fake).view(-1)
        lossG = loss_fn(output, torch.ones_like(output))
        generator.zero_grad()
        lossG.backward()
        optimizer_generator.step()

        with torch.no_grad():
            fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
            writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)

        loop.set_description(f"epochs: {epoch+1} / {epochs}")
        loop.set_postfix(Loss_D=f"{lossD:.4f}", Loss_G=f"{lossG:.4f}")

        step += 1
