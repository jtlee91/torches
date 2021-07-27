import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)


class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.conv_features = ["0", "5", "10", "19", "28"]
        self.model = torchvision.models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = list()
        for idx, layer in enumerate(self.model):
            x = layer(x)  # x equals to feature of the layer

            if str(idx) in self.conv_features:
                features.append(x)
        return x

def load_image(img):
    image = Image.open(img)
    image = loader(image).unsqueeze(0)
    return image.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224

loader = torchvision.transforms.Compose([
    torchvision.transforms.Resize((image_size, image_size)),
    torchvision.transforms.ToTensor()
])

original_img = load_image("src/image.jpg")
style_img = load_image("src/style.jpg")

model = VGG().to(device).eval()
# generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated = original_img.clone().requires_grad_(True)
# generated = style_img.clone().requires_grad_(True)

total_steps = 6000
learning_rate = 3e-2
alpha = 1
beta = 0.5

optimizer = torch.optim.Adam([generated], lr=learning_rate)

loop = tqdm(range(total_steps))

for step in loop:
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0
    for gen_feature, ori_feature, style_feature in zip(generated_features, original_img_features, style_features):
        channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - ori_feature) ** 2)
        G = gen_feature.view(channel, height*width).mm(gen_feature.view(channel, height*width).t())
        A = style_feature.view(channel, height*width).mm(style_feature.view(channel, height*width).t())
        style_loss += torch.mean((G-A)**2)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 199:
        torchvision.utils.save_image(generated, f"./generated_a1b05_{step+1}.png")

    loop.set_description("Steps")
    loop.set_postfix(ori_loss=original_loss.item(), style_loss=style_loss.item(), total_loss=total_loss.item())
