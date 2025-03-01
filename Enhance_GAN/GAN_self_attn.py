import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.nn.utils import spectral_norm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Custom Dataset class for CelebA
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.lower().endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Define transformations (resize, crop, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize(64),  # Resize images to 64x64
    transforms.CenterCrop(64),  # Crop center to 64x64
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load CelebA dataset
dataset_path = 'D:/WPI/Resume/Preparation/GANs/img_align_celeba/img_align_celeba'  # Adjust the path
dataset = CelebADataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Check how many images are loaded
print(f"Total number of images loaded: {len(dataset)}")

# Ensure saved_images directory exists
os.makedirs("saved_images", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# # Self-Attention layer for both Generator and Discriminator
# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         batch_size, C, width, height = x.size()
#         query = self.query_conv(x).view(batch_size, -1, width * height)
#         key = self.key_conv(x).view(batch_size, -1, width * height)
#         value = self.value_conv(x).view(batch_size, -1, width * height)

#         attention = torch.bmm(query.transpose(1, 2), key)
#         attention = torch.softmax(attention, dim=-1)

#         out = torch.bmm(value, attention.transpose(1, 2))
#         out = out.view(batch_size, C, width, height)

#         return self.gamma * out + x

# Define Generator with Progressive Growing
# class Generator(nn.Module):
#     def __init__(self, z_dim=100, img_channels=3, feature_maps=64, start_res=4):
#         super(Generator, self).__init__()

#         self.start_res = start_res
#         self.z_dim = z_dim
#         self.img_channels = img_channels
#         self.feature_maps = feature_maps

#         self.initial_layers = nn.Sequential(
#             spectral_norm(nn.ConvTranspose2d(z_dim, feature_maps * 8, 4, 1, 0, bias=False)),
#             nn.BatchNorm2d(feature_maps * 8),
#             nn.ReLU(True)
#         )

#         self.middle_layers = nn.Sequential(
#             spectral_norm(nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False)),
#             nn.BatchNorm2d(feature_maps * 4),
#             nn.ReLU(True),
#             SelfAttention(feature_maps * 4),  # Adding Self-Attention
#         )

#         self.final_layers = nn.Sequential(
#             spectral_norm(nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False)),
#             nn.BatchNorm2d(feature_maps * 2),
#             nn.ReLU(True),
#             spectral_norm(nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False)),
#             nn.Tanh()
#         )

#     def forward(self, z, grow_res=False):
#         if grow_res:
#             self.middle_layers[1] = nn.BatchNorm2d(self.feature_maps * 4)
#             self.final_layers[1] = nn.BatchNorm2d(self.feature_maps * 2)
#         x = self.initial_layers(z)
#         x = self.middle_layers(x)
#         return self.final_layers(x)


class Generator(nn.Module):
    def __init__(self, z_dim, ngf):
        super(Generator, self).__init__()

        # Define the progressive layers, starting from the lowest resolution (e.g., 4x4)
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, kernel_size=4, stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 8)
        )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 4)
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 2)
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf)
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Add self-attention after block 3 (or wherever you prefer)
        self.attn = SelfAttention(ngf * 2)  # SelfAttention should be defined elsewhere

    def forward(self, z):
        x = self.initial(z)
        x = self.block1(x)
        x = self.block2(x)
        
        # Apply self-attention
        x = self.attn(x)

        x = self.block3(x)
        x = self.block4(x)
        return x

# Self-attention module definition (just an example)
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query(x).view(batch_size, -1, height * width)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return self.gamma * out + x


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 4)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 8)
        )

        self.attn = SelfAttention(ndf * 4)  # Apply self-attention at this block

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.block1(img)
        x = self.block2(x)
        x = self.block3(x)

        # Apply self-attention
        x = self.attn(x)

        x = self.block4(x)
        x = x.view(x.size(0), -1)  # Flatten
        out = self.fc(x)
        return out


# Initialize models and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
generator = Generator(z_dim=z_dim, ngf=64).to(device)
discriminator = Discriminator(ndf=64).to(device)


# Loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
def train(generator, discriminator, dataloader, epochs=20):
    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            valid = torch.full((batch_size, 1), 0.9, device=device)  # Label smoothing
            fake = torch.full((batch_size, 1), 0.1, device=device)

            # Train Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, z_dim, 1, 1, device=device)  # Latent noise input
            fake_imgs = generator(z)

            # Ensure the output of discriminator is flattened to [batch_size, 1]
            real_output = discriminator(real_imgs).view(batch_size, -1)
            fake_output = discriminator(fake_imgs.detach()).view(batch_size, -1)

            # Calculate loss
            real_loss = adversarial_loss(real_output, valid)
            fake_loss = adversarial_loss(fake_output, fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(fake_imgs).view(batch_size, -1), valid)
            g_loss.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save generated images and model weights after every epoch
        save_generated_images(generator, epoch, device)
        torch.save(generator.state_dict(), f"weights/generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"weights/discriminator_epoch_{epoch}.pth")

# Function to save generated images
def save_generated_images(generator, epoch, device, num_images=16):
    z = torch.randn(num_images, z_dim, 1, 1, device=device)
    generated_imgs = generator(z).detach().cpu()
    grid = vutils.make_grid(generated_imgs, nrow=4, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title(f"Epoch {epoch}")
    plt.axis('off')
    plt.savefig(f"saved_images/generated_epoch_{epoch}.png")
    plt.close()
    # Function to save generated images
def save_generated_images(generator, epoch, device, num_images=64):
    z = torch.randn(num_images, z_dim, 1, 1, device=device)  # Latent noise for generation
    gen_imgs = generator(z).detach().cpu()


    # Rescale images to [0, 1] for visualization
    gen_imgs = gen_imgs / 2 + 0.5  # Rescale to [0, 1]

    # Save the images
    vutils.save_image(gen_imgs.data, f"saved_images/generated_{epoch}.png", normalize=True, nrow=8)

    print(f"Generated images saved for epoch {epoch}.")


# Start training
print("Starting Training...")
train(generator, discriminator, dataloader, epochs=20)
print("Training Complete. Models saved in 'weights/' and images in 'saved_images/'")
