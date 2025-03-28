import torch
torch.cuda.empty_cache()
from torch import nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

print(torch.__version__)
print(torch.cuda.is_available())

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to dataset
sun_rgbd_dir = 'D:/WPI/Resume/Preparation/Diffusion model/SUNRGBD'

# Read Dataset
def read_sun_rgbd_data(sun_rgbd_dir):
    image_dir = os.path.join(sun_rgbd_dir, 'rgb_dir')
    depth_dir = os.path.join(sun_rgbd_dir, 'depth_dir')
    captions_file = os.path.join(sun_rgbd_dir, 'captions.json')

    with open(captions_file, 'r') as f:
        captions_data = json.load(f)

    transform_rgb = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_depth = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    data = []
    for image_file in os.listdir(image_dir):
        image_id = os.path.splitext(image_file)[0]  # Remove extension
        caption = captions_data.get(image_id, "")
      
        if not caption:
            continue

        rgb_path = os.path.join(image_dir, image_file)
        depth_path = os.path.join(depth_dir, image_id + ".png")

        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path).convert('L')

        rgb_tensor = transform_rgb(rgb_image)
        depth_tensor = transform_depth(depth_image)

        data.append((rgb_tensor, depth_tensor, caption))

    return data

# Dataset Class
class SunRGBDDataset(Dataset):
    def __init__(self, data, text_encoder):
        self.data = data
        self.text_encoder = text_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_image, depth_image, caption = self.data[idx]
        text_embedding = self.text_encoder.encode(caption)
        return depth_image.to(device), text_embedding.to(device), rgb_image.to(device)

# Text Encoder using DistilBERT
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="distilbert-base-uncased"):
        super(TextEncoder, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)
        self.model = DistilBertModel.from_pretrained(pretrained_model_name)
    
    def encode(self, text_input):
        inputs = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

# UNet Model
class UNetDiffusionModel(nn.Module):
    def __init__(self, latent_dim=768, depth_channels=1, image_channels=3):
        super(UNetDiffusionModel, self).__init__()

        self.encoder1 = self.conv_block(depth_channels + latent_dim, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)

        self.decoder3 = self.deconv_block(512, 256)
        self.decoder2 = self.deconv_block(256, 128)
        self.decoder1 = self.deconv_block(128, 64)
        self.output = nn.Conv2d(64, image_channels, kernel_size=3, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, depth_map, text_embedding):
        # Remove singleton dimensions and expand properly
        text_embedding = text_embedding.squeeze(1).squeeze(1).squeeze(1)  # Shape: [4, 768]
        text_embedding = text_embedding.unsqueeze(2).unsqueeze(3)  # Shape: [4, 768, 1, 1]
        text_embedding = text_embedding.expand(-1, -1, depth_map.size(2), depth_map.size(3))  # Shape: [4, 768, 256, 256]

        x = torch.cat([depth_map, text_embedding], dim=1)

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.bottleneck(x)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)

        return torch.sigmoid(self.output(x))

# Training Function
def train_model(model, train_loader, num_epochs=5, lr=1e-4, accumulate_grad_steps=4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.to(device)
    model.train()

    from torch.cuda.amp import autocast, GradScaler

    
    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        for depth_map, text_embedding, target_image in train_loader:
            optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with autocast():
                output_image = model(depth_map, text_embedding)
                loss = criterion(output_image, target_image)
            
            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Load Data and Train
text_encoder = TextEncoder().to(device)
data = read_sun_rgbd_data(sun_rgbd_dir)
train_dataset = SunRGBDDataset(data, text_encoder)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Reduced batch size to 1

unet_model = UNetDiffusionModel().to(device)
train_model(unet_model, train_loader)

# Save Model
torch.save(unet_model.state_dict(), "diffusion_model.pth")
