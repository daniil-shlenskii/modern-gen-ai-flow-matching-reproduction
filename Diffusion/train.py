import argparse
import yaml

import torch
import torch.optim as optim
import torch.nn.functional as F

from forward import Diffusion
from unet_model import CustomUNet2DModel
from cfm import OptimalTransportConditionalFlowMatcher, VariancePreservingConditionalFlowMatcher

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_epoch(model, mode, matching, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        if len(batch) == 2: images, _ = batch
        else: images = batch
        images = images.to(device)

        if mode == 'SM':
            t = torch.randint(1, matching.timesteps, size=(images.size(0),)).to(device)
            x_t, noise = matching.noise_images(images, t)
            predicted_noise = model(x_t, t).sample
            loss = criterion(noise, predicted_noise)
        elif mode in ['FM', 'OT']:
            t, x_t, ut = matching.sample_location_and_conditional_flow(torch.randn_like(images).to(device), images)
            predicted_flow = model(t, x_t)
            loss = criterion(ut, predicted_flow)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration from the file
    config = load_config(args.config)
    # Access parameters
    img_size = config['img_size']
    in_channels = config['in_channels']
    dataloader = config['dataloader']
    mode = config['mode']
    timesteps = config['timesteps']
    device = torch.device(config['device'])
    num_epochs = config['num_epochs']
    lr = config['lr']
    save_model_path = config['save_path']

    if mode == 'FM':
        fp = VariancePreservingConditionalFlowMatcher()
    elif mode == 'OT':
        fp = OptimalTransportConditionalFlowMatcher()
    elif mode == 'SM':
        fp = Diffusion(img_size=img_size, timesteps=timesteps, device=device)

    model = CustomUNet2DModel(
        sample_size=img_size,
        in_channels=in_channels,
        out_channels=in_channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        dropout=0.1,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = F.mse_loss

    print("Training started...")
    for epoch in range(num_epochs):
        train_epoch(model, mode, fp, dataloader, optimizer, criterion, device)

    torch.save(model.state_dict(), save_model_path)
    
if __name__ == "__main__":
    main()