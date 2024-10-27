import argparse
import yaml

import torch
import numpy as np
from src.models.forward import Diffusion, generate_fm
from src.models.unet_model import CustomUNet2DModel

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Parse command-line arguments
    args = parse_args()

    config = load_config(args.config)

    mode = config['mode']
    img_size = config['img_size']
    in_channels = config['in_channels']
    device = torch.device(config['device'])
    num_images = config['num_images']
    timesteps = config['timesteps']
    model_path = config['model_path']
    save_path = config['save_path']

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
    model.load_state_dict(torch.load(model_path, map_location=device))

    if mode == 'SM':
        diffusion = (Diffusion(img_size=img_size, timesteps=timesteps, device=device)
        images = diffusion.sample(model, num_images=num_images, device=device))
    if mode == 'FM':
        images = generate_fm(model, batch_size=num_images, timesteps=timesteps, device=device)
    if mode == 'OT':
        images = generate_fm(model, batch_size=num_images, timesteps=timesteps, device=device)

    np.save(save_path, images.cpu().numpy())

if __name__ == "__main__":
    main()
