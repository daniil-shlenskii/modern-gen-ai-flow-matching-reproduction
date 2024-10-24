import torch
from torch import nn
from tqdm import tqdm
from torchdiffeq import odeint

class Diffusion():
  def __init__(self, img_size=32, timesteps=1000, start=1e-4, end=0.02, device='cpu'):
    self.img_size = img_size
    self.timesteps = timesteps
    self.device = device
    # Prerequisuites
    self.betas = self.linear_beta_schedule(timesteps, start=start, end=end).to(device)
    self.alphas = 1 - self.betas
    self.alpha_hat = torch.cumprod(self.alphas, axis=0)

  def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

  def noise_images(self, x, t):
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
    noise = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

  def sample(self, model, batch_size=64, timesteps=1000):
      model.eval()
      with torch.no_grad():
          x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device)
          for i in tqdm(reversed(range(1, timesteps)), position=0):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                predicted_noise = model(x, t).sample
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
      model.train()
      x = (x.clamp(-1, 1) + 1) / 2
      x = (x * 255).type(torch.uint8)
      return x

def generate_fm(model, batch_size, timesteps=2, device='cpu'):
  model.eval()
  with torch.no_grad():
      x = torch.randn(batch_size, 3, 32, 32, device=device)

      t_span = torch.linspace(0, 1, timesteps, device=device)
      traj = odeint(
          model, x, t_span, rtol=1e-4, atol=1e-4, method="dopri5"
      )

      traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
      img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)l

  return img