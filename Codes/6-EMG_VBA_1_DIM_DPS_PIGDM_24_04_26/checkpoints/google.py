# download_model.py
from diffusers import UNet2DModel, DDPMScheduler
import torch
import os

model_id = "google/ddpm-ema-celebahq-256"

net = UNet2DModel.from_pretrained(model_id)
scheduler = DDPMScheduler.from_pretrained(model_id)

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
torch.save({
    'model_state_dict': net.state_dict(),
    'alphas_cumprod': scheduler.alphas_cumprod,
    'config': dict(net.config),
}, os.path.join(save_dir, "ddpm_ema_celebahq_256.pt"))