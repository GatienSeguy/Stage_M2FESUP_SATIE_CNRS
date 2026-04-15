import os
import glob
import torch
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm

from .Forward import forward_process


def train(net, schedule, dataloader, epochs, lr=2e-4, ema_decay=0.9999, grad_clip=1.0, ckpt_dir='checkpoints', device='mps'):
    """
    Args :
        net        : nn.Module — réseau ε_θ(x_t, t)
        schedule   : NoiseSchedule
        dataloader : itère des batches (B, C, H, W) dans [-1, 1]
        epochs     : nombre total d'époques
    Returns :
        net, ema_net, losses
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    net = net.to(device)
    net.train()

    alphas_bar = schedule.alphas_bar.to(device)
    sigmas = schedule.sigmas.to(device)
    T = schedule.T

    # EMA
    ema_net = deepcopy(net)
    ema_net.eval()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
    losses = []
    start_epoch = 0

    # Reprendre si checkpoint existant
    existing = sorted(glob.glob(os.path.join(ckpt_dir, "epoch*.pt")))
    if existing:
        ckpt = torch.load(existing[-1], map_location=device, weights_only=True)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        losses = ckpt.get('losses', [])
        ema_net.load_state_dict(ckpt.get('ema_state_dict', ckpt['model_state_dict']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr / 100, last_epoch=start_epoch - 1
        )
        print(f"Reprise depuis epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:03d}/{epochs}")

        for batch in pbar:
            x0 = batch.to(device)
            B = x0.shape[0]

            t = torch.randint(0, T, (B,), device=device)
            x_t, epsilon = forward_process(x0, t, alphas_bar, sigmas)
            eps_pred = net(x_t, t)

            loss = F.mse_loss(eps_pred, epsilon)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            # EMA
            with torch.no_grad():
                for ema_p, p in zip(ema_net.parameters(), net.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        torch.save({
            'model_state_dict': net.state_dict(),
            'ema_state_dict': ema_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'losses': losses,
        }, os.path.join(ckpt_dir, f"epoch{epoch+1:03d}.pt"))

    return net, ema_net, losses