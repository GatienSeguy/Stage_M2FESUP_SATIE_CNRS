import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from degrade import load_and_resize
from diffusion.Schedules import DDPMSchedule
from diffusion.Reverse import sample_conditional
from operators_torch import GaussianBlurOperator


# =====================================================================
# HYPERPARAMÈTRES — modifier uniquement cette section
# =====================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# Image & dégradation 
IMAGE_DIR  = os.path.join(BASE_DIR, 'observations/chat.jpg')
SIGMA_BLUR = 2.0
SIGMA_NOISE = SIGMA_BLUR / 100

# Modèle
IMG_SIZE   = 256
IN_CH      = 3
MODEL_NAME = 'ddpm_ema_celebahq_256'  

# Schedule (utilisé uniquement pour les modèles perso sans alphas_cumprod)
T          = 50
BETA_START = 1e-4
BETA_END   = 0.02

# Chemins
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultats/chat')
CKPT_DIR   = os.path.join(BASE_DIR, 'checkpoints')
DEVICE     = 'mps'

# EMG-VBA
EMG_N_ITER     = 100
EMG_SKIP_AFTER = 0
A_0 = 1e-3
B_0 = 1e-3
C_0 = 1e-3
D_0 = 1e-3

N_SAMPLES     = 1
MONITOR_STEPS = [999, 900, 800, 700, 600, 500, 400, 300, 200, 100]

# Opérateur
OPERATOR = GaussianBlurOperator(kernel_size=9, sigma=SIGMA_BLUR,
                                img_size=IMG_SIZE, n_channels=IN_CH)
# =====================================================================


def degrade_image(image_path, op, img_size, n_channels, sigma_noise, output_dir):
    """Crée l'observation y = A x₀ + σ_b ε."""
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    x0 = load_and_resize(image_path, img_size, n_channels=n_channels)
    x0_11 = 2.0 * x0 - 1.0
    y_11 = op.create_observation(x0_11, sigma_noise)
    
    if isinstance(y_11, torch.Tensor):
        y_11 = y_11.cpu().numpy()
    if n_channels == 1:
        y_11 = y_11.reshape(img_size, img_size)

    np.save(os.path.join(output_dir, f"{name}_clean.npy"), x0_11)
    np.save(os.path.join(output_dir, f"{name}_degraded.npy"), y_11)

    y_01 = (y_11 + 1.0) / 2.0
    if n_channels == 1:
        Image.fromarray((y_01 * 255).clip(0, 255).astype(np.uint8), mode='L') \
             .save(os.path.join(output_dir, f"{name}_degraded.png"))
    else:
        Image.fromarray((y_01.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), mode='RGB') \
             .save(os.path.join(output_dir, f"{name}_degraded.png"))

    print(f"[Dégradation] {img_size}x{img_size} x{n_channels}, opérateur : {type(op).__name__}")
    return name, x0_11, y_11


def load_net(ckpt_dir, model_name, device, T=1000, beta_start=1e-4, beta_end=0.02):
    """Charge un checkpoint .pt — détecte automatiquement le type de modèle.

    Formats supportés :
      1) Modèle diffusers (google) : clés 'config' + 'model_state_dict' + 'alphas_cumprod'
      2) Modèle perso (UNet)       : clés 'model_state_dict' ou 'ema_state_dict'
                                     + optionnel 'hyperparams'
    """
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # --- Détection du type de modèle ---
    is_diffusers = 'config' in ckpt and 'alphas_cumprod' in ckpt

    if is_diffusers:
        # Modèle diffusers (ex: google/ddpm-ema-celebahq-256)
        from diffusers import UNet2DModel
        net = UNet2DModel.from_config(ckpt['config']).to(device)
        net.load_state_dict(ckpt['model_state_dict'])

        alphas_cumprod = ckpt['alphas_cumprod']
        if not isinstance(alphas_cumprod, torch.Tensor):
            alphas_cumprod = torch.as_tensor(alphas_cumprod)
        schedule = DDPMSchedule.from_alphas_cumprod(alphas_cumprod.float())

        print(f"[Modèle diffusers] {ckpt_path}  "
              f"(T={schedule.T}, in_ch={net.config.in_channels}, "
              f"size={net.config.sample_size})")
    else:
        # Modèle perso (UNet maison)
        from model import UNet
        hp = ckpt.get('hyperparams', {})
        net = UNet(
            in_ch=hp.get('in_ch', 1),
            base_ch=hp.get('base_ch', 128),
            time_dim=hp.get('time_dim', 256),
        ).to(device)

        state = ckpt.get('ema_state_dict', ckpt['model_state_dict'])
        net.load_state_dict(state)

        # Schedule : depuis le checkpoint ou depuis les hyperparamètres du main
        if 'alphas_cumprod' in ckpt:
            alphas_cumprod = ckpt['alphas_cumprod']
            if not isinstance(alphas_cumprod, torch.Tensor):
                alphas_cumprod = torch.as_tensor(alphas_cumprod)
            schedule = DDPMSchedule.from_alphas_cumprod(alphas_cumprod.float())
        else:
            schedule = DDPMSchedule(T=T, beta_start=beta_start, beta_end=beta_end)

        print(f"[Modèle perso] {ckpt_path}  (T={schedule.T})")

    net.eval()
    return net, schedule


def to_hwc(a):
    """(C,H,W) → (H,W,C) si RGB, (H,W) si mono."""
    if a.ndim == 3 and a.shape[0] == 3:
        return a.transpose(1, 2, 0)
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    return a


def main():
    device = DEVICE
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    op = OPERATOR

    # 1. Dégradation
    name, x0_11, y_11 = degrade_image(IMAGE_DIR, op, IMG_SIZE, IN_CH, SIGMA_NOISE, OUTPUT_DIR)

    # 2. Précalculs opérateur
    y_flat   = op._as_tensor(y_11.ravel())
    Aty      = op.adjoint(y_flat)
    AtA_diag = op.compute_AtA_diag()
    print(f"[Opérateur] {type(op).__name__}  —  n={op.input_dim()}, m={op.output_dim()}")

    # 3. Modèle + schedule
    net, schedule = load_net(CKPT_DIR, MODEL_NAME, device, T=T, beta_start=BETA_START, beta_end=BETA_END)

    # 4. Reverse conditionnel EMG-VBA
    shape = (IN_CH, IMG_SIZE, IMG_SIZE)
    x_rec, diagnostics = sample_conditional(
        net, schedule, y_flat, op, shape,
        n_samples=N_SAMPLES, device=device,
        emg_n_iter=EMG_N_ITER, emg_skip_after=EMG_SKIP_AFTER,
        a_0=A_0, b_0=B_0, c_0=C_0, d_0=D_0,
        Aty=Aty, AtA_diag=AtA_diag, monitor_steps=MONITOR_STEPS,
    )

    # 5. Sauvegarde reconstruction
    x_rec_np = x_rec[0].cpu().numpy()   # (C, H, W), [0, 1]
    rec_path = os.path.join(OUTPUT_DIR, f"{name}_reconstructed_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png")
    if IN_CH == 1:
        Image.fromarray((x_rec_np[0] * 255).clip(0, 255).astype(np.uint8), mode='L').save(rec_path)
    else:
        Image.fromarray((x_rec_np.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), mode='RGB').save(rec_path)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_reconstructed_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.npy"), x_rec_np)
    print(f"[Reconstruction] Sauvegardée : {rec_path}")

    # 6. Affichage comparaison
    x0_img  = to_hwc((x0_11 + 1) / 2)
    y_img   = to_hwc((y_11  + 1) / 2)
    rec_img = to_hwc(x_rec_np)
    imkw = {} if IN_CH == 3 else {'cmap': 'gray'}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes,
                              [x0_img, y_img, rec_img],
                              ["Originale", f"Dégradée (σ_blur={SIGMA_BLUR})", "Reconstruite (EMG-VBA)"]):
        ax.imshow(np.clip(img, 0, 1), vmin=0, vmax=1, **imkw)
        ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_comparison_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=120)
    plt.show()

    # 7. Énergie libre F(q) par pas monitored
    energie = diagnostics['energie_par_step']
    if energie:
        fig, axes = plt.subplots(2, 5, figsize=(20, 6))
        for idx, t_val in enumerate(sorted(energie.keys(), reverse=True)):
            ax = axes[idx // 5][idx % 5]
            data = energie[t_val]
            iters = range(1, len(data['energie_libre']) + 1)
            ax.plot(iters, data['energie_libre'], 'b-o', markersize=2, label='F(q)')
            ax.set_title(f"t = {t_val}")
            ax.set_xlabel("Itération k")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)
        fig.suptitle("F(q) doit croître à t fixé (Résultat XII.2.1)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_energie_emgvba_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=150)
        plt.show()

    # 8. Décomposition F = log_jointe - entropie
    if energie:
        fig, axes = plt.subplots(2, 5, figsize=(22, 8))
        for idx, t_val in enumerate(sorted(energie.keys(), reverse=True)):
            ax = axes[idx // 5][idx % 5]
            data = energie[t_val]
            iters = range(1, len(data['energie_libre']) + 1)
            ax.plot(iters, data['energie_libre'], 'b-o', markersize=2, label='F(q)')
            ax.plot(iters, data['log_jointe'], 'g--', linewidth=1, label='log jointe')
            ax.plot(iters, [-e for e in data['entropie']], 'r--', linewidth=1, label='-E[log q]')
            ax.set_title(f"t = {t_val}")
            ax.set_xlabel("Itération k")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)
        fig.suptitle("Décomposition F(q) = E[log p] - E[log q]")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_decomposition_F_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=150)
        plt.show()

    # 9. Variances σ_b² et σ_r²
    tau_b_final = diagnostics['tau_b_final']
    tau_r_final = diagnostics['tau_r_final']
    if tau_b_final:
        ts = sorted(tau_b_final.keys(), reverse=True)
        sigma2_b = np.array([1.0 / tau_b_final[t] for t in ts])
        sigma2_r = np.array([1.0 / tau_r_final[t] for t in ts])

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(ts, sigma2_b, '-', label=r'$\sigma_b^2$ (bruit mesure)', color='tab:blue')
        ax.plot(ts, sigma2_r, '-', label=r'$\sigma_r^2$ (a priori)',     color='tab:red')
        ax.set_xlabel("t (pas de diffusion)")
        ax.set_ylabel("Variance")
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(r"Évolution de $\sigma_b^2$ et $\sigma_r^2$ au cours du reverse diffusion")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_variances_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=150)
        plt.show()


if __name__ == "__main__":
    main()