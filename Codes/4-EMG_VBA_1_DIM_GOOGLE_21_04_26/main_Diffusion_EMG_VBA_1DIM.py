import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import UNet2DModel

from degrade import load_and_resize, blur
from diffusion.Schedules import DDPMSchedule
from diffusion.Reverse import sample_conditional
from operators import GaussianBlurOperator


# ==================
# HYPERPARAMÈTRES
# ===================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'observations/tete.png')
SIGMA_BLUR = 2.0
SIGMA_NOISE = SIGMA_BLUR /100

IMG_SIZE = 256
IN_CH = 3

MODEL_NAME = 'ddpm_ema_celebahq_256'  # fichier checkpoints/<MODEL_NAME>.pt
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultats/tete')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
DEVICE = 'mps'

# EMG-VBA
EMG_N_ITER = 100
EMG_SKIP_AFTER = 5
A_0 = 1e-3
B_0 = 1e-3
C_0 = 1e-3
D_0 = 1e-3

N_SAMPLES = 1

MONITOR_STEPS = [999, 900, 800, 700, 600, 500, 400, 300, 200, 100]

OPERATOR = GaussianBlurOperator(kernel_size=9, sigma=SIGMA_BLUR,
                                img_size=IMG_SIZE, n_channels=IN_CH)
# ==============


def degrade_image(image_path, op, img_size, n_channels, output_dir):
    """Crée l'observation y = A x₀ + sigma_b epsilon."""
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    x0 = load_and_resize(image_path, img_size, n_channels=n_channels)

    # Passage en [-1, 1] — x0 a shape (H,W) si 1 canal, (3,H,W) si RGB
    x0_11 = 2.0 * x0 - 1.0

    # Observation via l'opérateur ; l'opérateur gère ndim==3 (multi-canal)
    y_11 = op.create_observation(x0_11, SIGMA_NOISE)
    if n_channels == 1:
        y_11 = y_11.reshape(img_size, img_size)

    np.save(os.path.join(output_dir, f"{name}_clean.npy"), x0_11)
    np.save(os.path.join(output_dir, f"{name}_degraded.npy"), y_11)

    y_01 = (y_11 + 1.0) / 2.0
    if n_channels == 1:
        img_arr = (y_01 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_arr, mode='L').save(
            os.path.join(output_dir, f"{name}_degraded.png"))
    else:
        img_arr = (y_01.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_arr, mode='RGB').save(
            os.path.join(output_dir, f"{name}_degraded.png"))

    print(f"[Dégradation] {img_size}x{img_size} x{n_channels}, opérateur : {type(op).__name__}")

    return(name, x0_11, y_11)

def load_pretrained_net(ckpt_dir, model_name, device):
    """Charge un checkpoint local (.pt) contenant model_state_dict + alphas_cumprod + config.

    Le fichier attendu est : <ckpt_dir>/<model_name>.pt
    """
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Réseau : UNet2DModel reconstruit depuis config + state_dict
    net = UNet2DModel.from_config(ckpt['config']).to(device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    # Schedule : on repart des alphas_cumprod stockés (source de vérité du DDPM)
    alphas_cumprod = ckpt['alphas_cumprod']
    if not isinstance(alphas_cumprod, torch.Tensor):
        alphas_cumprod = torch.as_tensor(alphas_cumprod)
    schedule = DDPMSchedule.from_alphas_cumprod(alphas_cumprod.float())

    print(f"[Modèle] Chargé : {ckpt_path}  (T={schedule.T}, "
          f"in_ch={net.config.in_channels}, size={net.config.sample_size})")
    return net, schedule


def main():
    device = DEVICE
    
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    op = OPERATOR

    # 1. Dégradation
    name, x0_11, y_11 = degrade_image(IMAGE_DIR, op, IMG_SIZE, IN_CH, OUTPUT_DIR)

    # 2. Précalculs (via l'opérateur, aucune matrice)
    y_flat  = y_11.ravel()
    Aty     = op.adjoint(y_flat)              # A^T y
    AtA_diag = op.compute_AtA_diag()          # diag(A^T A)

    print(f"[Opérateur] {type(op).__name__}  —  "
          f"n={op.input_dim()}, m={op.output_dim()}")
 
    # 3. Modèle + schedule pré-entraînés (checkpoint local)
    net, schedule = load_pretrained_net(CKPT_DIR, MODEL_NAME, device)

    # 4. Reverse conditionnel avec correction EMG-VBA
    shape = (IN_CH, IMG_SIZE, IMG_SIZE)
    x_rec, diagnostics = sample_conditional(
        net, schedule, y_flat, op, shape,
        n_samples=N_SAMPLES, device=device,
        emg_n_iter=EMG_N_ITER, emg_skip_after=EMG_SKIP_AFTER,
        a_0=A_0, b_0=B_0, c_0=C_0, d_0=D_0,
        Aty=Aty, AtA_diag=AtA_diag, monitor_steps=MONITOR_STEPS,
    )

    # x_rec : (N, C, H, W) en [0, 1]
    x_rec_np = x_rec[0].cpu().numpy()                # (C, H, W)
    rec_path = os.path.join(OUTPUT_DIR, f"{name}_reconstructed_SigmaBlur_{SIGMA_BLUR}.png")
    if IN_CH == 1:
        img_arr = (x_rec_np[0] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_arr, mode='L').save(rec_path)
    else:
        img_arr = (x_rec_np.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_arr, mode='RGB').save(rec_path)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_reconstructed_sblur_{SIGMA_BLUR}__snoise_{SIGMA_NOISE}.npy"), x_rec_np)
    print(f"[Reconstruction] Sauvegardée : {rec_path}")

    # 5. Affichage résultats image
    def to_hwc(a):
        """(C,H,W) -> (H,W,C) si RGB, sinon (H,W)."""
        if a.ndim == 3 and a.shape[0] == 3:
            return a.transpose(1, 2, 0)
        if a.ndim == 3 and a.shape[0] == 1:
            return a[0]
        return a

    x0_img  = to_hwc((x0_11 + 1) / 2)
    y_img   = to_hwc((y_11  + 1) / 2)
    rec_img = to_hwc(x_rec_np)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    imshow_kw = {} if IN_CH == 3 else {'cmap': 'gray'}
    for ax, img, title in zip(axes,
                              [x0_img, y_img, rec_img],
                              ["Originale", f"Floutée (σ={SIGMA_BLUR})", "Reconstruite (EMG-VBA)"]):
        ax.imshow(np.clip(img, 0, 1), vmin=0, vmax=1, **imshow_kw)
        ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_comparison_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=120)
    plt.show()

    #6.1 Affichage énergie libre F(q) au cours des K itérations EMG-VBA (à t fixé)
    energie = diagnostics['energie_par_step']
    fig, axes = plt.subplots(2, 5, figsize=(20, 6))
    for idx, t_val in enumerate(sorted(energie.keys(), reverse=True)):
        ax = axes[idx // 5][idx % 5]
        data = energie[t_val]  # maintenant c'est un dict de listes
        iters = range(1, len(data['F']) + 1)

        ax.plot(iters, data['F'], 'b-o', markersize=2, label='F(q)')
        
        ax.set_title(f"t = {t_val}")
        ax.set_xlabel("Itération k")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("F(q) doit croître à t fixé (Résultat XII.2.1)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_energie_emgvba_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=150)
    plt.show()

    #6.2 Décomposition de F(q) = log_jointe - entropie
    energie = diagnostics['energie_par_step']
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for idx, t_val in enumerate(sorted(energie.keys(), reverse=True)):
        ax = axes[idx // 5][idx % 5]
        data = energie[t_val]  # maintenant c'est un dict de listes
        iters = range(1, len(data['F']) + 1)

        ax.plot(iters, data['F'], 'b-o', markersize=2, label='F(q)')
        ax.plot(iters, data['log_jointe'], 'g--', linewidth=1, label='log jointe')
        ax.plot(iters, data['entropie'], 'r--', linewidth=1, label='-Entropie')

        ax.set_title(f"t = {t_val}")
        ax.set_xlabel("Itération k")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Décomposition F(q) = E[log p] - E[log q]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_Décomposition_Energie_libre_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=150)
    plt.show()

    #7. Affichage de sigma_b² et sigma_r² au cours du processus de débruitage (tout t)
    tau_b_final = diagnostics['tau_b_final']  # dict {t: tau_b final} ou liste indexée par t
    tau_r_final = diagnostics['tau_r_final']

    # Tri par t décroissant (T -> 0, sens du reverse)
    ts = sorted(tau_b_final.keys(), reverse=True)
    sigma2_b = np.array([1.0 / tau_b_final[t] for t in ts])
    sigma2_r = np.array([1.0 / tau_r_final[t] for t in ts])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(ts, sigma2_b, '-', label=r'$\sigma_b^2$ (bruit mesure)', color='tab:blue')
    ax.plot(ts, sigma2_r, '-', label=r'$\sigma_r^2$ (a priori)',     color='tab:red')
    ax.set_xlabel("t (pas de diffusion)")
    ax.set_ylabel("Variance")
    ax.set_yscale('log')
    ax.invert_xaxis()  # t décroît de T à 0 dans le sens du reverse
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(r"Évolution de $\sigma_b^2$ et $\sigma_r^2$ au cours du reverse diffusion")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_variances_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
