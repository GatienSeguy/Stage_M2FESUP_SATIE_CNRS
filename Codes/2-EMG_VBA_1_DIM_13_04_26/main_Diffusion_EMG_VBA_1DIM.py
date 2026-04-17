import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from degrade import load_and_resize, blur, build_A
from model import UNet
from diffusion.Schedules import DDPMSchedule
from diffusion.Reverse import sample_conditional

# ==================
# HYPERPARAMÈTRES
# ===================
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR      = os.path.join(BASE_DIR, 'observations/lune.jpg')
SIGMA_BLUR     = 1.5
IMG_SIZE       = 64
OUTPUT_DIR     = os.path.join(BASE_DIR, 'resultats')
CKPT_DIR       = os.path.join(BASE_DIR, 'checkpoints')
DEVICE         = 'mps' 

# Schedule DDPM
T              = 1000
BETA_START     = 1e-4
BETA_END       = 0.02

# EMG-VBA
EMG_N_ITER     = 100
EMG_SKIP_AFTER = 0
A_0            = 1e-3
B_0            = 1e-3
C_0            = 1e-3
D_0            = 1e-3

N_SAMPLES      = 1

MONITOR_STEPS = [999, 900, 800, 700, 600, 500, 400, 300, 200, 100]
# ==============


def degrade_image(image_path, sigma_blur, img_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    x0 = load_and_resize(image_path, img_size)
    y  = blur(x0, sigma_blur)

    x0_11 = 2.0 * x0 - 1.0
    y_11  = 2.0 * y  - 1.0

    np.save(os.path.join(output_dir, f"{name}_clean.npy"), x0_11)
    np.save(os.path.join(output_dir, f"{name}_blur.npy"),  y_11)
    Image.fromarray((y * 255).clip(0, 255).astype(np.uint8), mode='L') \
         .save(os.path.join(output_dir, f"{name}_blur.png"))

    print(f"[Dégradation] {img_size}x{img_size}, sigma_blur = {sigma_blur}")

    return(name, x0_11, y_11)


def load_net(ckpt_dir, device):

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))

    if not ckpts:
        raise FileNotFoundError(f"Aucun checkpoint dans {ckpt_dir}")
    
    ckpt_path = ckpts[-1]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    net = UNet(in_ch=1).to(device)

    state = ckpt.get('ema_state_dict', ckpt['model_state_dict'])

    net.load_state_dict(state)

    net.eval()

    print(f"[Modèle] Chargé : {ckpt_path}")
    return( net, ckpt.get('hyperparams', {}))


def main():
    device = DEVICE
    
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    # 1. Dégradation
    name, x0_11, y_11 = degrade_image(IMAGE_DIR, SIGMA_BLUR, IMG_SIZE, OUTPUT_DIR)

    # 2. Opérateur A
    print("[Opérateur] Construction de A...")
    A = build_A(SIGMA_BLUR, IMG_SIZE)
    AtA = A.T @ A
    y_flat = y_11.ravel()
    Aty = A.T @ y_flat
    AtA_diag = np.diag(AtA).copy()

    # 3. Modèle + schedule
    net, _ = load_net(CKPT_DIR, device)
    schedule = DDPMSchedule(T=T, beta_start=BETA_START, beta_end=BETA_END)

    # 4. Reverse conditionnel avec correction EMG-VBA
    shape = (1, IMG_SIZE, IMG_SIZE)
    x_rec, diagnostics = sample_conditional(
        net, schedule, y_flat, A, shape,
        n_samples=N_SAMPLES, device=device,
        emg_n_iter=EMG_N_ITER, emg_skip_after=EMG_SKIP_AFTER,
        a_0=A_0, b_0=B_0, c_0=C_0, d_0=D_0,
        AtA=AtA, Aty=Aty, AtA_diag=AtA_diag, monitor_steps=MONITOR_STEPS
    )

    x_rec_np = x_rec[0, 0].cpu().numpy()
    rec_path = os.path.join(OUTPUT_DIR, f"{name}_reconstructed_SigmaBlur_{SIGMA_BLUR}.png")
    Image.fromarray((x_rec_np * 255).clip(0, 255).astype(np.uint8), mode='L').save(rec_path)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_reconstructed_SigmaBlur_{SIGMA_BLUR}.npy"), x_rec_np)
    print(f"[Reconstruction] Sauvegardée : {rec_path}")

    # 5. Affichage résultats image
    x0_img = (x0_11 + 1) / 2
    y_img  = (y_11  + 1) / 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes,
                              [x0_img, y_img, x_rec_np],
                              ["Originale", f"Floutée (σ={SIGMA_BLUR})", "Reconstruite (EMG-VBA)"]):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_comparison_SigmaBlur_{SIGMA_BLUR}.png"), dpi=120)
    plt.show()

    #6. Affichage énergie libre F(q) au cours des K itérations EMG-VBA (à t fixé)
    energie = diagnostics['energie_par_step']
    fig, axes = plt.subplots(2, 5, figsize=(20, 6))
    for idx, t_val in enumerate(sorted(energie.keys(), reverse=True)):
        ax = axes[idx // 5][idx % 5]
        nrj = energie[t_val]
        ax.plot(range(1, len(nrj) + 1), nrj, 'o-', markersize=3)
        ax.set_title(f"t = {t_val}")
        ax.set_xlabel("Itération k")
        ax.set_ylabel("F(q)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("F(q) doit croître à t fixé (Résultat XII.2.1)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_energie_emgvba_SigmaBlur_{SIGMA_BLUR}.png"), dpi=150)
    plt.show()

    #7. Affichage de sigma_b² et sigma_r² au cours du processus de débruitage (tout t)
    tau_b_final = diagnostics['tau_b_final']  # dict {t: tau_b final} ou liste indexée par t
    tau_r_final = diagnostics['tau_r_final']

    # Tri par t décroissant (T -> 0, sens du reverse)
    ts = sorted(tau_b_final.keys(), reverse=True)
    sigma2_b = np.array([1.0 / tau_b_final[t] for t in ts])
    sigma2_r = np.array([1.0 / tau_r_final[t] for t in ts])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(ts, 1/sigma2_b, '-', label=r'$\sigma_b^2$ (bruit mesure)', color='tab:blue')
    ax.plot(ts, 1/sigma2_r, '-', label=r'$\sigma_r^2$ (a priori)',     color='tab:red')
    ax.set_xlabel("t (pas de diffusion)")
    ax.set_ylabel("Variance")
    ax.set_yscale('log')
    ax.invert_xaxis()  # t décroît de T à 0 dans le sens du reverse
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(r"Évolution de $\sigma_b^2$ et $\sigma_r^2$ au cours du reverse diffusion")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_variances_SigmaBlur_{SIGMA_BLUR}.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
