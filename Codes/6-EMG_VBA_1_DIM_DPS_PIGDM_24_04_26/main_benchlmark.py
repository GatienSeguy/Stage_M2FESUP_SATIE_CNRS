"""
Benchmark : EMG-VBA vs DPS vs PiGDM
Teste les trois méthodes sur la même image, mesure temps + métriques.
"""
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from degrade import load_and_resize
from diffusion.Schedules import DDPMSchedule
from diffusion.Reverse import sample_conditional as sample_emgvba, emg_vba_correction
from diffusion.Reverse_generique import sample_conditional as sample_generic
from diffusion.correction_dps import dps_correction
from diffusion.correction_PiGDM import pigdm_correction
from operators_torch import GaussianBlurOperator

# =====================================================================
# HYPERPARAMÈTRES
# =====================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, 'observations/tete.png')
SIGMA_BLUR = 2.0
SIGMA_NOISE = SIGMA_BLUR / 100

IMG_SIZE   = 256
IN_CH      = 3
MODEL_NAME = 'ddpm_ema_celebahq_256'

OUTPUT_DIR = os.path.join(BASE_DIR, 'resultats/benchmark')
CKPT_DIR   = os.path.join(BASE_DIR, 'checkpoints')
DEVICE     = 'mps'

# EMG-VBA
EMG_N_ITER = 100
A_0 = B_0 = C_0 = D_0 = 1e-3
MONITOR_STEPS = [999, 800, 600, 400, 300, 200, 100, 50, 20, 5] 

# Grilles d'hyperparamètres à tester
# DPS_ZETAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10]
DPS_ZETAS = [0.1,0.2]
# PIGDM_SIGMA2BS = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,1e-1]
PIGDM_SIGMA2BS = [2/100,1/100]

OPERATOR = GaussianBlurOperator(kernel_size=9, sigma=SIGMA_BLUR,
                                img_size=IMG_SIZE, n_channels=IN_CH)

N_SAMPLES = 1
# =====================================================================


def compute_metrics(x_rec, x_true, y, op):
    """Calcule PSNR, SSIM, et erreur de cohérence."""
    # x_rec et x_true en (C, H, W) numpy, [0, 1]
    if x_rec.ndim == 3 and x_rec.shape[0] == 3:
        x_rec_hwc  = x_rec.transpose(1, 2, 0)
        x_true_hwc = x_true.transpose(1, 2, 0)
    else:
        x_rec_hwc  = x_rec
        x_true_hwc = x_true

    psnr = peak_signal_noise_ratio(x_true_hwc, np.clip(x_rec_hwc, 0, 1), data_range=1.0)
    ssim = structural_similarity(x_true_hwc, np.clip(x_rec_hwc, 0, 1),
                                  data_range=1.0, channel_axis=2 if x_rec.ndim == 3 else None)

    # Cohérence : ‖A x_rec - y‖ / ‖y‖  (en [-1,1])
    x_rec_11 = 2.0 * x_rec - 1.0
    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = y
    A_xrec = op.forward_np(x_rec_11.ravel()) if hasattr(op, 'forward_np') else op.forward(
        torch.tensor(x_rec_11.ravel(), dtype=torch.float32, device='cpu')
    ).cpu().numpy()
    coherence = np.linalg.norm(A_xrec - y_np.ravel()) / (np.linalg.norm(y_np.ravel()) + 1e-12)

    return {'psnr': psnr, 'ssim': ssim, 'coherence': coherence}


def load_net(ckpt_dir, model_name, device):
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    from diffusers import UNet2DModel
    net = UNet2DModel.from_config(ckpt['config']).to(device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    alphas_cumprod = ckpt['alphas_cumprod']
    if not isinstance(alphas_cumprod, torch.Tensor):
        alphas_cumprod = torch.as_tensor(alphas_cumprod)
    schedule = DDPMSchedule.from_alphas_cumprod(alphas_cumprod.float())

    print(f"[Modèle] {ckpt_path}")
    return net, schedule


def to_hwc(a):
    if a.ndim == 3 and a.shape[0] == 3:
        return a.transpose(1, 2, 0)
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    return a


def main():
    device = DEVICE
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    op = OPERATOR

    # 1. Image + dégradation
    img = Image.open(IMAGE_DIR)
    img = img.convert('RGB') if IN_CH == 3 else img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    x0 = np.array(img, dtype=np.float64) / 255.0
    if IN_CH == 3 and x0.ndim == 3:
        x0 = x0.transpose(2, 0, 1)
    x0_11 = 2.0 * x0 - 1.0

    y_11 = op.create_observation(x0_11, SIGMA_NOISE)
    if isinstance(y_11, torch.Tensor):
        y_11_np = y_11.cpu().numpy()
    else:
        y_11_np = y_11

    y_flat = op._as_tensor(y_11_np.ravel())
    Aty = op.adjoint(y_flat)
    AtA_diag = op.compute_AtA_diag()

    x0_01 = (x0_11 + 1) / 2  # ground truth [0,1]

    # 2. Modèle
    net, schedule = load_net(CKPT_DIR, MODEL_NAME, device)
    shape = (IN_CH, IMG_SIZE, IMG_SIZE)

    # 3. Stocker les résultats
    results = []

    # =========================================================
    # EMG-VBA
    # =========================================================
    print("\n" + "="*60)
    print("EMG-VBA (aucun hyperparamètre)")
    print("="*60)

    t0 = time.time()
    x_rec, diagnostics = sample_emgvba(
        net, schedule, y_flat, op, shape,
        n_samples=N_SAMPLES, device=device,
        emg_n_iter=EMG_N_ITER, emg_skip_after=0,
        a_0=A_0, b_0=B_0, c_0=C_0, d_0=D_0,
        Aty=Aty, AtA_diag=AtA_diag, monitor_steps=MONITOR_STEPS,
    )
    elapsed = time.time() - t0

    x_rec_np = x_rec[0].cpu().numpy()
    metrics = compute_metrics(x_rec_np, x0_01, y_11_np, op)
    metrics['method'] = 'EMG-VBA'
    metrics['hyperparam'] = '-'
    metrics['time'] = elapsed
    results.append(metrics)

    # Sauvegarder l'image
    rec_img = to_hwc(x_rec_np)
    fname = 'emgvba.png'
    Image.fromarray((np.clip(rec_img, 0, 1) * 255).astype(np.uint8),
                     mode='RGB' if IN_CH == 3 else 'L') \
         .save(os.path.join(OUTPUT_DIR, fname))
    metrics['filename'] = fname

    print(f"  PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}  "
          f"Cohérence={metrics['coherence']:.6f}  Temps={elapsed:.1f}s")

    # =========================================================
    # DPS — grid de zeta
    # =========================================================
    for zeta in DPS_ZETAS:
        print(f"\n{'='*60}")
        print(f"DPS  ζ={zeta}")
        print(f"{'='*60}")

        t0 = time.time()
        x_rec, _ = sample_generic(
            net, schedule, y_flat, op, shape,
            correction_fn=dps_correction,
            correction_kwargs={'zeta': zeta},
            n_samples=N_SAMPLES, device=device,
            skip_after=0,
            Aty=Aty, AtA_diag=AtA_diag, monitor_steps=[],
        )
        elapsed = time.time() - t0

        x_rec_np = x_rec[0].cpu().numpy()
        metrics = compute_metrics(x_rec_np, x0_01, y_11_np, op)
        metrics['method'] = 'DPS'
        metrics['hyperparam'] = f'ζ={zeta}'
        metrics['time'] = elapsed
        results.append(metrics)

        rec_img = to_hwc(x_rec_np)
        fname = f'dps_zeta_{zeta}.png'
        Image.fromarray((np.clip(rec_img, 0, 1) * 255).astype(np.uint8),
                         mode='RGB' if IN_CH == 3 else 'L') \
             .save(os.path.join(OUTPUT_DIR, fname))
        metrics['filename'] = fname

        print(f"  PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}  "
              f"Cohérence={metrics['coherence']:.6f}  Temps={elapsed:.1f}s")

    # =========================================================
    # PiGDM — grid de sigma_b^2
    # =========================================================
    for s2b in PIGDM_SIGMA2BS:
    
        print(f"\n{'='*60}")
        print(f"PiGDM  sigma_b^2={s2b}")
        print(f"{'='*60}")

        correction_kwargs = {'sigma2_b': s2b}
        t0 = time.time()
        x_rec, _ = sample_generic(
            net, schedule, y_flat, op, shape,
            correction_fn=pigdm_correction,
            correction_kwargs= correction_kwargs,
            n_samples=N_SAMPLES, device=device,
            skip_after=0,
            Aty=Aty, AtA_diag=AtA_diag, monitor_steps=[],
        )
        elapsed = time.time() - t0

        x_rec_np = x_rec[0].cpu().numpy()
        metrics = compute_metrics(x_rec_np, x0_01, y_11_np, op)
        metrics['method'] = 'PiGDM'
        metrics['hyperparam'] = f'sigma_b^2={s2b}'
        metrics['time'] = elapsed
        results.append(metrics)

        rec_img = to_hwc(x_rec_np)
        fname = f'pigdm_s2b_{s2b}.png'
        Image.fromarray((np.clip(rec_img, 0, 1) * 255).astype(np.uint8),
                         mode='RGB' if IN_CH == 3 else 'L') \
             .save(os.path.join(OUTPUT_DIR, fname))
        metrics['filename'] = fname

        print(f"  PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}  "
              f"Cohérence={metrics['coherence']:.6f}  Temps={elapsed:.1f}s")

    # =========================================================
    # 4. Tableau récapitulatif
    # =========================================================
    print("\n\n" + "="*90)
    print(f"{'Méthode':<12} {'Hyperparam':<18} {'PSNR':>7} {'SSIM':>8} {'Cohérence':>12} {'Temps (s)':>10}")
    print("-"*90)
    for r in results:
        print(f"{r['method']:<12} {r['hyperparam']:<18} {r['psnr']:>7.2f} {r['ssim']:>8.4f} "
              f"{r['coherence']:>12.6f} {r['time']:>10.1f}")
    print("="*90)

    # =========================================================
    # 5. Scatter plot PSNR vs Cohérence
    # =========================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # EMG-VBA (point unique)
    emg = [r for r in results if r['method'] == 'EMG-VBA'][0]
    ax.scatter(emg['coherence'], emg['psnr'], s=200, c='red', marker='*',
               zorder=10, label='EMG-VBA (aucun hyperparam.)')

    # DPS
    dps_results = [r for r in results if r['method'] == 'DPS']
    dps_coh  = [r['coherence'] for r in dps_results]
    dps_psnr = [r['psnr'] for r in dps_results]
    ax.plot(dps_coh, dps_psnr, 'o-', color='blue', label='DPS (zeta variable)')
    for r in dps_results:
        ax.annotate(r['hyperparam'].replace('ζ=', ''),
                    (r['coherence'], r['psnr']), fontsize=7, ha='left')

    # PiGDM
    pigdm_results = [r for r in results if r['method'] == 'PiGDM']
    pigdm_coh  = [r['coherence'] for r in pigdm_results]
    pigdm_psnr = [r['psnr'] for r in pigdm_results]
    ax.plot(pigdm_coh, pigdm_psnr, 's-', color='green', label='PiGDM (rt^2 variable)')
    for r in pigdm_results:
        ax.annotate(r['hyperparam'].replace('rt^2=', ''),
                    (r['coherence'], r['psnr']), fontsize=7, ha='left')

    ax.set_xlabel("Cohérence ‖Ax_rec - y‖ / ‖y‖  (↓ mieux)")
    ax.set_ylabel("PSNR (dB)  (↑ mieux)")
    ax.set_title(f"PSNR vs Cohérence — sigma_blur={SIGMA_BLUR}, sigma_noise={SIGMA_NOISE}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'benchmark_psnr_vs_coherence.png'), dpi=150)
    plt.show()

    # =========================================================
    # 6. Bar plot temps d'exécution
    # =========================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    labels = [f"{r['method']}\n{r['hyperparam']}" for r in results]
    times  = [r['time'] for r in results]
    colors = ['red' if r['method'] == 'EMG-VBA'
              else 'blue' if r['method'] == 'DPS'
              else 'green'
              for r in results]

    ax.barh(range(len(results)), times, color=colors, alpha=0.7)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Temps (secondes)")
    ax.set_title("Temps d'exécution par méthode")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'benchmark_temps.png'), dpi=150)
    plt.show()

    # =========================================================
    # 7. Galerie visuelle
    # =========================================================
    n_methods = len(results)
    fig, axes = plt.subplots(1, min(n_methods + 2, 17), figsize=(3 * min(n_methods + 2, 17), 3.5))

    # Originale + Dégradée
    imkw = {} if IN_CH == 3 else {'cmap': 'gray'}
    axes[0].imshow(np.clip(to_hwc(x0_01), 0, 1), **imkw)
    axes[0].set_title("Originale", fontsize=8); axes[0].axis('off')

    y_img = to_hwc((y_11_np.reshape(x0_11.shape) + 1) / 2)
    axes[1].imshow(np.clip(y_img, 0, 1), **imkw)
    axes[1].set_title(f"Dégradée\nsigma_blur={SIGMA_BLUR}", fontsize=8); axes[1].axis('off')

    # Reconstructions
    for idx, r in enumerate(results):
        if idx + 2 >= len(axes):
            break
        method_name = r['method']
        hp = r['hyperparam']
        img_path = os.path.join(OUTPUT_DIR, r['filename'])
        if os.path.exists(img_path):
            rec = np.array(Image.open(img_path)) / 255.0
            axes[idx + 2].imshow(np.clip(rec, 0, 1))
        else:
            axes[idx + 2].text(0.5, 0.5, f'fichier manquant:\n{r["filename"]}',
                               ha='center', va='center',
                               transform=axes[idx + 2].transAxes, fontsize=7)
        axes[idx + 2].set_title(f"{method_name}\n{hp}\nPSNR={r['psnr']:.1f}", fontsize=7)
        axes[idx + 2].axis('off')

    plt.suptitle("Comparaison visuelle", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'benchmark_galerie.png'), dpi=120)
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
        plt.savefig(os.path.join(OUTPUT_DIR, f"emgvba_energie_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=150)
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
        plt.savefig(os.path.join(OUTPUT_DIR, f"emgvba_decomposition_F_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=150)
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
        plt.savefig(os.path.join(OUTPUT_DIR, f"emgvba_variances_sblur_{SIGMA_BLUR}_snoise_{SIGMA_NOISE}.png"), dpi=150)
        plt.show()

    # 10. Processus reverse visuel
    print('snapshots' in diagnostics )
    if 'snapshots' in diagnostics and diagnostics['snapshots']:
        snaps = diagnostics['snapshots']
        n_snaps = len(snaps)
        fig, axes = plt.subplots(1, n_snaps, figsize=(2.5 * n_snaps, 3))
        for idx, t_val in enumerate(sorted(snaps.keys(), reverse=True)):
            img_t = snaps[t_val]
            axes[idx].imshow(np.clip(to_hwc(img_t), 0, 1))
            axes[idx].set_title(f"t = {t_val}", fontsize=8)
            axes[idx].axis('off')
        fig.suptitle("Processus reverse — EMG-VBA")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'emgvba_reverse_process.png'), dpi=150)
        plt.show()

    # Sauvegarder les résultats en CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, 'benchmark_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'hyperparam', 'psnr', 'ssim', 'coherence', 'time', 'filename'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[CSV] Résultats sauvegardés : {csv_path}")


if __name__ == "__main__":
    main()