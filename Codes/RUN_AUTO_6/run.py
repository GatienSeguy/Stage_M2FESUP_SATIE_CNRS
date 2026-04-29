"""
run.py — Lance le benchmark avec les paramètres d'un fichier JSON.

Usage :
    python run.py config.json
    python run.py configs/visage.json
"""
import sys
import os
import json
import time
import numpy as np
import torch
from PIL import Image
import json as json_mod
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from diffusion.Schedules import DDPMSchedule
from diffusion.Reverse import sample_conditional as sample_emgvba
from diffusion.Reverse_generique import sample_conditional as sample_generic
from diffusion.correction_dps import dps_correction
from diffusion.correction_PiGDM import pigdm_correction
from operators_torch import GaussianBlurOperator
import importlib.util


def load_operator(cfg, img_size, in_ch):
    """Charge l'opérateur depuis le JSON.

    Formats supportés :

    1) Blur gaussien (par défaut) :
       "operator": {"type": "blur", "sigma": 2.0, "kernel_size": 9}

    2) Opérateur custom depuis un fichier Python :
       "operator": {
           "type": "custom",
           "file": "my_operators.py",
           "class": "InpaintingOperator",
           "params": {"mask_ratio": 0.5}
       }

    3) Pas de champ "operator" → blur gaussien avec sigma_blur du config
    """
    op_cfg = cfg.get('operator', {'type': 'blur'})

    if op_cfg['type'] == 'blur':
        return GaussianBlurOperator(
            kernel_size=op_cfg.get('kernel_size', 9),
            sigma=op_cfg.get('sigma', cfg.get('sigma_blur', 2.0)),
            img_size=img_size,
            n_channels=in_ch,
        )

    elif op_cfg['type'] == 'custom':
        filepath = op_cfg['file']
        class_name = op_cfg['class']
        params = op_cfg.get('params', {})

        spec = importlib.util.spec_from_file_location("custom_op", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        OpClass = getattr(module, class_name)
        return OpClass(img_size=img_size, n_channels=in_ch, **params)

    else:
        raise ValueError(f"Type d'opérateur inconnu : {op_cfg['type']}")


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_metrics(x_rec, x_true, y_np, op):
    if x_rec.ndim == 3 and x_rec.shape[0] == 3:
        x_rec_hwc  = x_rec.transpose(1, 2, 0)
        x_true_hwc = x_true.transpose(1, 2, 0)
    else:
        x_rec_hwc  = x_rec
        x_true_hwc = x_true

    psnr = peak_signal_noise_ratio(x_true_hwc, np.clip(x_rec_hwc, 0, 1), data_range=1.0)
    ssim = structural_similarity(x_true_hwc, np.clip(x_rec_hwc, 0, 1),
                                  data_range=1.0, channel_axis=2 if x_rec.ndim == 3 else None)

    x_rec_11 = 2.0 * x_rec - 1.0
    A_xrec = op.forward(
        torch.tensor(x_rec_11.ravel(), dtype=torch.float32, device='cpu')
    ).cpu().numpy()
    coherence = np.linalg.norm(A_xrec - y_np.ravel()) / (np.linalg.norm(y_np.ravel()) + 1e-12)

    return {'psnr': float(psnr), 'ssim': float(ssim), 'coherence': float(coherence)}


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
    return net, schedule


def to_hwc(a):
    if a.ndim == 3 and a.shape[0] == 3:
        return a.transpose(1, 2, 0)
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    return a


def run(cfg):
    device = cfg.get('device', 'mps')
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    base_dir    = cfg.get('base_dir', os.path.dirname(os.path.abspath(__file__)))
    image_path  = cfg['image_path']
    sigma_blur  = cfg.get('sigma_blur', 2.0)
    sigma_noise = cfg.get('sigma_noise', sigma_blur / 100)
    img_size    = cfg.get('img_size', 256)
    in_ch       = cfg.get('in_ch', 3)
    model_name  = cfg.get('model_name', 'ddpm_ema_celebahq_256')
    ckpt_dir    = cfg.get('ckpt_dir', os.path.join(base_dir, 'checkpoints'))
    output_dir  = cfg['output_dir']
    run_name    = cfg.get('run_name', os.path.splitext(os.path.basename(image_path))[0])

    emg_n_iter = cfg.get('emg_n_iter', 100)
    emg_skip   = cfg.get('emg_skip_after', 0)
    a0 = cfg.get('a_0', 1e-3)
    b0 = cfg.get('b_0', 1e-3)
    c0 = cfg.get('c_0', 1e-3)
    d0 = cfg.get('d_0', 1e-3)

    methods        = cfg.get('methods', ['emgvba'])
    dps_zetas      = cfg.get('dps_zetas', [0.1, 0.5, 1.0, 2.0, 5.0, 10])
    pigdm_sigma2bs = cfg.get('pigdm_sigma2bs', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    monitor_steps  = cfg.get('monitor_steps', [999, 800, 600, 400, 300, 200, 100, 50, 20, 5])

    os.makedirs(output_dir, exist_ok=True)

    op = load_operator(cfg, img_size, in_ch)

    img = Image.open(image_path)
    img = img.convert('RGB') if in_ch == 3 else img.convert('L')
    img = img.resize((img_size, img_size), Image.BICUBIC)
    x0 = np.array(img, dtype=np.float64) / 255.0
    if in_ch == 3 and x0.ndim == 3:
        x0 = x0.transpose(2, 0, 1)
    x0_11 = 2.0 * x0 - 1.0

    y_11 = op.create_observation(x0_11, sigma_noise)
    if isinstance(y_11, torch.Tensor):
        y_11_np = y_11.cpu().numpy()
    else:
        y_11_np = y_11

    y_flat   = op._as_tensor(y_11_np.ravel())
    Aty      = op.adjoint(y_flat)
    AtA_diag = op.compute_AtA_diag()
    x0_01    = (x0_11 + 1) / 2

    net, schedule = load_net(ckpt_dir, model_name, device)
    shape = (in_ch, img_size, img_size)

    results = []
    diag = None

    if 'emgvba' in methods:
        print(f"\n[{run_name}] EMG-VBA...")
        t0 = time.time()
        x_rec, diag = sample_emgvba(
            net, schedule, y_flat, op, shape,
            n_samples=1, device=device,
            emg_n_iter=emg_n_iter, emg_skip_after=emg_skip,
            a_0=a0, b_0=b0, c_0=c0, d_0=d0,
            Aty=Aty, AtA_diag=AtA_diag, monitor_steps=monitor_steps,
        )
        elapsed = time.time() - t0

        x_rec_np = x_rec[0].cpu().numpy()
        m = compute_metrics(x_rec_np, x0_01, y_11_np, op)
        fname = f'{run_name}_emgvba.png'
        m.update({'method': 'EMG-VBA', 'hyperparam': '-', 'time': elapsed,
                  'run_name': run_name, 'filename': fname})
        results.append(m)

        Image.fromarray((np.clip(to_hwc(x_rec_np), 0, 1) * 255).astype(np.uint8),
                         mode='RGB' if in_ch == 3 else 'L') \
             .save(os.path.join(output_dir, fname))

        if diag.get('tau_b_final') and len(diag['tau_b_final']) > 0:
            
            diag_path = os.path.join(output_dir, f'{run_name}_diagnostics.json')
            with open(diag_path, 'w') as f:
                json_mod.dump({
                    'tau_b': {str(k): v for k, v in diag['tau_b_final'].items()},
                    'tau_r': {str(k): v for k, v in diag['tau_r_final'].items()},
                }, f)
            print(f"  Diagnostics → {diag_path}")

        print(f"  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  Coh={m['coherence']:.4f}  T={elapsed:.0f}s")

    if 'dps' in methods:
        for zeta in dps_zetas:
            print(f"[{run_name}] DPS ζ={zeta}...")
            t0 = time.time()
            x_rec, _ = sample_generic(
                net, schedule, y_flat, op, shape,
                correction_fn=dps_correction,
                correction_kwargs={'zeta': zeta},
                n_samples=1, device=device, skip_after=emg_skip,
                Aty=Aty, AtA_diag=AtA_diag, monitor_steps=[],
            )
            elapsed = time.time() - t0

            x_rec_np = x_rec[0].cpu().numpy()
            m = compute_metrics(x_rec_np, x0_01, y_11_np, op)
            fname = f'{run_name}_dps_z{zeta}.png'
            m.update({'method': 'DPS', 'hyperparam': f'z={zeta}', 'time': elapsed,
                      'run_name': run_name, 'filename': fname})
            results.append(m)

            Image.fromarray((np.clip(to_hwc(x_rec_np), 0, 1) * 255).astype(np.uint8),
                             mode='RGB' if in_ch == 3 else 'L') \
                 .save(os.path.join(output_dir, fname))
            print(f"  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  Coh={m['coherence']:.4f}  T={elapsed:.0f}s")

    if 'pigdm' in methods:
        for s2b in pigdm_sigma2bs:
            print(f"[{run_name}] PiGDM σ²_b={s2b}...")
            t0 = time.time()
            x_rec, _ = sample_generic(
                net, schedule, y_flat, op, shape,
                correction_fn=pigdm_correction,
                correction_kwargs={'sigma2_b': s2b},
                n_samples=1, device=device, skip_after=emg_skip,
                Aty=Aty, AtA_diag=AtA_diag, monitor_steps=[],
            )
            elapsed = time.time() - t0

            x_rec_np = x_rec[0].cpu().numpy()
            m = compute_metrics(x_rec_np, x0_01, y_11_np, op)
            fname = f'{run_name}_pigdm_s2b{s2b}.png'
            m.update({'method': 'PiGDM', 'hyperparam': f's2b={s2b}', 'time': elapsed,
                      'run_name': run_name, 'filename': fname})
            results.append(m)

            Image.fromarray((np.clip(to_hwc(x_rec_np), 0, 1) * 255).astype(np.uint8),
                             mode='RGB' if in_ch == 3 else 'L') \
                 .save(os.path.join(output_dir, fname))
            print(f"  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  Coh={m['coherence']:.4f}  T={elapsed:.0f}s")

    # CSV
    import csv
    csv_path = os.path.join(output_dir, f'{run_name}_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_name', 'method', 'hyperparam',
                                                'psnr', 'ssim', 'coherence', 'time', 'filename'])
        writer.writeheader()
        writer.writerows(results)

    print(f"[{run_name}] CSV → {csv_path}")

    # Plots benchmark
    make_plots(results, diag, output_dir, run_name,
               x0_01, y_11_np, x0_11.shape,
               sigma_blur, sigma_noise, in_ch, img_size)

    return results


def make_plots(results, diag, output_dir, run_name,
               x0_01, y_11_np, x0_shape,
               sigma_blur, sigma_noise, in_ch, img_size):
    import matplotlib.pyplot as plt

    # 1. PSNR vs Cohérence
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    emg = [r for r in results if r['method'] == 'EMG-VBA']
    if emg:
        e = emg[0]
        ax.scatter(e['coherence'], e['psnr'], s=200, c='red', marker='*',
                   zorder=10, label='EMG-VBA')
    dps_r = [r for r in results if r['method'] == 'DPS']
    if dps_r:
        ax.plot([r['coherence'] for r in dps_r], [r['psnr'] for r in dps_r],
                'o-', color='blue', label='DPS (zeta variable)')
        for r in dps_r:
            ax.annotate(r['hyperparam'].replace('z=', ''),
                        (r['coherence'], r['psnr']), fontsize=7, ha='left')
    pigdm_r = [r for r in results if r['method'] == 'PiGDM']
    if pigdm_r:
        ax.plot([r['coherence'] for r in pigdm_r], [r['psnr'] for r in pigdm_r],
                's-', color='green', label='PiGDM (sigma2_b variable)')
        for r in pigdm_r:
            ax.annotate(r['hyperparam'].replace('s2b=', ''),
                        (r['coherence'], r['psnr']), fontsize=7, ha='left')
    ax.set_xlabel("Cohérence ||Ax_rec - y|| / ||y||  (- mieux)")
    ax.set_ylabel("PSNR (dB)  (+ mieux)")
    ax.set_title(f"PSNR vs Cohérence — sigma_blur={sigma_blur}, sigma_noise={sigma_noise}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{run_name}_psnr_vs_coherence.png'), dpi=150)
    plt.close(fig)

    # 1bis. PSNR vs SSIM
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    if emg:
        e = emg[0]
        ax.scatter(e['ssim'], e['psnr'], s=200, c='red', marker='*',
                   zorder=10, label='EMG-VBA')
    if dps_r:
        ax.plot([r['ssim'] for r in dps_r], [r['psnr'] for r in dps_r],
                'o-', color='blue', label='DPS (zeta variable)')
        for r in dps_r:
            ax.annotate(r['hyperparam'].replace('z=', ''),
                        (r['ssim'], r['psnr']), fontsize=7, ha='left')
    if pigdm_r:
        ax.plot([r['ssim'] for r in pigdm_r], [r['psnr'] for r in pigdm_r],
                's-', color='green', label='PiGDM (sigma2_b variable)')
        for r in pigdm_r:
            ax.annotate(r['hyperparam'].replace('s2b=', ''),
                        (r['ssim'], r['psnr']), fontsize=7, ha='left')
    ax.set_xlabel("SSIM  (+ mieux)")
    ax.set_ylabel("PSNR (dB)  (+ mieux)")
    ax.set_title(f"PSNR vs SSIM — sigma_blur={sigma_blur}, sigma_noise={sigma_noise}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{run_name}_psnr_vs_ssim.png'), dpi=150)
    plt.close(fig)

    # 2. Bar plot temps
    if results:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        labels = [f"{r['method']}\n{r['hyperparam']}" for r in results]
        times  = [r['time'] for r in results]
        colors = ['red' if r['method'] == 'EMG-VBA'
                  else 'blue' if r['method'] == 'DPS'
                  else 'green' for r in results]
        ax.barh(range(len(results)), times, color=colors, alpha=0.7)
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Temps (secondes)")
        ax.set_title("Temps d'exécution par méthode")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{run_name}_temps.png'), dpi=150)
        plt.close(fig)

    # 3. Galerie visuelle
    n = len(results)
    if n > 0:
        cols = min(n + 2, 17)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.5))
        if cols == 1:
            axes = [axes]
        imkw = {} if in_ch == 3 else {'cmap': 'gray'}
        axes[0].imshow(np.clip(to_hwc(x0_01), 0, 1), **imkw)
        axes[0].set_title("Originale", fontsize=8); axes[0].axis('off')
        y_img = to_hwc((y_11_np.reshape(x0_shape) + 1) / 2)
        axes[1].imshow(np.clip(y_img, 0, 1), **imkw)
        axes[1].set_title(f"Dégradée\nsigma_blur={sigma_blur}", fontsize=8); axes[1].axis('off')
        for idx, r in enumerate(results):
            if idx + 2 >= len(axes):
                break
            img_path = os.path.join(output_dir, r['filename'])
            if os.path.exists(img_path):
                rec = np.array(Image.open(img_path)) / 255.0
                axes[idx + 2].imshow(np.clip(rec, 0, 1), **imkw)
            axes[idx + 2].set_title(f"{r['method']}\n{r['hyperparam']}\nPSNR={r['psnr']:.1f}", fontsize=7)
            axes[idx + 2].axis('off')
        plt.suptitle("Comparaison visuelle", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{run_name}_galerie.png'), dpi=120)
        plt.close(fig)

    # Plots EMG-VBA diagnostics
    if diag is None:
        return
    suffix = f"sblur_{sigma_blur}_snoise_{sigma_noise}"
    energie = diag.get('energie_par_step') or {}

    # 4. F(q) par step
    if energie:
        ts_e = sorted(energie.keys(), reverse=True)
        n_e = len(ts_e)
        nrows = 2 if n_e > 5 else 1
        ncols = (n_e + nrows - 1) // nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for idx, t_val in enumerate(ts_e):
            ax = axes[idx // ncols][idx % ncols]
            data = energie[t_val]
            iters = range(1, len(data['energie_libre']) + 1)
            ax.plot(iters, data['energie_libre'], 'b-o', markersize=2, label='F(q)')
            ax.set_title(f"t = {t_val}")
            ax.set_xlabel("Itération k")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)
        fig.suptitle("F(q) doit croître à t fixé")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{run_name}_emgvba_energie_{suffix}.png"), dpi=150)
        plt.close(fig)

        # 5. Décomposition F = log_jointe - entropie
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
        for idx, t_val in enumerate(ts_e):
            ax = axes[idx // ncols][idx % ncols]
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
        plt.savefig(os.path.join(output_dir, f"{run_name}_emgvba_decomposition_F_{suffix}.png"), dpi=150)
        plt.close(fig)

    # 6. Variances sigma_b^2 / sigma_r^2
    tau_b = diag.get('tau_b_final') or {}
    tau_r = diag.get('tau_r_final') or {}
    if tau_b:
        ts = sorted(tau_b.keys(), reverse=True)
        sigma2_b = np.array([1.0 / tau_b[t] for t in ts])
        sigma2_r = np.array([1.0 / tau_r[t] for t in ts])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(ts, sigma2_b, '-', label=r'$\sigma_b^2$ (bruit mesure)', color='tab:blue')
        ax.plot(ts, sigma2_r, '-', label=r'$\sigma_r^2$ (a priori)', color='tab:red')
        ax.set_xlabel("t (pas de diffusion)")
        ax.set_ylabel("Variance")
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(r"Évolution de $\sigma_b^2$ et $\sigma_r^2$")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{run_name}_emgvba_variances_{suffix}.png"), dpi=150)
        plt.close(fig)

        # 7. Ratio tau_b / tau_r
        ratio = np.array([tau_b[t] / tau_r[t] for t in ts])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(ts, ratio, '-')
        ax.set_xlabel("t (pas de diffusion)")
        ax.set_ylabel(r"$\tau_b / \tau_r$")
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title(r"Ratio $\tau_b / \tau_r$")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{run_name}_emgvba_ratio_variances_{suffix}.png"), dpi=150)
        plt.close(fig)

    # 8. Processus reverse visuel
    snaps_xt = diag.get('snapshots_xt')
    if snaps_xt:
        snaps_mu = diag['snapshots_mu']
        snaps_tw = diag['snapshots_tweedie']
        sigmas   = diag.get('sigma_per_step', {})
        ts = sorted(snaps_xt.keys(), reverse=True)
        n_snaps = len(ts)
        fig, axes = plt.subplots(4, n_snaps, figsize=(2.5 * n_snaps, 6), squeeze=False)
        for idx, t_val in enumerate(ts):
            axes[0, idx].imshow(np.clip(to_hwc(snaps_xt[t_val]), 0, 1))
            axes[0, idx].set_title(f"t={t_val}", fontsize=7)
            axes[0, idx].axis('off')
            axes[1, idx].imshow(np.clip(to_hwc(snaps_mu[t_val]), 0, 1))
            axes[1, idx].axis('off')
            axes[2, idx].imshow(np.clip(to_hwc(snaps_tw[t_val]), 0, 1))
            axes[2, idx].axis('off')
            if t_val in sigmas:
                sig = sigmas[t_val].reshape(in_ch, img_size, img_size).mean(axis=0)
                sig2 = 255 * (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-12)
                axes[3, idx].imshow(sig2 + 1e-10, cmap='hot')
            axes[3, idx].axis('off')
        fig.suptitle("Processus reverse — EMG-VBA  (lignes: x_t / mu_post / tweedie / Sigma)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{run_name}_emgvba_reverse_process.png'), dpi=150)
        plt.close(fig)


def expand_images(cfg):
    """Resolve image_dir / image_paths / image_pattern -> list of image paths."""
    import glob as _glob
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    if 'image_paths' in cfg:
        return list(cfg['image_paths'])
    if 'image_pattern' in cfg:
        return sorted(_glob.glob(cfg['image_pattern']))
    if 'image_dir' in cfg:
        d = cfg['image_dir']
        return sorted([os.path.join(d, f) for f in os.listdir(d)
                       if f.lower().endswith(exts)])
    return [cfg['image_path']]


def run_config(cfg):
    images = expand_images(cfg)
    if len(images) == 1:
        cfg2 = dict(cfg)
        cfg2['image_path'] = images[0]
        return run(cfg2)
    base_out = cfg.get('output_dir', 'resultats')
    for img in images:
        cfg2 = dict(cfg)
        cfg2['image_path'] = img
        name = os.path.splitext(os.path.basename(img))[0]
        cfg2['run_name'] = name
        cfg2['output_dir'] = os.path.join(base_out, name)
        print(f"\n--- image: {img} ---")
        run(cfg2)


if __name__ == '__main__':
    import glob
    if len(sys.argv) < 2:
        print("Usage: python run.py config.json [config2.json ...]")
        sys.exit(1)
    paths = []
    for arg in sys.argv[1:]:
        matched = glob.glob(arg)
        if matched:
            paths.extend(sorted(matched))
        else:
            paths.append(arg)
    if not paths:
        print(f"No config matched: {sys.argv[1:]}")
        sys.exit(1)
    for p in paths:
        print(f"\n=== {p} ===")
        run_config(load_config(p))
