"""
plot_ratios.py — Trace les ratios tau_b/tau_r pour toutes les images.

Lit les fichiers *_diagnostics.json dans resultats/**/ et superpose les courbes.

Usage : python plot_ratios.py
"""
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import json

if len(sys.argv) < 2:
    print("Usage: python plot_ratios.py <dossier_resultats>")
    print("  Ex:  python plot_ratios.py resultats/CELEBA_VAL")
    sys.exit(1)

output_base = sys.argv[1]
json_files = sorted(glob.glob(os.path.join(output_base, "**", "*_diagnostics.json"), recursive=True))

if not json_files:
    print("Aucun fichier diagnostics trouvé.")
    exit()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']

for i, path in enumerate(json_files):
    with open(path) as f:
        data = json.load(f)
    tau_b = {int(k): v for k, v in data['tau_b'].items()}
    tau_r = {int(k): v for k, v in data['tau_r'].items()}

    ts = sorted(tau_b.keys(), reverse=True)
    ratio    = np.array([tau_b[t] / tau_r[t] for t in ts])
    sigma2_b = np.array([1.0 / tau_b[t] for t in ts])
    sigma2_r = np.array([1.0 / tau_r[t] for t in ts])

    name = os.path.basename(path).replace('_diagnostics.json', '')
    c = colors[i % len(colors)]

    axes[0].plot(ts, ratio, '-', color=c, label=name, linewidth=1.5)
    axes[1].plot(ts, sigma2_b, '-', color=c, label=name, linewidth=1.5)
    axes[2].plot(ts, sigma2_r, '-', color=c, label=name, linewidth=1.5)

# Ratio
axes[0].set_xlabel("t (pas de diffusion)")
axes[0].set_ylabel(r"$\tau_b / \tau_r$")
axes[0].set_yscale('log')
axes[0].invert_xaxis()
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_title(r"Ratio $\tau_b / \tau_r$")

# sigma_b^2
axes[1].set_xlabel("t (pas de diffusion)")
axes[1].set_ylabel(r"$\sigma_b^2$")
axes[1].set_yscale('log')
axes[1].invert_xaxis()
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_title(r"$\sigma_b^2$ estimé")

# sigma_r^2
axes[2].set_xlabel("t (pas de diffusion)")
axes[2].set_ylabel(r"$\sigma_r^2$")
axes[2].set_yscale('log')
axes[2].invert_xaxis()
axes[2].grid(True, alpha=0.3)
axes[2].legend()
axes[2].set_title(r"$\sigma_r^2$ estimé")

plt.suptitle("Comparaison entre images", fontsize=14)
plt.tight_layout()

out_path = os.path.join(output_base, "comparaison_ratios.png")
plt.savefig(out_path, dpi=150)
print(f"Plot sauvegardé : {out_path}")
plt.show()
