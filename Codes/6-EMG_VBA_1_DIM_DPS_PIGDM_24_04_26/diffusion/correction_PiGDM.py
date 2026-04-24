"""
Pi GDM — Pseudo-Inverse Guided Diffusion Model (Song et al., 2023)

Approximation gaussienne de la loi a posteriori p(x₀ | x_t, y) :
    X0 | X_t, Y  approx  N(mu_post, Sigma_post)

avec :
    Sigma_post = (A^TA / Sigma_b^2 + I / r_t^2)^{-1}
    mu_post = Sigma_post (A^Ty / Sigma_b^2 + x0hat / r_t^2)

Exploité en diagonal (composante par composante) pour éviter
l'inversion matricielle n×n :
    Sigma_post_i = 1 / (diag(A^TA)_i / Sigma_b^2 + 1 / r_t^2)
    mu_post_i = Sigma_post_i · (A^Ty_i / Sigma_b^2 + x0hat_i / r_t^2)

Hyperparamètres : r_t^2 (incertitude denoiser) et Sigma_b^2 (bruit mesure)
                  — tous deux fixés manuellement.

Référence : Section VII.6 et X.3 du rapport (Résultat X.3.1, eq. 40-41).
"""

import torch


def pigdm_correction(x_t, xhat0, alpha_bar_t, y, op,
                     step_size=1.0, sigma2_b=1e-2,
                     Aty=None, AtA_diag=None,
                     warm_start=None):

    # Résidu
    A_xhat0 = op.forward(xhat0)
    residual = y - A_xhat0

    # Gradient ΠGDM : √αbar / σ²_b · Aᵀ(y - A x̂₀)
    sqrt_abar = alpha_bar_t ** 0.5
    grad = (sqrt_abar / sigma2_b) * op.adjoint(residual)

    # Correction sur xhat0
    mu_post = xhat0 + step_size * grad

    state = {
        'mu': mu_post,
        'Sigma': None,
        'tau_r': None,
        'tau_b': 1.0 / sigma2_b,
        'historique': None,
    }

    return mu_post, state