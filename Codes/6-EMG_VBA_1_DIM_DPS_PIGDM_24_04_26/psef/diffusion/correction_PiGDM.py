import torch 

def pigdm_correction(x_t, xhat0, alpha_bar_t, y, op,
                     sigma2_b=1e-2,
                     Aty=None, AtA_diag=None,
                     warm_start=None):
    """
    ΠGDM exact (Song et al., 2023) — approximation diagonale.
    
    mu_post_i = xhat0_i + r²_t · (AᵀA)_ii / (σ²_b + r²_t · (AᵀA)_ii) 
                · (Aᵀ(y - A xhat0))_i / (AᵀA)_ii
    
    Simplifié :
    mu_post_i = xhat0_i + r²_t / (σ²_b + r²_t · (AᵀA)_ii) · (Aᵀ(y - A xhat0))_i
    
    Seul hyperparamètre : σ²_b (variance du bruit, fixée manuellement).
    r²_t = (1 - αbar_t) / αbar_t est déterminé par le schedule.
    """
    # r²_t déterminé par le schedule (pas un hyperparamètre)
    r2_t = (1.0 - alpha_bar_t) / max(alpha_bar_t, 1e-8)

    # Résidu
    A_xhat0 = op.forward(xhat0)
    residual = y - A_xhat0                          # y - A x̂₀
    At_residual = op.adjoint(residual)              # Aᵀ(y - A x̂₀)

    # Poids diagonal : r²_t / (σ²_b + r²_t · (AᵀA)_ii)
    weight = r2_t / (sigma2_b + r2_t * AtA_diag)

    # Correction
    mu_post = xhat0 + weight * At_residual

    state = {
        'mu': mu_post,
        'Sigma': None,
        'tau_r': 1.0 / r2_t,
        'tau_b': 1.0 / sigma2_b,
        'historique': None,
    }

    return mu_post, state