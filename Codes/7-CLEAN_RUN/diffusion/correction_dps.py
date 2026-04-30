import torch


def dps_correction(x_t, xhat0, alpha_bar_t, y, op,
                   zeta=1e-2,
                   Aty=None, AtA_diag=None,
                   warm_start=None):
    """
    Correction DPS : mu_post = xhat0 - zeta * A^T(A xhat0 - y)

    Paramètres
    ----------
    x_t       : torch.Tensor — x_t aplati (n,)
    xhat0     : torch.Tensor — estimé Tweedie aplati (n,)
    alpha_bar_t : float
    y         : torch.Tensor — observation aplatie (m,)
    op        : LinearOperator (forward / adjoint)
    zeta      : float — step size (hyperparamètre à tuner)
    Aty       : torch.Tensor — A^T y précalculé (optionnel)
    AtA_diag  : torch.Tensor — non utilisé par DPS, pour compatibilité
    warm_start: dict — non utilisé par DPS, pour compatibilité

    Retourne
    --------
    mu_post : torch.Tensor (n,)
    state   : dict (pour compatibilité avec l'interface)
    """
    # Résidu dans l'espace des observations
    A_xhat0 = op.forward(xhat0)
    residual = A_xhat0 - y                    # A x0hat - y

    # Gradient de vraisemblance (dans l'espace image)
    grad = op.adjoint(residual)               # A^T(A x0hat - y)

    # Correction
    mu_post = xhat0 - zeta * grad

    state = {
        'mu': mu_post,
        'Sigma': None,
        'tau_r': None,
        'tau_b': None,
        'historique': None,
    }

    return mu_post, state