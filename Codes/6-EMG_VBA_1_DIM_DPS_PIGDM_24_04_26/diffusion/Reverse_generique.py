"""
Reverse conditionnel générique.

La méthode de correction est passée en paramètre :
    correction_fn(x_t, xhat0, alpha_bar_t, y, op, **kwargs) → (mu_post, state)

Utilisable avec :
    - emg_vba_correction  (estimation automatique de tau_r, tau_b)
    - dps_correction      (hyperparamètre zeta)
    - pigdm_correction    (hyperparamètres r_t^2, σ_b^2)
"""

import torch
from tqdm import tqdm
from .Forward import tweedie_estimate


def reverse_step_ddpm(x_t, mu_post, t_val, alphas_bar, alphas, betas):
    alpha_bar_t    = alphas_bar[t_val]
    alpha_bar_prev = alphas_bar[t_val - 1]
    beta_t         = betas[t_val]
    alpha_t        = alphas[t_val]

    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    mu = coef_x0 * mu_post + coef_xt * x_t

    beta_tilde = (1.0 - alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    z = torch.randn_like(x_t)

    return mu + torch.sqrt(beta_tilde) * z


@torch.no_grad()
def sample_conditional(net, schedule, y_flat, op, shape,
                       correction_fn,
                       correction_kwargs=None,
                       n_samples=1, device='mps',
                       skip_after=0,
                       Aty=None, AtA_diag=None,
                       monitor_steps=None):
    """
    Paramètres
    ----------
    net             : réseau de diffusion (UNet ou diffusers)
    schedule        : DDPMSchedule
    y_flat          : torch.Tensor — observation aplatie
    op              : LinearOperator
    shape           : tuple (C, H, W)
    correction_fn   : callable(x_t, xhat0, alpha_bar_t, y, op, **kwargs) → (mu_post, state)
    correction_kwargs : dict — arguments supplémentaires passés à correction_fn
                        Ex: {'zeta': 0.01} pour DPS,
                            {'sigma2_b': 0.01, 'r2_t': None} pour ΠGDM,
                            {'n_iter': 100, 'a_0': 1e-3, ...} pour EMG-VBA
    n_samples       : int
    device          : str
    skip_after      : int — ne pas corriger pour t < skip_after
    Aty             : torch.Tensor — Aᵀy précalculé
    AtA_diag        : torch.Tensor — diag(AᵀA) précalculé
    monitor_steps   : list[int] — pas t où sauvegarder les diagnostics
    """
    if correction_kwargs is None:
        correction_kwargs = {}

    net.eval()
    alphas_bar = schedule.alphas_bar.to(device)
    sigmas     = schedule.sigmas.to(device)
    C, H, W   = shape

    x_t = torch.randn(n_samples, *shape, device=device)

    warm_starts = [None] * n_samples

    monitor_set = set(monitor_steps) if monitor_steps is not None else set()
    diagnostics = {
        'energie_par_step': {},
        'tau_b_final': {},
        'tau_r_final': {},
    }

    for t_val in tqdm(reversed(range(schedule.T)), total=schedule.T, desc="Sampling conditionnel"):
        t_batch     = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        alpha_bar_t = alphas_bar[t_val].item()

        # Étape 1 : Tweedie
        output = net(x_t, t_batch)
        eps_pred = output.sample if hasattr(output, 'sample') else output
        xhat0 = tweedie_estimate(x_t, eps_pred, t_batch, alphas_bar, sigmas)

        # Étape 2 : Correction conditionnelle
        if t_val >= skip_after:
            mu_post_list = []
            for b in range(n_samples):
                x_t_flat   = x_t[b].reshape(-1)
                xhat0_flat = xhat0[b].reshape(-1)

                mu_post, state = correction_fn(
                    x_t_flat, xhat0_flat, alpha_bar_t, y_flat, op,
                    Aty=Aty, AtA_diag=AtA_diag,
                    warm_start=warm_starts[b],
                    **correction_kwargs,
                )

                warm_starts[b] = state

                # Diagnostics
                if b == 0 and t_val in monitor_set and state.get('historique') is not None:
                    diagnostics['energie_par_step'][t_val] = state['historique']
                if b == 0:
                    tau_b = state.get('tau_b')
                    tau_r = state.get('tau_r')
                    if tau_b is not None:
                        diagnostics['tau_b_final'][t_val] = tau_b.item() if isinstance(tau_b, torch.Tensor) else tau_b
                    if tau_r is not None:
                        diagnostics['tau_r_final'][t_val] = tau_r.item() if isinstance(tau_r, torch.Tensor) else tau_r

                mu_post_list.append(mu_post.reshape(C, H, W).float().to(device))

            mu_post_tensor = torch.stack(mu_post_list, dim=0)
        else:
            mu_post_tensor = xhat0

        # Étape 3 : Pas reverse
        if t_val == 0:
            x_t = mu_post_tensor
        elif schedule.variant in ('ddpm', 'vpou'):
            x_t = reverse_step_ddpm(x_t, mu_post_tensor, t_val, alphas_bar,
                                     schedule.alphas.to(device), schedule.betas.to(device))

    return (x_t.clamp(-1, 1) + 1) / 2, diagnostics 