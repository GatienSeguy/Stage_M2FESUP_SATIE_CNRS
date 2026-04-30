import torch
import numpy as np
from tqdm import tqdm

from .Forward import tweedie_estimate
from emg_vba_1_dim_torch import EMG_VBA

def reverse_step_ddpm(x_t, mu_post, t_val, alphas_bar, alphas, betas):
    
    alpha_bar_t    = alphas_bar[t_val] # alphabar_t - Def III.3.1
    alpha_bar_prev = alphas_bar[t_val - 1] # alphabar_t_{t-1}
    beta_t         = betas[t_val] # beta_t
    alpha_t        = alphas[t_val] # alpha_t = 1 - beat_t

    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    mu = coef_x0 * mu_post + coef_xt * x_t

    beta_tilde = (1.0 - alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    z = torch.randn_like(x_t)

    return mu + torch.sqrt(beta_tilde) * z


def reverse_step_ve(x_t, mu_post, t_val, sigmas):
    sigma_t    = sigmas[t_val] # simga_t -Résultat III.1.1
    sigma_prev = sigmas[t_val - 1] # sigma_{t-1}
    delta_sigma2 = sigma_t**2 - sigma_prev**2

    score = (mu_post - x_t) / (sigma_t**2)
    z = torch.randn_like(x_t)

    return x_t + delta_sigma2 * score + torch.sqrt(delta_sigma2) * z


def emg_vba_correction(x_t, xhat0, alpha_bar_t, y, op,
                       n_iter=5,
                       a_0=1e-3, b_0=1e-3, c_0=1e-3, d_0=1e-3,
                       Aty=None, AtA_diag=None,
                       warm_start=None):

    n = x_t.numel()

    
    # if warm_start is not None:
    #     mu_init    = warm_start['mu']
    #     Sigma_init = warm_start['Sigma']

    if warm_start is not None:
        mu_old      = warm_start['mu']
        Sigma_old   = warm_start['Sigma']
        mu_r_old    = warm_start['mu_r']
        Sigma_r_old = warm_start['Sigma_r']
        tau_b_old   = warm_start['tau_b']
        tau_r_old   = warm_start['tau_r']

        # Calculer la nouvelle q_r à t-1 (éq. 94-95 avec anciens tau)
        abar = alpha_bar_t
        Sigma_r_new_inv = abar / (1 - abar) + tau_b_old * AtA_diag + tau_r_old
        Sigma_r_new = 1.0 / Sigma_r_new_inv

        AtA_mu = op.adjoint(op.forward(mu_old))
        terme_mesure = Aty - AtA_mu + AtA_diag * mu_old
        mu_r_new = Sigma_r_new * (
            (abar ** 0.5) / (1 - abar) * x_t
            + tau_b_old * terme_mesure
            + tau_r_old * xhat0
        )

        # Transport OT
        ratio = torch.sqrt(Sigma_r_new / Sigma_r_old)
        mu_init    = mu_r_new + ratio * (mu_old - mu_r_old)
        Sigma_init = (Sigma_r_new / Sigma_r_old) * Sigma_old
    
    else:
        mu_init    = xhat0.clone()
        r2_t       = (1.0 - alpha_bar_t) / max(alpha_bar_t, 1e-8)
        Sigma_init = torch.full((n,), r2_t, device=x_t.device, dtype=x_t.dtype)

    emg = EMG_VBA(
        op=op, y=y,
        alpha_bar_t=alpha_bar_t,
        xt=x_t, xhat0=xhat0,
        a_0=a_0, b_0=b_0, c_0=c_0, d_0=d_0,
        Aty=Aty, AtA_diag=AtA_diag,
    )

    result = emg.executer(
        n_iter=n_iter,
        mu_init=mu_init,
        Sigma_init=Sigma_init,
        a_0_init=a_0, b_0_init=b_0, c_0_init=c_0, d_0_init=d_0,
        verbose=False, affichage=False,
    )

    state = {
        'mu':    result['mu'],
        'Sigma': result['Sigma'],
        'tau_r': result['tau_r'],
        'tau_b': result['tau_b'],
        'historique': result['historique'],
        'mu_r':     result['mu_r'],# POUR OPTIMAL TRANSPORT
        'Sigma_r':  result['Sigma_r'], # POUR OPTIMAL TRANSPORT
    }

    return result['mu'], state

# =====================================================================
# Pour chaque pas t = T, ..., 0 :
# 1. Tweedie (12)
# 2. EMG-VBA :  mu_post approx E[X_0|X_t, Y] (27)
# (estime tau_r, tau_b en interne  (75)/(85))
# 3. Reverse :  x_{t-1} approx q(.|x_t, mu_post) - Section VI.2
# Remplacer x_0 chapeau par mu_post <=> score conditionnel  (36) RÉDIGER preuve clean un jour
# =====================================================================

@torch.no_grad()
def sample_conditional(net, schedule, y_flat, op, shape,
                       n_samples=1, device='mps',
                       emg_n_iter=5, emg_skip_after=10,
                       a_0=1e-3, b_0=1e-3, c_0=1e-3, d_0=1e-3,
                       Aty=None, AtA_diag=None, monitor_steps=None):
    net.eval()
    alphas_bar = schedule.alphas_bar.to(device)
    sigmas     = schedule.sigmas.to(device)
    C, H, W   = shape

    # x_T approx N(0, I)
    x_t = torch.randn(n_samples, *shape, device=device)

    # Warm start EMG-VBA : un état par sample
    warm_starts = [None] * n_samples

    monitor_set = set(monitor_steps) if monitor_steps is not None else set()
    diagnostics = {
        'energie_par_step': {},
        'tau_b_final': {},
        'tau_r_final': {},
        'snapshots_xt': {},
        'snapshots_mu': {},
        'snapshots_tweedie': {},
        'sigma_per_step': {},
    }
    snapshot_steps = {999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 20, 5, 0}

    for t_val in tqdm(reversed(range(schedule.T)), total=schedule.T, desc="Sampling conditionnel"):
        t_batch     = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        alpha_bar_t = alphas_bar[t_val].item()

        # Étape 1 : Tweedie
        output = net(x_t, t_batch)
        eps_pred = output.sample if hasattr(output, 'sample') else output
        xhat0    = tweedie_estimate(x_t, eps_pred, t_batch, alphas_bar, sigmas)

        # Étape 2 : EMG-VBA -> mu_post
        if t_val >= emg_skip_after:
            mu_post_list = []
            for b in range(n_samples):
                x_t_flat   = x_t[b].reshape(-1)        
                xhat0_flat = xhat0[b].reshape(-1)

                mu_post, state = emg_vba_correction(
                    x_t_flat, xhat0_flat, alpha_bar_t, y_flat, op,
                    n_iter=emg_n_iter,
                    a_0=a_0, b_0=b_0, c_0=c_0, d_0=d_0,
                    Aty=Aty, AtA_diag=AtA_diag,
                    warm_start=warm_starts[b],
                )
                warm_starts[b] = state
                if b == 0 and t_val in monitor_set:
                    diagnostics['energie_par_step'][t_val] = state['historique']
                if b == 0:
                    tau_b = state['tau_b']
                    tau_r = state['tau_r']
                    diagnostics['tau_b_final'][t_val] = tau_b.item() if isinstance(tau_b, torch.Tensor) else tau_b
                    diagnostics['tau_r_final'][t_val] = tau_r.item() if isinstance(tau_r, torch.Tensor) else tau_r
                    
                mu_post_list.append(mu_post.reshape(C, H, W).float().to(device))

            mu_post_tensor = torch.stack(mu_post_list, dim=0)
        else:
            mu_post_tensor = xhat0  # pas de correction, mu_post = x_0 chapeau

        if t_val in snapshot_steps:
                snap_xt = (x_t[0].clamp(-1, 1) + 1) / 2
                diagnostics['snapshots_xt'][t_val] = snap_xt.cpu().numpy()
                snap_tweedie = (xhat0[0].clamp(-1, 1) + 1) / 2
                diagnostics['snapshots_tweedie'][t_val] = snap_tweedie.cpu().numpy()
                snap_mu = (mu_post_tensor[0].clamp(-1, 1) + 1) / 2
                diagnostics['snapshots_mu'][t_val] = snap_mu.cpu().numpy()
                diagnostics['sigma_per_step'][t_val] = state['Sigma'].cpu().numpy() if isinstance(state['Sigma'], torch.Tensor) else state['Sigma']
                

        # Étape 3 : Pas reverse
        if t_val == 0:
            x_t = mu_post_tensor
        elif schedule.variant in ('ddpm', 'vpou'):
            x_t = reverse_step_ddpm(x_t, mu_post_tensor, t_val, alphas_bar,
                                     schedule.alphas.to(device), schedule.betas.to(device))
        elif schedule.variant == 've':
            x_t = reverse_step_ve(x_t, mu_post_tensor, t_val, sigmas)

    return (x_t.clamp(-1, 1) + 1) / 2, diagnostics