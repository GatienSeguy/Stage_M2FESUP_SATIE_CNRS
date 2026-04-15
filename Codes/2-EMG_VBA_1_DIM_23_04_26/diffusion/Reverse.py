import torch
import numpy as np
from tqdm import tqdm

from .Forward import tweedie_estimate
from emg_vba_1_dim import EMG_VBA


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


def emg_vba_correction(x_t_np, xhat0_np, alpha_bar_t, y_np, A_matrix,
                       n_iter=5,
                       a_0=1e-3, b_0=1e-3, c_0=1e-3, d_0=1e-3,
                       AtA=None, Aty=None, AtA_diag=None,
                       warm_start=None):
    
    n = len(x_t_np)

    # --- Initialisation ---
    if warm_start is not None:
        mu_init    = warm_start['mu']
        Sigma_init = warm_start['Sigma']
    else:
        mu_init    = xhat0_np.copy()  # centré sur Tweedie — (12)
        r2_t       = (1.0 - alpha_bar_t) / max(alpha_bar_t, 1e-8)  # r^2_t = (1-alpha_bar_t)/alpha_bar_t — Hypothèses VII.6.1
        Sigma_init = np.full(n, r2_t)

    # --- EMG-VBA ---
    emg = EMG_VBA(
        A=A_matrix, y=y_np,
        alpha_bar_t=alpha_bar_t,
        xt=x_t_np, xhat0=xhat0_np,
        a_0=a_0, b_0=b_0, c_0=c_0, d_0=d_0,
        AtA=AtA, Aty=Aty, AtA_diag=AtA_diag,
    )

    result = emg.executer(
        n_iter=n_iter,
        mu_init=mu_init,
        Sigma_init=Sigma_init,
        a_0_init=a_0, b_0_init=b_0, c_0_init=c_0, d_0_init=d_0,
        verbose=False, affichage=False,
    )

    state = {
        'mu':    result['mu'], # mu_post — moyenne a posteriori
        'Sigma': result['Sigma'], # Sigma — variances marginales — (51)
        'tau_r': result['tau_r'],  # <tau_r> = tilde a_r/tilde b_r — (84)/(85)
        'tau_b': result['tau_b'],# <tau_b> = tilde a_b/tilde b_b — (75)/(76)
        'historique': result['historique']
    }

    return result['mu'], state


# =====================================================================
#   Pour chaque pas t = T, ..., 1 :
#     1. Tweedie (12)
#     2. EMG-VBA :  mu_post approx E[X_0|X_t, Y] (27)
#                   (estime tau_r, tau_b en interne  (75)/(85))
#     3. Reverse :  x_{t-1} approx q(.|x_t, mu_post) - Section VI.2
#
#   Remplacer x_0 chapeau par mu_post <=> score conditionnel  (36) RÉDIGER preuve un jour
# =====================================================================

@torch.no_grad()
def sample_conditional(net, schedule, y_np, A_matrix, shape,
                       n_samples=1, device='mps',
                       emg_n_iter=5, emg_skip_after=10,
                       a_0=1e-3, b_0=1e-3, c_0=1e-3, d_0=1e-3,
                       AtA=None, Aty=None, AtA_diag=None, monitor_steps=None):
    net.eval()
    alphas_bar = schedule.alphas_bar.to(device)
    sigmas     = schedule.sigmas.to(device)
    C, H, W   = shape

    # x_T approx N(0, I)
    x_t = torch.randn(n_samples, *shape, device=device)

    # Warm start EMG-VBA : un état par sample
    warm_starts = [None] * n_samples

    monitor_set = set(monitor_steps) if monitor_steps is not None else set()
    diagnostics = {'energie_par_step': {}}

    for t_val in tqdm(reversed(range(schedule.T)), total=schedule.T, desc="Sampling conditionnel"):
        t_batch     = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        alpha_bar_t = alphas_bar[t_val].item()

        # --- Étape 1 : Tweedie ---
        eps_pred = net(x_t, t_batch)
        xhat0    = tweedie_estimate(x_t, eps_pred, t_batch, alphas_bar, sigmas)

        # --- Étape 2 : EMG-VBA -> mu_post ---
        if t_val >= emg_skip_after:
            mu_post_list = []
            for b in range(n_samples):
                x_t_np   = x_t[b].cpu().numpy().ravel()
                xhat0_np = xhat0[b].cpu().numpy().ravel()

                mu_post, state = emg_vba_correction(
                    x_t_np, xhat0_np, alpha_bar_t, y_np, A_matrix,
                    n_iter=emg_n_iter,
                    a_0=a_0, b_0=b_0, c_0=c_0, d_0=d_0,
                    AtA=AtA, Aty=Aty, AtA_diag=AtA_diag,
                    warm_start=warm_starts[b],
                )
                warm_starts[b] = state
                if b == 0 and t_val in monitor_set:
                    diagnostics['energie_par_step'][t_val] = list(
                        state['historique']['energie_libre']
                    )
                mu_post_list.append(torch.from_numpy(mu_post.reshape(C, H, W)).float())

            mu_post_tensor = torch.stack(mu_post_list, dim=0).to(device)
        else:
            mu_post_tensor = xhat0  # pas de correction, mu_post = x_0 chapeau

        # --- Étape 3 : Pas reverse ---
        if t_val == 0:
            x_t = mu_post_tensor
        elif schedule.variant in ('ddpm', 'vpou'):
            x_t = reverse_step_ddpm(x_t, mu_post_tensor, t_val, alphas_bar,
                                     schedule.alphas.to(device), schedule.betas.to(device))
        elif schedule.variant == 've':
            x_t = reverse_step_ve(x_t, mu_post_tensor, t_val, sigmas)

    return (x_t.clamp(-1, 1) + 1) / 2, diagnostics