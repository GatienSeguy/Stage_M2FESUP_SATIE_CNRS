import torch


def forward_process(x0, t, alphas_bar, sigmas):
    alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)
    sigma_t = sigmas[t].view(-1, 1, 1, 1)
    sqrt_alpha = torch.sqrt(alpha_bar_t)
    epsilon = torch.randn_like(x0)
    x_t = sqrt_alpha * x0 + sigma_t * epsilon
    
    return(x_t, epsilon)


def tweedie_estimate(x_t, eps_pred, t, alphas_bar, sigmas):
    if isinstance(t, int):
        alpha_bar_t = alphas_bar[t]
        sigma_t = sigmas[t]
    
    else:
        alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)
        sigma_t = sigmas[t].view(-1, 1, 1, 1)

    sqrt_alpha = torch.sqrt(alpha_bar_t)
    return((x_t - sigma_t * eps_pred) / sqrt_alpha)