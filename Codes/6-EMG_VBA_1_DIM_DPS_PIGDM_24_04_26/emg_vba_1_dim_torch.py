import torch
import matplotlib.pyplot as plt


def _default_device(dtype=torch.float32):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and dtype != torch.float64:
        return torch.device("mps")
    return torch.device("cpu")


class EMG_VBA:
    def __init__(self, op, y, alpha_bar_t, xt, xhat0, a_0, b_0, c_0, d_0,
                 Aty=None, AtA_diag=None, device=None, dtype=torch.float32):
        self.device = device if device is not None else _default_device(dtype)
        self.dtype = dtype

        self.op = op

        self.y = self._to_tensor(y)

        self.alpha_bar_t = self._to_scalar(alpha_bar_t)

        self.xt = self._to_tensor(xt)

        self.xhat0 = self._to_tensor(xhat0)

        self.n = self.xhat0.numel()

        self.m = self.y.numel()

        self.a_0 = self._to_scalar(a_0)
        self.b_0 = self._to_scalar(b_0)
        self.c_0 = self._to_scalar(c_0)
        self.d_0 = self._to_scalar(d_0)

        # Précalculs
        self.Aty      = self._to_tensor(Aty) if Aty is not None else op.adjoint(self.y)
        self.AtA_diag = self._to_tensor(AtA_diag)


    def _to_tensor(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device).to(dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def _to_scalar(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device).to(dtype=self.dtype)
        return torch.tensor(x, device=self.device, dtype=self.dtype)


    def initialiser(self, mu_init, Sigma_init, a_0_init, b_0_init, c_0_init, d_0_init):
        self.mu    = self._to_tensor(mu_init).clone()
        self.Sigma = self._to_tensor(Sigma_init).clone()

        self.mu_k_1    = self.mu.clone()
        self.Sigma_k_1 = self.Sigma.clone()


    def mise_a_jour_taur_taub(self):
        ## q^r(tau_b)
        self.a_tilde_b = self.m / 2.0 + self.c_0                          # (75)
        self._A_mu = self.op.forward(self.mu)
        res = self.y - self._A_mu

        self.b_tilde_b = (self.d_0
                          + 0.5 * (torch.dot(res, res)
                                   + torch.dot(self.AtA_diag, self.Sigma)))  # (76)

        ## q^r(tau_r)
        self.a_tilde_r = self.n / 2.0 + self.a_0                          # (84)
        diff = self.mu - self.xhat0
        self.b_tilde_r = (self.b_0
                          + 0.5 * (torch.dot(diff, diff)
                                   + torch.sum(self.Sigma)))                # (85)

        # Moyennes
        self.tau_r_moy = self.a_tilde_r / self.b_tilde_r
        self.tau_b_moy = self.a_tilde_b / self.b_tilde_b


    def calculer_distributions_reference_x0(self):
        abar = self.alpha_bar_t

        self.Sigma_r_inv = (abar / (1.0 - abar)
                            + self.tau_b_moy * self.AtA_diag
                            + self.tau_r_moy)                              # (94)
        self.Sigma_r = 1.0 / self.Sigma_r_inv

        AtA_mu = self.op.adjoint(self._A_mu)
        terme_mesure = self.Aty - AtA_mu + self.AtA_diag * self.mu         # (95)

        self.mu_r = self.Sigma_r * (
            torch.sqrt(abar) / (1.0 - abar) * self.xt
            + self.tau_b_moy * terme_mesure
            + self.tau_r_moy * self.xhat0
        )


    def calculer_pas_sousopt(self):
        abar = self.alpha_bar_t

        G_mu    = (self.mu_r - self.mu) / self.Sigma_r
        G_Sigma = 0.5 * (1.0 / self.Sigma - 1.0 / self.Sigma_r)

        d1_Sigma = self.Sigma - self.Sigma**2 / self.Sigma_r
        d1_mu = (self.Sigma / self.Sigma_r) * (self.mu_r - self.mu)

        grad_g1 = torch.dot(G_mu, d1_mu) + torch.dot(G_Sigma, d1_Sigma)

        d11_Sigma = 2.0 * d1_Sigma**2 / self.Sigma
        d11_mu = 2.0 * d1_Sigma * d1_mu / self.Sigma

        coeff = abar / (1.0 - abar) + self.tau_r_moy
        A_d1_mu = self.op.forward(d1_mu)
        terme_M = (coeff * torch.dot(d1_mu, d1_mu)
                   + self.tau_b_moy * torch.dot(A_d1_mu, A_d1_mu))

        terme_D = 0.5 * torch.sum(d1_Sigma**2 / self.Sigma**2)

        H11 = (torch.dot(G_mu, d11_mu)
               + torch.dot(G_Sigma, d11_Sigma)
               - terme_M - terme_D)

        s1 = -grad_g1 / H11
        return torch.clamp(s1, 0.0, 1.0)


    def mise_a_jour_x0(self, s1):
        self.mu_k_1    = self.mu.clone()
        self.Sigma_k_1 = self.Sigma.clone()

        Sigma_new_inv = (1 - s1) / self.Sigma + s1 / self.Sigma_r         # (115)
        Sigma_new = 1.0 / Sigma_new_inv
        mu_new = Sigma_new * ((1 - s1) * self.mu / self.Sigma
                              + s1 * self.mu_r / self.Sigma_r)             # (116)

        self.mu    = mu_new
        self.Sigma = Sigma_new
        self._A_mu = self.op.forward(self.mu)                              # rafraîchir le cache


    def calculer_energie_libre_negative(self):
        abar = self.alpha_bar_t
        tau_r = self.tau_r_moy
        tau_b = self.tau_b_moy

        log_tau_r = torch.special.digamma(self.a_tilde_r) - torch.log(self.b_tilde_r)
        log_tau_b = torch.special.digamma(self.a_tilde_b) - torch.log(self.b_tilde_b)

        ## Entropie — (122)
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device, dtype=self.dtype))
        Entropie = -0.5 * torch.sum(torch.log(self.Sigma)) - 0.5 * self.n * (1.0 + log_2pi)

        ## Forward — (123)
        forward_diff = self.xt - torch.sqrt(abar) * self.mu
        Forward = -0.5 / (1 - abar) * (torch.dot(forward_diff, forward_diff)
                                         + abar * torch.sum(self.Sigma))

        ## Mesure — (123)
        mesure_diff = self.y - self._A_mu
        Mesure = ((self.m / 2.0 + self.c_0 - 1.0) * log_tau_b
                  - (self.d_0 + 0.5 * (torch.dot(mesure_diff, mesure_diff)
                                        + torch.dot(self.AtA_diag, self.Sigma))) * tau_b)

        ## A priori
        a_priori_diff = self.mu - self.xhat0
        A_priori = ((self.n / 2.0 + self.a_0 - 1.0) * log_tau_r
                    - (self.b_0 + 0.5 * (torch.dot(a_priori_diff, a_priori_diff)
                                          + torch.sum(self.Sigma))) * tau_r)

        Log_jointe = Forward + Mesure + A_priori
        F = Log_jointe - Entropie

        return {
            'F': F.item(),
            'log_jointe': Log_jointe.item(),
            'entropie': Entropie.item(),
            'forward': Forward.item(),
            'mesure': Mesure.item(),
            'a_priori': A_priori.item(),
        }


    def calculer_score_conditionnel(self):
        abar = self.alpha_bar_t
        return (torch.sqrt(abar) * self.mu - self.xt) / (1.0 - abar)


    def executer(self, n_iter, mu_init, Sigma_init, a_0_init, b_0_init, c_0_init, d_0_init,
                 verbose=True, affichage=True):

        self.initialiser(mu_init, Sigma_init, a_0_init, b_0_init, c_0_init, d_0_init)

        historique = {
            'energie_libre': [], 'log_jointe': [], 'entropie': [],
            'forward': [], 'mesure': [], 'a_priori': [],
            's1': [], 'tau_r': [], 'tau_b': [],
        }

        E_km1 = None
        for k in range(n_iter):
            # Étape 1
            self.mise_a_jour_taur_taub()

            # Étape 2
            self.calculer_distributions_reference_x0()

            # Étape 3
            s1 = self.calculer_pas_sousopt()

            # Étape 4
            self.mise_a_jour_x0(s1)

            # Calcul NRJ libre negative
            result_nrj = self.calculer_energie_libre_negative()

            historique['energie_libre'].append(result_nrj['F'])
            historique['log_jointe'].append(result_nrj['log_jointe'])
            historique['entropie'].append(result_nrj['entropie'])
            historique['forward'].append(result_nrj['forward'])
            historique['mesure'].append(result_nrj['mesure'])
            historique['a_priori'].append(result_nrj['a_priori'])

            s1_val = s1.item()
            historique['s1'].append(s1_val)
            historique['tau_r'].append(self.tau_r_moy.item())
            historique['tau_b'].append(self.tau_b_moy.item())

            if verbose and (k % 10 == 0 or k == n_iter - 1):
                print(f"  k={k:3d} | NRJ libre negative={result_nrj['F']:+.2f} | s1={s1_val:.4f} | "
                      f"r2_t={1/historique['tau_r'][-1]:.6f} | "
                      f"sig2_b={1/historique['tau_b'][-1]:.6f}")

            # Critère d'arrêt
            E_k = result_nrj['F']
            print("E_k : ", E_k)
            print("E_k-1 : ", E_km1)
            if E_km1 is not None:
                print("difference absolue : ", abs(E_k - E_km1) / (abs(E_km1) + 1e-12))
            print(k)

            if E_km1 is not None and abs(E_k - E_km1) / (abs(E_km1) + 1e-12) < 1e-5:
                # print("Critère d'arrêt : ",abs(E_k - E_km1) / (abs(E_km1) + 1e-12))
                break
            E_km1 = E_k

        if affichage:
            plt.plot(historique['energie_libre'])
            plt.title("Énergie libre négative en fonction des itérations")
            plt.xlabel("#itération EMG-VBA")
            plt.ylabel("Énergie libre négative")

        return {
            'mu': self.mu.clone(),
            'Sigma': self.Sigma.clone(),
            'tau_r': self.tau_r_moy,
            'tau_b': self.tau_b_moy,
            'score_conditionnel': self.calculer_score_conditionnel(),
            'historique': historique,
        }