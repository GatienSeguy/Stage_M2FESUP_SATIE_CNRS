import torch


def _default_device(dtype=torch.float32):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and dtype != torch.float64:
        return torch.device("mps")
    return torch.device("cpu")


class EMG_VBA:
    def __init__(self, op, y, alpha_bar_t, xt, xhat0,
                 a_0=1e-3, b_0=1e-3, c_0=1e-3, d_0=1e-3,
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

        # Hyperparamètres des priors Gamma (non-informatifs)
        self.a_0 = self._to_scalar(a_0)
        self.b_0 = self._to_scalar(b_0)
        self.c_0 = self._to_scalar(c_0)
        self.d_0 = self._to_scalar(d_0)

        # Précalculs (invariants au cours des itérations)
        self.Aty = self._to_tensor(Aty) if Aty is not None else op.adjoint(self.y)
        self.AtA_diag = self._to_tensor(AtA_diag)

    #  Utilitaires de conversion 

    def _to_tensor(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def _to_scalar(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.tensor(x, device=self.device, dtype=self.dtype)

    #  Initialisation 

    def initialiser(self, mu_init, Sigma_init):
        self.mu = self._to_tensor(mu_init).clone()
        self.Sigma = self._to_tensor(Sigma_init).clone()

    #  Étape 1 : Mise à jour de tau_r et tau_b 

    def mise_a_jour_taur_taub(self):
        # tau_b : précision du bruit de mesure (scalaire)
        self.a_tilde_b = self.m / 2.0 + self.c_0
        self._A_mu = self.op.forward(self.mu)
        res = self.y - self._A_mu
        self.b_tilde_b = (self.d_0
                          + 0.5 * (torch.dot(res, res)
                                   + torch.dot(self.AtA_diag, self.Sigma)))
        self.tau_b_moy = self.a_tilde_b / self.b_tilde_b

        # tau_r : précision du prior Tweedie (scalaire)
        self.a_tilde_r = self.n / 2.0 + self.a_0
        diff = self.mu - self.xhat0
        self.b_tilde_r = (self.b_0
                          + 0.5 * (torch.dot(diff, diff)
                                   + torch.sum(self.Sigma)))
        self.tau_r_moy = self.a_tilde_r / self.b_tilde_r

    #  Étape 2 : Distribution de référence q_r(x_0) 

    def calculer_distributions_reference_x0(self):
        abar = self.alpha_bar_t

        self.Sigma_r_inv = (abar / (1.0 - abar)
                            + self.tau_b_moy * self.AtA_diag
                            + self.tau_r_moy)
        self.Sigma_r = 1.0 / self.Sigma_r_inv

        AtA_mu = self.op.adjoint(self._A_mu)
        terme_mesure = self.Aty - AtA_mu + self.AtA_diag * self.mu

        self.mu_r = self.Sigma_r * (
            torch.sqrt(abar) / (1.0 - abar) * self.xt
            + self.tau_b_moy * terme_mesure
            + self.tau_r_moy * self.xhat0
        )

    #  Étape 3 : Pas sous-optimal s1 

    def calculer_pas_sousopt(self):
        abar = self.alpha_bar_t

        G_mu = (self.mu_r - self.mu) / self.Sigma_r
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

    #  Étape 4 : Mise à jour de q(x_0) 

    def mise_a_jour_x0(self, s1):
        Sigma_new_inv = (1 - s1) / self.Sigma + s1 / self.Sigma_r
        Sigma_new = 1.0 / Sigma_new_inv
        mu_new = Sigma_new * ((1 - s1) * self.mu / self.Sigma
                              + s1 * self.mu_r / self.Sigma_r)

        self.mu = mu_new
        self.Sigma = Sigma_new
        self._A_mu = self.op.forward(self.mu)

    #  Énergie libre négative F(q) 

    def calculer_energie_libre_negative(self):
        abar = self.alpha_bar_t
        tau_r = self.tau_r_moy
        tau_b = self.tau_b_moy

        log_tau_r = torch.special.digamma(self.a_tilde_r) - torch.log(self.b_tilde_r)
        log_tau_b = torch.special.digamma(self.a_tilde_b) - torch.log(self.b_tilde_b)

        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device, dtype=self.dtype))

        # Entropie de q(x_0)
        Entropie = -0.5 * torch.sum(torch.log(self.Sigma)) - 0.5 * self.n * (1.0 + log_2pi)

        # Terme forward p(x_t | x_0)
        forward_diff = self.xt - torch.sqrt(abar) * self.mu
        Forward = -0.5 / (1 - abar) * (torch.dot(forward_diff, forward_diff)
                                         + abar * torch.sum(self.Sigma))

        # Terme mesure p(y | x_0, tau_b)
        mesure_diff = self.y - self._A_mu
        Mesure = ((self.m / 2.0 + self.c_0 - 1.0) * log_tau_b
                  - (self.d_0 + 0.5 * (torch.dot(mesure_diff, mesure_diff)
                                        + torch.dot(self.AtA_diag, self.Sigma))) * tau_b)

        # Terme prior p(x_0 | xhat0, tau_r)
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

    #  Score conditionnel 

    def calculer_score_conditionnel(self):
        abar = self.alpha_bar_t
        return (torch.sqrt(abar) * self.mu - self.xt) / (1.0 - abar)

    #  Boucle principale 

    def executer(self, n_iter, mu_init, Sigma_init,
                 a_0_init=None, b_0_init=None, c_0_init=None, d_0_init=None,
                 verbose=False, affichage=False, tol=1e-7):
        self.initialiser(mu_init, Sigma_init)

        historique = {
            'energie_libre': [], 'log_jointe': [], 'entropie': [],
            'forward': [], 'mesure': [], 'a_priori': [],
            's1': [], 'tau_r': [], 'tau_b': [],
        }

        E_km1 = None
        for k in range(n_iter):

            self.mise_a_jour_taur_taub()



            self.calculer_distributions_reference_x0()


            s1 = self.calculer_pas_sousopt()


            self.mise_a_jour_x0(s1)

            result_nrj = self.calculer_energie_libre_negative()

            historique['energie_libre'].append(result_nrj['F'])
            historique['log_jointe'].append(result_nrj['log_jointe'])
            historique['entropie'].append(result_nrj['entropie'])
            historique['forward'].append(result_nrj['forward'])
            historique['mesure'].append(result_nrj['mesure'])
            historique['a_priori'].append(result_nrj['a_priori'])
            historique['s1'].append(s1.item())
            historique['tau_r'].append(self.tau_r_moy.item())
            historique['tau_b'].append(self.tau_b_moy.item())

            if verbose and (k % 10 == 0 or k == n_iter - 1):
                print(f"  k={k:3d} | F={result_nrj['F']:+.2f} | s1={s1.item():.4f} | "
                      f"sig2_r={1/self.tau_r_moy.item():.6f} | "
                      f"sig2_b={1/self.tau_b_moy.item():.6f}")

            # Critère d'arrêt sur la variation relative de F(q)
            E_k = result_nrj['F']
            if E_km1 is not None and abs(E_k - E_km1) / (abs(E_km1) + 1e-12) < tol:
                break
            E_km1 = E_k

        return {
            'mu': self.mu.clone(),
            'Sigma': self.Sigma.clone(),
            'tau_r': self.tau_r_moy,
            'tau_b': self.tau_b_moy,
            'score_conditionnel': self.calculer_score_conditionnel(),
            'historique': historique,
            'mu_r': self.mu_r.clone(),
            'Sigma_r': self.Sigma_r.clone(),
        }
