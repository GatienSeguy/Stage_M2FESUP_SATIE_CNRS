import numpy as np
from scipy.special import digamma
import matplotlib.pyplot as plt

class EMG_VBA:
    def __init__(self, op, y, alpha_bar_t, xt, xhat0, a_0, b_0, c_0, d_0, Aty=None, AtA_diag=None):
        self.op = op
        self.y = y

        self.y = y

        self.alpha_bar_t = alpha_bar_t

        self.xt = xt

        self.xhat0 = xhat0

        self.n = len(xhat0)

        self.m = len(y)

        self.a_0, self.b_0 = a_0, b_0

        self.c_0, self.d_0 = c_0, d_0

        # Précalculs
        self.Aty      = Aty if Aty is not None else op.adjoint(y)
        self.AtA_diag = AtA_diag    # diag(A^T A), précalculé à l'extérieur
 


    def initialiser(self, mu_init, Sigma_init, a_0_init, b_0_init, c_0_init, d_0_init):

        # q(x_{0,i}) — (51)
        self.mu    = np.array(mu_init, dtype=float).copy()
        self.Sigma = np.array(Sigma_init, dtype=float).copy()

        # Sauvegarde k−1 (inutile car s2 = 0,  POUR APRÈS)
        self.mu_k_1    = self.mu.copy()
        self.Sigma_k_1 = self.Sigma.copy()


    def mise_a_jour_taur_taub(self):
        ## q^r(tau_b)
        self.a_tilde_b = self.m / 2.0 + self.c_0 #(75)

        res = self.y - self.op.forward(self.mu)

        self.b_tilde_b = (self.d_0 + 0.5 * (np.dot(res, res) + np.dot(self.AtA_diag, self.Sigma))) # (76)

        ##  q^r(tau_r)
        self.a_tilde_r = self.n / 2.0 + self.a_0 #(84)

        diff = self.mu - self.xhat0

        self.b_tilde_r = (self.b_0 + 0.5 * (np.dot(diff, diff) + np.sum(self.Sigma))) #(85)

        #Moyennes
        self.tau_r_moy = self.a_tilde_r / self.b_tilde_r 
        self.tau_b_moy = self.a_tilde_b / self.b_tilde_b 


    def calculer_distributions_reference_x0(self):
        abar = self.alpha_bar_t

        #  q^r(x_{0,i}) 
        self.Sigma_r_inv = (abar / (1.0 - abar) + self.tau_b_moy * self.AtA_diag + self.tau_r_moy)  #(94)
    
        self.Sigma_r = 1.0 / self.Sigma_r_inv

        AtA_mu = self.op.adjoint(self.op.forward(self.mu))
        terme_mesure = self.Aty - AtA_mu + self.AtA_diag * self.mu #(95)

        self.mu_r = self.Sigma_r * (
            np.sqrt(abar) / (1.0 - abar) * self.xt
            + self.tau_b_moy * terme_mesure
            + self.tau_r_moy * self.xhat0
        )


    def calculer_pas_sousopt(self):
        abar = self.alpha_bar_t

        # G_{mu,i}  et  G_{sigma,i} - (128 / 129)
        G_mu    = (self.mu_r - self.mu) / self.Sigma_r

        G_Sigma = 0.5 * (1.0 / self.Sigma - 1.0 / self.Sigma_r)

        # Dérivées d'ordre 1 par rapport à s1 - (124)
        d1_Sigma = self.Sigma - self.Sigma**2 / self.Sigma_r

        d1_mu = (self.Sigma / self.Sigma_r) * (self.mu_r - self.mu)

        ##############
        #METTRE DERIVEES ORDRE 1 PAR RAPPORT A S2 APRÈS 
        ##############


        ## Gradient  - (124)
        grad_g1 = np.dot(G_mu, d1_mu) + np.dot(G_Sigma, d1_Sigma)

        ##############
        #METTRE GRADIENT PAR RAPPORT A S2 APRÈS 
        ##############

        # Dérivées d'ordre 2 par rapport a s1 - (125)
        d11_Sigma = 2.0 * d1_Sigma**2 / self.Sigma

        d11_mu = 2.0 * d1_Sigma * d1_mu / self.Sigma

        ##############
        #METTRE DERIVEES ORDRE 2 PAR RAPPORT A S1 et S2 APRÈS 
        ##############

        ## Hessienne - (138)
        # Terme M
        coeff = abar / (1.0 - abar) + self.tau_r_moy

        A_d1_mu = self.op.forward(d1_mu) 
        
        terme_M = (coeff * np.dot(d1_mu, d1_mu) + self.tau_b_moy * np.dot(A_d1_mu, A_d1_mu))
        
        # Terme D
        terme_D = 0.5 * np.sum(d1_Sigma**2 / self.Sigma**2)

        # Hessienne totale
        H11 = (np.dot(G_mu,d11_mu)+ np.dot(G_Sigma,d11_Sigma) - terme_M - terme_D)

        ## Pas sous-optimal

        s1 = -grad_g1 / H11

        return(np.clip(s1,0.0,1.0))
    

    def mise_a_jour_x0(self,s1):
        # maj de q(x0)
        self.mu_k_1   = self.mu.copy()

        self.Sigma_k_1 = self.Sigma.copy()

        Sigma_new_inv = (1 - s1)/self.Sigma + s1/ self.Sigma_r #(115)

        Sigma_new = 1/ Sigma_new_inv

        mu_new = Sigma_new * (( 1 - s1 )*self.mu / self.Sigma + s1 * self.mu_r/self.Sigma_r) #(116)

        self.mu    = mu_new

        self.Sigma = Sigma_new


    def calculer_energie_libre_negative(self):
        abar = self.alpha_bar_t

        tau_r = self.tau_r_moy

        tau_b = self.tau_b_moy

        log_tau_r = digamma(self.a_tilde_r) - np.log(self.b_tilde_r) 

        log_tau_b = digamma(self.a_tilde_b) - np.log(self.b_tilde_b)

        ## Entropie - (122)  E_q[log q]
        Entropie = -0.5 * np.sum(np.log(self.Sigma)) - 0.5 * self.n * (1.0 + np.log(2.0 * np.pi))  

        
        ## Terme log jointe
        # Forward - (123)
        forward_diff = self.xt - np.sqrt(abar)*self.mu

        Forward = -0.5/(1 - abar) * (np.dot(forward_diff,forward_diff) + abar*np.sum(self.Sigma)) 

        ## Mesure - (123)
        mesure_diff = self.y - self.op.forward(self.mu)   

        Mesure = ((self.m / 2.0 + self.c_0 - 1.0) * log_tau_b - (self.d_0 + 0.5 * (np.dot(mesure_diff, mesure_diff) + np.dot(self.AtA_diag, self.Sigma))) * tau_b)
        

        #A priori
        a_priori_diff = self.mu - self.xhat0

        A_priori = ((self.n / 2.0 + self.a_0 - 1.0) * log_tau_r - (self.b_0 + 0.5 * (np.dot(a_priori_diff, a_priori_diff) + np.sum(self.Sigma))) * tau_r)
        
        Log_jointe = Forward + Mesure + A_priori

        Energie_libre_negative = Log_jointe - Entropie

        return(Energie_libre_negative)
    

    def calculer_score_conditionnel(self):

        abar = self.alpha_bar_t

        return( (np.sqrt(abar) * self.mu - self.xt) / (1.0 - abar))
    

    def executer(self, n_iter, mu_init, Sigma_init, a_0_init, b_0_init, c_0_init, d_0_init, verbose=True, affichage = True):

        self.initialiser(mu_init, Sigma_init, a_0_init, b_0_init, c_0_init, d_0_init)

        historique = {
            'energie_libre': [],
            's1':            [],
            'tau_r':         [],
            'tau_b':         [],
        }

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
            Energie_libre_negative = self.calculer_energie_libre_negative()

            historique['energie_libre'].append(Energie_libre_negative)
            historique['s1'].append(s1)
            historique['tau_r'].append(self.tau_r_moy)
            historique['tau_b'].append(self.tau_b_moy)

            if verbose and (k % 10 == 0 or k == n_iter - 1):
                print(f"  k={k:3d} | NRJ libre negative={Energie_libre_negative:+.2f} | s1={s1:.4f} | "
                      f"r2_t={1/self.tau_r_moy:.6f} | "
                      f"sig2_b={1/self.tau_b_moy:.6f}")
            
            if k > 0 and s1 < 1e-6:
                break

        if affichage == True:
            plt.plot(historique['energie_libre'])  
            plt.title("Énergie libre négative en fonction des itérations")
            plt.xlabel("#itération EMG-VBA")
            plt.ylabel("Énergie libre négative")

        return {
            'mu':                  self.mu.copy(),
            'Sigma':               self.Sigma.copy(),
            'tau_r':               self.tau_r_moy,
            'tau_b':               self.tau_b_moy,
            'score_conditionnel':  self.calculer_score_conditionnel(),
            'historique':          historique,
        }