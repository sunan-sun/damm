import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools

from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2, norm, multivariate_normal
from scipy.special import logsumexp


class GMM:
    def __init__(self, x, init_cluster=10, Psi_0=None, nu_0=None, mu_0=None, kappa_0=None):
        self.x = x
        self.M, self.N = x.shape
        self.init_cluster = init_cluster
        self.rng = np.random.default_rng()
        self.z = self.rng.integers(0, self.init_cluster, size=self.M)

        self.Psi_0 = Psi_0 if Psi_0 is not None else np.eye(self.N)
        self.nu_0 = nu_0 if nu_0 is not None else 5
        self.mu_0 = mu_0 if mu_0 is not None else np.zeros((self.N,))
        self.kappa_0 = kappa_0 if kappa_0 is not None else 1

        self.index_lists = None
        self.logProb = None
        self.Pi = None

    def reorder_assignments(self, z):
        M = z.shape[0]
        rearrange_list = []
        for i in range(M):
            label = z[i]
            if label not in rearrange_list:
                rearrange_list.append(label)
                z[i] = len(rearrange_list) - 1
            else:
                z[i] = rearrange_list.index(label)
        K = max(z) + 1
        return z, K

    def update_index_lists(self, z):
        M = z.shape[0]
        K = max(z) + 1
        index_lists = [[] for _ in range(K)]
        for i in range(M):
            index_lists[z[i]].append(i)
        return index_lists

    def sample_inverse_wishart(self, Psi, nu):
        dim = Psi.shape[0]
        L = np.linalg.cholesky(Psi)
        A = np.zeros((dim, dim))
        for i in range(dim):
            A[i, i] = np.sqrt(chi2.rvs(df=nu - i))
            for j in range(i):
                A[i, j] = norm.rvs()
        AA_T = A @ A.T
        inv_AA_T = np.linalg.inv(AA_T)
        return L @ inv_AA_T @ L.T

    def sample_multi_normal(self, mu, Sigma):
        M = Sigma.shape[0]
        A = np.linalg.cholesky(Sigma)
        z = np.random.randn(M)
        x = mu + A @ z
        return x

    def compute_niw_posterior(self, Psi_0, nu_0, mu_0, kappa_0, x_k):
        M_k, _ = x_k.shape
        mean_ = np.mean(x_k, axis=0)
        x_k_mean = x_k - mean_
        scatter_ = x_k_mean.T @ x_k_mean
        Psi_n   = Psi_0 + scatter_ + kappa_0*M_k/(kappa_0+M_k)*np.outer(mean_-mu_0, mean_-mu_0)
        nu_n    = nu_0 + M_k
        mu_n    = (kappa_0*mu_0 + M_k*mean_)/(kappa_0+M_k)
        kappa_n = kappa_0 + M_k
        return Psi_n, nu_n, mu_n, kappa_n

    def sample_label(self, x, Pi, gaussian_lists):
        K = len(gaussian_lists)
        logProb =  np.array([gaussian_lists[k].logpdf(x) for k in range(K)])
        logProb += np.log(Pi.reshape(-1, 1))
        log_denom = logsumexp(logProb, axis=0, keepdims=True)
        postProb = np.exp(logProb - log_denom)
        prob_cumsum = np.cumsum(postProb, axis=0)
        uniform_draws = np.random.rand(x.shape[0])
        z = np.argmax(prob_cumsum >= uniform_draws, axis=0)
        self.logProb = logProb
        return z

    def log_proposal_ratio(self):
        Pi = self.Pi
        logProb = self.logProb
        index_lists = self.index_lists

        logProposalRatio = 0.0
        for k in range(2):
            idx = index_lists[k]
            # logsumexp over all clusters for each data point in cluster k
            log_sum = logsumexp(logProb[:, idx], axis=0)
            # Subtract log Pi[k] and logProb[k, idx] for each point
            logProposalRatio += np.sum(log_sum - (np.log(Pi[k]) + logProb[k, idx]))

        return logProposalRatio


    def log_target_ratio(self):
        Pi = self.Pi
        logProb = self.logProb
        index_lists = self.index_lists

        logTargetRatio = 0.0
        for k in range(2):
            idx = index_lists[k]
            logTargetRatio += np.sum((np.log(Pi[k]) + logProb[k, idx]))


        index_lists_combined = index_lists[0] + index_lists[1]
        Psi_n, nu_n, mu_n, kappa_n = self.compute_niw_posterior(self.Psi_0, self.nu_0, self.mu_0, self.kappa_0, self.x[index_lists_combined])
        Sigma_k = self.sample_inverse_wishart(Psi_n, nu_n)
        Sigma_k = 0.5 * (Sigma_k + Sigma_k.T)
        mu_k = self.sample_multi_normal(mu_n, Sigma_k/kappa_n)
        gaussian_combined = multivariate_normal(mu_k, Sigma_k, allow_singular=True)

        logTargetRatio -= np.sum(gaussian_combined.logpdf(self.x[index_lists_combined]))


        return logTargetRatio




    def fit(self, T=100):
        z = self.z
        for t in range(T):
            z, K = self.reorder_assignments(z)
            self.index_lists = self.update_index_lists(z)
            Pi = np.zeros((K, ))
            gaussian_lists = []
            for k in range(K):
                Pi[k] = self.rng.gamma(len(self.index_lists[k]), 1)
                Psi_n, nu_n, mu_n, kappa_n = self.compute_niw_posterior(self.Psi_0, self.nu_0, self.mu_0, self.kappa_0, self.x[self.index_lists[k]])
                Sigma_k = self.sample_inverse_wishart(Psi_n, nu_n)
                Sigma_k = 0.5 * (Sigma_k + Sigma_k.T)
                mu_k = self.sample_multi_normal(mu_n, Sigma_k/kappa_n)
                gaussian_lists.append(multivariate_normal(mu_k, Sigma_k, allow_singular=True))
            Pi /= np.sum(Pi)
            self.Pi = Pi
            z = self.sample_label(self.x, Pi, gaussian_lists)
        self.z = z
        return z





input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. Damm demo data
4. DEMO
Enter the corresponding option number: '''

x, _, _, _ = load_tools.load_data(int(1))

gmm = GMM(x)
z = gmm.fit()

plot_tools.plot_gmm(x, z)

plt.show()


