import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools, quat_tools

from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2, norm, multivariate_normal, invgamma
from scipy.special import logsumexp


class DAMM:
    def __init__(self, init_cluster: int = 10, T: int = 100, nu_0: float = 5, kappa_0: float = 1, psi_dir_0: float = 1):
        self.init_cluster = init_cluster
        self.T = T
        self.nu_0 = nu_0
        self.kappa_0 = kappa_0
        self.psi_dir_0 = psi_dir_0
        self.Psi_0 = None
        self.mu_0 = None
        self.x = None
        self.x_dir = None
        self.M = None
        self.N = None
        self.z = None
        self.rng = np.random.default_rng()

    @staticmethod
    def pre_process(x: np.ndarray, x_dot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove zero-norm rows and normalize x_dot to get direction."""
        x_dot_norm = np.linalg.norm(x_dot, axis=1)
        mask = x_dot_norm != 0
        x = x[mask]
        x_dot = x_dot[mask]
        x_dir = x_dot / x_dot_norm[mask].reshape(-1, 1)
        return x, x_dir

    def reorder_assignments(self, z: np.ndarray) -> tuple[np.ndarray, int]:
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

    def update_index_lists(self, z: np.ndarray) -> list:
        M = z.shape[0]
        K = max(z) + 1
        index_lists = [[] for _ in range(K)]
        for i in range(M):
            index_lists[z[i]].append(i)
        return index_lists

    def sample_inverse_wishart(self, Psi: np.ndarray, nu: float) -> np.ndarray:
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

    def sample_multi_normal(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        M = Sigma.shape[0]
        A = np.linalg.cholesky(Sigma)
        z = np.random.randn(M)
        x = mu + A @ z
        return x

    def sample_inv_gamma(self, psi: float, nu: float) -> float:
        alpha = nu / 2
        beta = (nu * psi) / 2
        var = invgamma.rvs(a=alpha, scale=beta)
        return var

    def compute_niw_posterior(self, Psi_0, nu_0, mu_0, kappa_0, psi_dir_0, x_k, x_dot_k):
        M_k, _ = x_k.shape
        mean_ = np.mean(x_k, axis=0)
        x_k_mean = x_k - mean_
        scatter_ = x_k_mean.T @ x_k_mean
        Psi_n   = Psi_0 + scatter_ + kappa_0*M_k/(kappa_0+M_k)*np.outer(mean_-mu_0, mean_-mu_0)
        nu_n    = nu_0 + M_k
        mu_n    = (kappa_0*mu_0 + M_k*mean_)/(kappa_0+M_k)
        kappa_n = kappa_0 + M_k
        mean_dir_ = quat_tools.karcher_mean(x_dot_k)
        scatter_dir_ = quat_tools.riem_scatter(mean_dir_, x_dot_k)
        psi_dir_n    = (nu_0 * psi_dir_0 + scatter_dir_)/(nu_0+M_k)
        return Psi_n, nu_n, mu_n, kappa_n, psi_dir_n, mean_dir_

    def sample_label(self, x, x_dir, Pi, gaussian_lists):
        K = len(gaussian_lists)
        m, n = x.shape
        mu_dir = [gaussian_lists[k]["mu_dir"] for k in range(K)]
        x_dir_norm = [np.linalg.norm(quat_tools.riem_log(mu_dir[k], x_dir), axis=1, keepdims=True) for k in range(K)]
        x_hat = [np.hstack((x, x_dir_norm[k])) for k in range(K)]
        logProb =  np.array([gaussian_lists[k]['rv'].logpdf(x_hat[k]) for k in range(K)])
        logProb += np.log(Pi.reshape(-1, 1))
        log_denom = logsumexp(logProb, axis=0, keepdims=True)
        postProb = np.exp(logProb - log_denom)
        prob_cumsum = np.cumsum(postProb, axis=0)
        uniform_draws = np.random.rand(x.shape[0])
        z = np.argmax(prob_cumsum >= uniform_draws, axis=0)
        return z

    def fit(self, x: np.ndarray, x_dir: np.ndarray) -> np.ndarray:
        """Fit the DAMM model to the data."""
        self.x = x
        self.x_dir = x_dir
        self.M, self.N = x.shape
        self.Psi_0 = np.eye(self.N)
        self.mu_0 = np.zeros((self.N,))
        z = self.rng.integers(0, self.init_cluster, size=self.M)
        for t in range(self.T):
            z, K = self.reorder_assignments(z)
            index_lists = self.update_index_lists(z)
            Pi = np.zeros((K, ))
            gaussian_lists = []
            for k in range(K):
                Pi[k] = self.rng.gamma(len(index_lists[k]), 1)
                Psi_n, nu_n, mu_n, kappa_n, psi_dir_n, mu_dir_k = self.compute_niw_posterior(
                    self.Psi_0, self.nu_0, self.mu_0, self.kappa_0, self.psi_dir_0, x[index_lists[k]], x_dir[index_lists[k]])
                Sigma_pos_k = self.sample_inverse_wishart(Psi_n, nu_n)
                Sigma_pos_k = 0.5 * (Sigma_pos_k + Sigma_pos_k.T)
                mu_pos_k = self.sample_multi_normal(mu_n, Sigma_pos_k/kappa_n)
                var_dir_k = self.sample_inv_gamma(psi_dir_n, nu_n)
                mu_k = np.zeros((self.N+1, ))
                mu_k[:self.N] = mu_pos_k
                Sigma_k = np.eye(self.N+1)
                Sigma_k[:self.N, :self.N] = Sigma_pos_k
                Sigma_k[-1, -1] = var_dir_k
                gaussian_lists.append({
                    "mu_dir": mu_dir_k,
                    "rv": multivariate_normal(mu_k, Sigma_k, allow_singular=True)
                })
            Pi /= np.sum(Pi)
            z = self.sample_label(x, x_dir, Pi, gaussian_lists)
        self.z = z
        return z

if __name__ == "__main__":
    input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. Damm demo data
4. DEMO
Enter the corresponding option number: '''

    x, x_dot, _, _ = load_tools.load_data(int(1))
    x, x_dir = DAMM.pre_process(x, x_dot)

    damm = DAMM(init_cluster=10, T=100, nu_0=5, kappa_0=1, psi_dir_0=1)
    z = damm.fit(x, x_dir)

    plot_tools.plot_gmm(x, z)
    plt.show()


