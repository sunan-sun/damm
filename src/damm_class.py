import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2, norm, multivariate_normal, invgamma
from scipy.special import logsumexp
from collections import OrderedDict

from .util import load_tools, plot_tools, quat_tools
from .gmm_class import GMM


def adjust_cov(cov, rel_scale=0.7, total_scale=1.5):
    vals, vecs = np.linalg.eigh(cov)
    print("Old eigenvalues:", vals)
    mean_val = np.mean(vals)
    new_vals = (1 - rel_scale) * vals + rel_scale * mean_val
    new_vals *= total_scale
    print("New eigenvalues:", new_vals)
    return vecs @ np.diag(new_vals) @ vecs.T


class DAMM:
    def __init__(self, x: np.ndarray, x_dir: np.ndarray, nu_0: float = 5, kappa_0: float = 1, psi_dir_0: float = 1):
        self.x = x
        self.x_dir = x_dir
        self.M, self.N = x.shape
        self.Psi_0 = np.eye(self.N)
        self.nu_0 = nu_0
        self.mu_0 = np.zeros(self.N)
        self.kappa_0 = kappa_0
        self.psi_dir_0 = psi_dir_0
        self.rng = np.random.default_rng()

    @staticmethod
    def pre_process(x: np.ndarray, x_dot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_dot_norm = np.linalg.norm(x_dot, axis=1)
        mask = x_dot_norm != 0
        x = x[mask]
        x_dot = x_dot[mask]
        x_dir = x_dot / x_dot_norm[mask][:, None]
        return x, x_dir

    def _reorder_assignments(self) -> None:
        z = self.z
        rearrange_list = []
        for i, label in enumerate(z):
            if label not in rearrange_list:
                rearrange_list.append(label)
                z[i] = len(rearrange_list) - 1
            else:
                z[i] = rearrange_list.index(label)
        self.z = z
        self.K = max(z) + 1

    def _update_index_lists(self) -> None:
        z = self.z
        self.index_lists = [[i for i in range(self.M) if z[i] == k] for k in range(self.K)]

    @staticmethod
    def sample_inverse_wishart(Psi: np.ndarray, nu: float) -> np.ndarray:
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

    @staticmethod
    def sample_multi_normal(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        A = np.linalg.cholesky(Sigma)
        z = np.random.randn(Sigma.shape[0])
        return mu + A @ z

    @staticmethod
    def sample_inv_gamma(psi: float, nu: float) -> float:
        alpha = nu / 2
        beta = (nu * psi) / 2
        return invgamma.rvs(a=alpha, scale=beta)

    @staticmethod
    def compute_niw_posterior(Psi_0, nu_0, mu_0, kappa_0, psi_dir_0, x_k, x_dot_k):
        M_k = x_k.shape[0]
        mean_ = np.mean(x_k, axis=0)
        x_k_mean = x_k - mean_
        scatter_ = x_k_mean.T @ x_k_mean
        Psi_n = Psi_0 + scatter_ + kappa_0 * M_k / (kappa_0 + M_k) * np.outer(mean_ - mu_0, mean_ - mu_0)
        nu_n = nu_0 + M_k
        mu_n = (kappa_0 * mu_0 + M_k * mean_) / (kappa_0 + M_k)
        kappa_n = kappa_0 + M_k
        mean_dir_ = quat_tools.karcher_mean(x_dot_k)
        scatter_dir_ = quat_tools.riem_scatter(mean_dir_, x_dot_k)
        psi_dir_n = (nu_0 * psi_dir_0 + scatter_dir_) / (nu_0 + M_k)
        return Psi_n, nu_n, mu_n, kappa_n, psi_dir_n, mean_dir_

    def _sample_parameters(self) -> None:
        Pi = np.array([self.rng.gamma(len(idx), 1) for idx in self.index_lists])
        gaussian_lists = []
        for k, idx in enumerate(self.index_lists):
            if len(idx) == 1:
                Psi_n, nu_n, mu_n, kappa_n, psi_dir_n, mu_dir_k = (
                    self.Psi_0, self.nu_0, self.mu_0, self.kappa_0, self.psi_dir_0, self.x_dir[idx[0]]
                )
            else:
                Psi_n, nu_n, mu_n, kappa_n, psi_dir_n, mu_dir_k = DAMM.compute_niw_posterior(
                    self.Psi_0, self.nu_0, self.mu_0, self.kappa_0, self.psi_dir_0,
                    self.x[idx], self.x_dir[idx]
                )
            Sigma_pos_k = DAMM.sample_inverse_wishart(Psi_n, nu_n)
            Sigma_pos_k = 0.5 * (Sigma_pos_k + Sigma_pos_k.T)
            mu_pos_k = DAMM.sample_multi_normal(mu_n, Sigma_pos_k / kappa_n)
            var_dir_k = DAMM.sample_inv_gamma(psi_dir_n, nu_n)
            mu_k = np.zeros(self.N + 1)
            mu_k[:self.N] = mu_pos_k
            Sigma_k = np.eye(self.N + 1)
            Sigma_k[:self.N, :self.N] = Sigma_pos_k
            Sigma_k[-1, -1] = var_dir_k
            gaussian_lists.append({
                "mu_dir": mu_dir_k,
                "rv": multivariate_normal(mu_k, Sigma_k, allow_singular=True)
            })
        Pi /= Pi.sum()
        self.Pi = Pi
        self.gaussian_lists = gaussian_lists

    def _sample_label(self) -> None:
        mu_dir = [g["mu_dir"] for g in self.gaussian_lists]
        x_dir_norm = [np.linalg.norm(quat_tools.riem_log(mu, self.x_dir), axis=1, keepdims=True) for mu in mu_dir]
        x_hat = [np.hstack((self.x, norm)) for norm in x_dir_norm]
        logProb = np.array([g['rv'].logpdf(xh) for g, xh in zip(self.gaussian_lists, x_hat)])
        logProb += np.log(self.Pi[:, None])
        postProb = np.exp(logProb - logsumexp(logProb, axis=0, keepdims=True))
        self.z = np.argmax(np.cumsum(postProb, axis=0) >= np.random.rand(self.M), axis=0)


    def compute_gamma(self, x: np.ndarray) -> np.ndarray:
        """
        Gamma is normalized logProb: K x M
        """
        logProb = np.array([g['rv'].logpdf(x) for g in self.gaussian_lists])
        if logProb.ndim == 1:
            logProb = logProb.reshape(-1, 1)

        logPrior = np.log([g['prior'] for g in self.gaussian_lists]).reshape(-1, 1)
        logProb += logPrior
        gamma = np.exp(logProb - logsumexp(logProb, axis=0, keepdims=True))
        return gamma

    def _split_proposal(self, index_list: list[int]) -> None:
        gmm = GMM(self.x[index_list])
        z_split = gmm.fit(init_cluster=2, T=100)
        if max(z_split) != 0:
            a = gmm.log_proposal_ratio() + gmm.log_target_ratio()
            if a > 0:
                self.z[index_list] = z_split + self.K
                print("Split accepted")

    @staticmethod
    def extract_gaussian(z: np.ndarray, x: np.ndarray) -> list:
        K = max(z) + 1
        M, N = x.shape
        Prior = [np.sum(z == k) / M for k in range(K)]
        Mu = np.array([np.mean(x[z == k], axis=0) for k in range(K)])
        # Sigma = np.array([np.cov(x[z == k].T) for k in range(K)])
        Sigma = np.array([adjust_cov(np.cov(x[z == k].T)) for k in range(K)])
        return Prior, Mu, Sigma, [
            {
                "prior": Prior[k],
                "mu": Mu[k],
                "sigma": Sigma[k],
                # "sigma": adjust_cov(Sigma[k]),
                "rv": multivariate_normal(Mu[k], Sigma[k], allow_singular=True)
            }
            for k in range(K)
        ]


    def fit(self, init_cluster: int = 10, T: int = 100) -> np.ndarray:
        self.z = self.rng.integers(0, init_cluster, size=self.M)
        for t in range(T):
            # print(t)
            self._reorder_assignments()
            self._update_index_lists()
            self._sample_parameters()
            self._sample_label()
            if t % 10 == 0 and t > 50:
                for idx in self.index_lists:
                    self._split_proposal(idx)
        self.Prior, self.Mu, self.Sigma, self.gaussian_lists = DAMM.extract_gaussian(self.z, self.x)

        return self.compute_gamma(self.x)
    


    def elasticUpdate(self, new_x, new_x_dot, new_gmm_struct):

        Prior  = new_gmm_struct["Prior"].tolist()
        Mu     = new_gmm_struct["Mu"]
        Sigma  = new_gmm_struct["Sigma"]
        K = len(Prior)

        gaussian_lists = []
        for k in range(K):
            gaussian_lists.append({   
                "prior" : Prior[k],
                "mu"    : Mu[k],
                "sigma" : Sigma[k],
                # "sigma" : adjust_cov(Sigma[k]),
                "rv"    : multivariate_normal(Mu[k], Sigma[k], allow_singular=True)
            })
        self.gaussian_lists = gaussian_lists

        gamma = self.compute_gamma(new_x)
        assignment_arr = np.argmax(gamma, axis = 0) # reverse order that we are assigning given the new gmm parameters; hence there's chance some component being empty
        unique_elements, counts = np.unique(assignment_arr, return_counts=True)
        for element, count in zip(unique_elements, counts):
            print("Current element", element)
            print("has number", count)

        # Remove Gaussians with count 0 or less than 10
        to_keep = [k for k, count in zip(unique_elements, counts) if count >= 15]
        removed_count = K - len(to_keep)
        if removed_count > 0:
            print(f"Removing {removed_count} Gaussian components with counts less than 10.")
            # Filter gaussian_lists
            self.gaussian_lists = [self.gaussian_lists[k] for k in to_keep]
            # Filter Prior, Mu, Sigma
            Prior = [Prior[k] for k in to_keep]
            Mu = Mu[to_keep]
            Sigma = Sigma[to_keep]
            K = len(to_keep)
            gamma = gamma[to_keep]

            # Update assignment_arr to map old indices to new indices
            mapping = {old_k: new_k for new_k, old_k in enumerate(to_keep)}
            assignment_arr = np.array([mapping.get(a, -1) for a in assignment_arr])
            # Reassign any assignments that mapped to -1 and gamma values accordingly
            if np.any(assignment_arr == -1):
                orphan_idx = np.where(assignment_arr == -1)[0]
                print(f"Reassigning {len(orphan_idx)} orphan points to nearest Gaussian.")
                x_orphans = new_x[orphan_idx]

                # Compute Mahalanobis distances for each orphan to all remaining components
                dists = np.zeros((len(x_orphans), K))
                for k in range(K):
                    mu_k = Mu[k]
                    sigma_k = Sigma[k]
                    inv_sigma = np.linalg.inv(sigma_k)
                    diff = x_orphans - mu_k
                    dists[:, k] = np.einsum('ij,ij->i', diff @ inv_sigma, diff)  # Mahalanobis

                nearest = np.argmin(dists, axis=1)
                assignment_arr[orphan_idx] = nearest

                gamma[:, orphan_idx] = 0.0
                for idx, comp in zip(orphan_idx, nearest):
                    gamma[comp, idx] = 1.0


        new_gmm_struct["Prior"] = Prior
        new_gmm_struct["Mu"] = Mu
        new_gmm_struct["Sigma"] = Sigma

        self.Prior = Prior
        self.Mu = Mu
        self.Sigma = Sigma
        self.K = K
        self.assignment_arr = assignment_arr


        return new_gmm_struct, self.assignment_arr, gamma
    

    @classmethod
    def from_existing(cls, new_x, new_x_dot, new_gmm_struct):
        obj = cls(new_x, new_x_dot)
        obj.Prior = np.copy(new_gmm_struct["Prior"])
        obj.Mu = np.copy(new_gmm_struct["Mu"])
        obj.Sigma = np.copy(new_gmm_struct["Sigma"])
        obj.gaussian_lists = [
            {"prior": obj.Prior[k],
            "mu": obj.Mu[k],
            "sigma": obj.Sigma[k],
            "rv": multivariate_normal(obj.Mu[k], obj.Sigma[k], allow_singular=True)}
            for k in range(len(obj.Prior))
        ]
        return obj

if __name__ == "__main__":
    input_message = '''
    Please choose a data input option:
    1. PC-GMM benchmark data
    2. LASA benchmark data
    3. Damm demo data
    4. DEMO
    Enter the corresponding option number: '''
    x, x_dot, _, _ = load_tools.load_data(1)

    x, x_dir = DAMM.pre_process(x, x_dot)

    damm = DAMM(x, x_dir, nu_0=5, kappa_0=1, psi_dir_0=1)
    gamma = damm.fit(init_cluster=30, T=100)

    plot_tools.plot_gmm(x, damm.z)
    plot_tools.plot_gamma(gamma)
    plt.show()


