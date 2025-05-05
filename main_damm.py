import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools, quat_tools

from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2, norm, multivariate_normal, invgamma
from scipy.special import logsumexp


def pre_process(x, x_dot):

    x_dot_norm = np.linalg.norm(x_dot, axis=1)

    x     = x[x_dot_norm!=0]
    x_dot = x_dot[x_dot_norm!=0]

    x_dir = x_dot / x_dot_norm[x_dot_norm!=0].reshape(-1, 1)
    
    # return np.hstack((x, x_dir)) 
    return x, x_dir


def reorder_assignments(z):
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


def update_index_lists(z):
    M = z.shape[0]
    K = max(z) + 1
    index_lists = [[] for _ in range(K)]
    for i in range(M):
        index_lists[z[i]].append(i)
    return index_lists 


def sample_inverse_wishart(Psi, nu):
    dim = Psi.shape[0]
    L = np.linalg.cholesky(Psi)  # Psi = L L^T
    A = np.zeros((dim, dim))
    for i in range(dim):
        A[i, i] = np.sqrt(chi2.rvs(df=nu - i))
        for j in range(i):
            A[i, j] = norm.rvs()
    AA_T = A @ A.T
    inv_AA_T = np.linalg.inv(AA_T)
    return L @ inv_AA_T @ L.T


def sample_multi_normal(mu, Sigma):
    M = Sigma.shape[0]
    A = np.linalg.cholesky(Sigma)  # Sigma = L L^T
    z = np.random.randn(M)
    x = mu + A @ z
    return x


def sample_inv_gamma(psi, nu):
    alpha = nu / 2
    beta = (nu * psi) / 2
    var = invgamma.rvs(a=alpha, scale=beta)

    return var


def compute_niw_posterior(Psi_0, nu_0, mu_0, kappa_0, psi_dir_0, x_k, x_dot_k):
    M_k, _ = x_k.shape

    mean_ = np.mean(x_k, axis=0)  
    x_k_mean = x_k - mean_        # broadcasting subtracts row-wise
    scatter_ = x_k_mean.T @ x_k_mean

    Psi_n   = Psi_0 + scatter_ + kappa_0*M_k/(kappa_0+M_k)*np.outer(mean_-mu_0, mean_-mu_0)
    nu_n    = nu_0 + M_k
    mu_n    = (kappa_0*mu_0 + M_k*mean_)/(kappa_0+M_k)
    kappa_n = kappa_0 + M_k
    
    mean_dir_ = quat_tools.karcher_mean(x_dot_k)

    scatter_dir_ = quat_tools.riem_scatter(mean_dir_, x_dot_k)
    psi_dir_n    = (nu_0 * psi_dir_0 + scatter_dir_)/(nu_0+M_k)

    return Psi_n, nu_n, mu_n, kappa_n, psi_dir_n, mean_dir_


def sample_label(x, x_dir, Pi, gaussian_lists): 
    K = len(gaussian_lists)
    m, n = x.shape
    mu_dir = [gaussian_lists[k]["mu_dir"] for k in range(K)]

    x_dir_norm = [np.linalg.norm(quat_tools.riem_log(mu_dir[k], x_dir), axis=1, keepdims=True) for k in range(K)]
    x_hat = [np.hstack((x, x_dir_norm[k])) for k in range(K)]

    logProb =  np.array([gaussian_lists[k]['rv'].logpdf(x_hat[k]) for k in range(K)])
    logProb += np.log(Pi.reshape(-1, 1))

    """
    maxPostLogProb = np.max(logProb, axis=0, keepdims=True)
    expProb = np.exp(logProb - np.tile(maxPostLogProb, (K, 1)))
    postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)
    """

    log_denom = logsumexp(logProb, axis=0, keepdims=True)
    postProb = np.exp(logProb - log_denom)
    
    prob_cumsum = np.cumsum(postProb, axis=0)
    uniform_draws = np.random.rand(x.shape[0])
    z = np.argmax(prob_cumsum >= uniform_draws, axis=0) 
    
    return z



input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. Damm demo data
4. DEMO
Enter the corresponding option number: '''

x, x_dot, _, _ = load_tools.load_data(int(1))
x, x_dir = pre_process(x, x_dot) 


# step 0: define param
init_cluster= 10
M, N = x.shape
T = 100
Psi_0 = np.eye(N) 
nu_0 = 5
mu_0 = np.zeros((N,))
kappa_0 = 1

psi_dir_0 = 1

# step 1: init z
rng = np.random.default_rng()
z = rng.integers(0, init_cluster, size=M)

# step 2: init for loop
for t in range(T):
    z, K = reorder_assignments(z)
    index_lists= update_index_lists(z)
    Pi = np.zeros((K, ))
    gaussian_lists = []

    # step 3: sample GMM param
    for k in range(K):
        Pi[k] = rng.gamma(len(index_lists[k]), 1)
        
        #step 3a: construct posterior NIW
        Psi_n, nu_n, mu_n, kappa_n, psi_dir_n, mu_dir_k = compute_niw_posterior(Psi_0, nu_0, mu_0, kappa_0, psi_dir_0, x[index_lists[k]], x_dir[index_lists[k]])

        #step 3b: sample Sigma, Mu from posterior NIW
        Sigma_pos_k = sample_inverse_wishart(Psi_n, nu_n)
        Sigma_pos_k = 0.5 * (Sigma_pos_k + Sigma_pos_k.T) #ensure symmetry
        mu_pos_k = sample_multi_normal(mu_n, Sigma_pos_k/kappa_n)

        var_dir_k = sample_inv_gamma(psi_dir_n, nu_n)

        #step 3c: store sampled Gaussian
        mu_k = np.zeros((N+1, ))
        mu_k[:N] = mu_pos_k
        Sigma_k = np.eye(N+1)
        Sigma_k[:N, :N] = Sigma_pos_k
        Sigma_k[-1, -1] = var_dir_k

        # gaussian_lists.append(multivariate_normal(mu_k, Sigma_k, allow_singular=True))
        gaussian_lists.append(
            {   
                "mu_dir": mu_dir_k,
                "rv"    : multivariate_normal(mu_k, Sigma_k, allow_singular=True)
            }
        )

    Pi /= np.sum(Pi)

    # step 4: sample labels
    z = sample_label(x, x_dir, Pi, gaussian_lists)


plot_tools.plot_gmm(x, z)

plt.show()


