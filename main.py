import numpy as np
import matplotlib.pyplot as plt
from src.util import load_tools, plot_tools, quat_tools
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2, norm, multivariate_normal, invgamma
from scipy.special import logsumexp
from collections import OrderedDict
from src.damm_class import DAMM


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


