import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import random


# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Times New Roman",
#     "font.size": 30
# })



def plot_gmm(x_train, label, damm=None):
    """ passing damm object to plot the ellipsoids of clustering results"""
    N = x_train.shape[1]

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
    color_mapping = np.take(colors, label)


    fig = plt.figure(figsize=(12, 10))
    if N == 2:
        ax = fig.add_subplot()
        ax.scatter(x_train[:, 0], x_train[:, 1], color=color_mapping[:], alpha=0.4, label="Demonstration")

        if damm is not None:
            est_K = damm.K
            Priors = damm.Prior
            Mu = damm.Mu.T
            Sigma = damm.Sigma

            # Use the same colors for ellipsoids as for clusters
            from gmr import GMM, plot_error_ellipses
            gmm = GMM(est_K, Priors, Mu.T, Sigma)
            # Pass the first est_K colors to match clusters
            plot_error_ellipses(ax, gmm, alpha=0.1, colors=colors[:est_K])
            for num in np.arange(0, len(Mu[0])):
                plt.text(Mu[0][num], Mu[1][num], str(num+1), fontsize=20)
        
        # plt.savefig('gmm_result.png', dpi=300)




    elif N == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], 'o', color=color_mapping[:], s=3, alpha=0.4, label="Demonstration")

        if damm is not None:
            K = damm.K
            Mu = damm.Mu.T

            for k in range(K):
                _, s, rotation = np.linalg.svd(damm.Sigma[k, :, :])  # find the rotation matrix and radii of the axes
                radii = np.sqrt(s) * 1.5                        # set the scale factor yourself
                u = np.linspace(0.0, 2.0 * np.pi, 60)
                v = np.linspace(0.0, np.pi, 60)
                x = radii[0] * np.outer(np.cos(u), np.sin(v))   # calculate cartesian coordinates for the ellipsoid surface
                y = radii[1] * np.outer(np.sin(u), np.sin(v))
                z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
                for i in range(len(x)):
                    for j in range(len(x)):
                        [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + damm.Mu[k, :]
                ax.plot_surface(x, y, z, rstride=3, cstride=3, color=colors[k], linewidth=0.1, alpha=0.3, shade=True)
                # Plot text at the mean position
                ax.text(Mu[0][k], Mu[1][k], Mu[2][k]+0.03, str(k+1), fontsize=20)

        ax.set_xlabel(r'$\xi_1$', fontsize=38, labelpad=20)
        ax.set_ylabel(r'$\xi_2$', fontsize=38, labelpad=20)
        ax.set_zlabel(r'$\xi_3$', fontsize=38, labelpad=20)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis='z', which='major', pad=15)
        ax.view_init(elev=30, azim=-20)
        # plt.savefig('gmm_result.png', dpi=300)



def plot_gamma(gamma_arr, **argv):

    K, M = gamma_arr.shape

    fig, axs = plt.subplots(K, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(K):
        axs[k].scatter(np.arange(M), gamma_arr[k, :], s=5, color=colors[k])
        axs[k].set_ylim([0, 1])
    
    if "title" in argv:
        axs[0].set_title(argv["title"])
    else:
        axs[0].set_title(r"$\gamma(\cdot)$ over Time")

