import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from numpy import ma
from scipy import optimize

from matplotlib import cm, ticker, pyplot
from tqdm import tqdm

from modules.bound_utils import binary_kl_bound
from nnlib.nnlib.matplotlib_utils import set_default_configs

set_default_configs(plt, seaborn)


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


def plot_line(ax, xs, data, label, marker):
    mean, std = np.mean(data, axis=1), np.std(data, axis=1)
    ax.plot(xs, mean, label=label, marker=marker)
    if data.shape[1] > 1:
        ax.fill_between(xs, mean-std, mean+std, alpha=0.2)


def plot_synthetic(m, noise):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "synthetic-MLP-noise" if noise else "synthetic-MLP"
    results_file_path = os.path.join(results_dir, exp_name, 'pair_results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = [20, 40, 60, 80, 100]
    data = np.zeros((5, 4, 1))
    for i in range(5):
        results_n = results[ns[i]][m][200]
        bounds = results_n['pair_bound']

        data[i, 0, 0] = results_n['gen_gap']
        data[i, 1, 0] = bounds[0]
        data[i, 2, 0] = bounds[1] - results_n['train_risk']
        data[i, 3, 0] = bounds[2]

    xs = np.arange(5)
    plot_line(ax, xs, data[:, 0, :], 'Error', 'o')
    plot_line(ax, xs, data[:, 1, :], 'Square', 'd')
    plot_line(ax, xs, data[:, 2, :], 'Binary KL', '^')
    plot_line(ax, xs, data[:, 3, :], 'Fast-rate', 'x')

    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig(f'figures/synthetic{m}-noise.pdf' if noise else f'figures/synthetic{m}.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_mnist(e):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "fcmi-mnist-4vs9-CNN" if e == 0 else f"fcmi-mnist-4vs9-CNN-{e}"
    results_file_path = os.path.join(results_dir, exp_name, 'results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = [75, 250, 1000, 4000]
    data = np.zeros((4, 4, 5))
    for i in range(4):
        results_n = results[ns[i]][100]
        for j in range(5):
            bounds = results_n[j]['pair_bound']

            data[i, 0, j] = results_n[j]['gen_gap']
            data[i, 1, j] = bounds[0]
            data[i, 2, j] = bounds[1] - results_n[j]['train_risk']
            data[i, 3, j] = bounds[2]

    xs = np.arange(4)
    plot_line(ax, xs, data[:, 0, :], 'Error', 'o')
    plot_line(ax, xs, data[:, 1, :], 'Square', 'd')
    plot_line(ax, xs, data[:, 2, :], 'Binary KL', '^')
    plot_line(ax, xs, data[:, 3, :], 'Fast-rate', 'x')

    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/mnist.pdf' if e == 0 else f'figures/mnist{e}.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_cifar():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "cifar10-pretrained-resnet50"
    results_file_path = os.path.join(results_dir, exp_name, 'results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = [1000, 5000, 20000]
    data = np.zeros((3, 4, 1))
    for i in range(3):
        results_n = results[ns[i]][40]
        bounds = results_n[0]['pair_bound']

        data[i, 0, 0] = results_n[0]['gen_gap']
        data[i, 1, 0] = bounds[0]
        data[i, 2, 0] = bounds[1] - results_n[0]['train_risk']
        data[i, 3, 0] = bounds[2]

    xs = np.arange(3)
    plot_line(ax, xs, data[:, 0, :], 'Error', 'o')
    plot_line(ax, xs, data[:, 1, :], 'Square', 'd')
    plot_line(ax, xs, data[:, 2, :], 'Binary KL', '^')
    plot_line(ax, xs, data[:, 3, :], 'Fast-rate', 'x')

    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/cifar.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_mnist_ld():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "fcmi-mnist-4vs9-CNN-LD"
    results_file_path = os.path.join(results_dir, exp_name, 'results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = np.arange(1, 11) * 4
    data = np.zeros((10, 4, 5))
    for i in range(10):
        results_n = results[4000][ns[i]]
        for j in range(5):
            bounds = results_n[j]['pair_bound']

            data[i, 0, j] = results_n[j]['gen_gap']
            data[i, 1, j] = bounds[0]
            data[i, 2, j] = bounds[1] - results_n[j]['train_risk']
            data[i, 3, j] = bounds[2]

    xs = np.arange(10)
    plot_line(ax, xs, data[:, 0, :], 'Error', 'o')
    plot_line(ax, xs, data[:, 1, :], 'Square', 'd')
    plot_line(ax, xs, data[:, 2, :], 'Binary KL', '^')
    plot_line(ax, xs, data[:, 3, :], 'Fast-rate', 'x')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/mnist_ld.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_clip():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    results_dir = "results"
    exp_name = "flickr-pretrained-clip"
    results_file_path = os.path.join(results_dir, exp_name, 'clip_results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = [1000, 4000, 15000]
    data = np.zeros((3, 4, 1))
    for i in range(3):
        results_n = results[ns[i]][40]
        bounds = results_n[0]['clip_bound']

        data[i, 0, 0] = results_n[0]['gen_gap']
        data[i, 1, 0] = bounds[0]
        data[i, 2, 0] = bounds[1] - results_n[0]['train_risk']
        data[i, 3, 0] = bounds[2]

    xs = np.arange(3)
    plot_line(ax, xs, data[:, 0, :], 'Error', 'o')
    plot_line(ax, xs, data[:, 1, :], 'Square', 'd')
    plot_line(ax, xs, data[:, 2, :], 'Binary KL', '^')
    plot_line(ax, xs, data[:, 3, :], 'Fast-rate', 'x')

    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_xticks(xs)
    ax.set_xticklabels(ns)
    ax.legend()

    fig.savefig('figures/clip.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_compare(rnd):
    N = 1000
    x = (np.arange(N) + 0.5) / N * np.log(2)
    y = (np.arange(N) + 0.5) / N

    X, Y = np.meshgrid(x, y)

    def optimize_fast(X, Y):
        def func(c2):
            c1 = -np.log(2 - np.exp(c2)) / c2 - 1
            return (c1 + 1) * Y + X / c2

        def grad(c2):
            dc1 = np.exp(c2) / (2 - np.exp(c2)) / c2 + np.log(2 - np.exp(c2)) / (c2 ** 2)
            return dc1 * Y - X / (c2 ** 2)

        res = optimize.minimize(func, np.array([0.1]), jac=grad, method="L-BFGS-B", bounds=((1e-9, np.log(2) - 1e-9),))
        return res.fun

    bound_trivial = np.ones((N, N))
    bound_square = np.sqrt(2 * X) + Y
    bound_square_rnd = np.sqrt(np.pi * X / 2) + Y

    if not os.path.exists('saved_fast_bkl.pkl'):
        bound_fast = np.zeros((N, N))
        bound_bkl = np.zeros((N, N))
        for i in tqdm(range(N)):
            for j in range(N):
                bound_fast[i, j] = optimize_fast(X[i, j], Y[i, j])
                bound_bkl[i, j] = binary_kl_bound(Y[i, j], X[i, j])

        with open('saved_fast_bkl.pkl', 'wb') as f:
            pickle.dump({
                "bound_fast": bound_fast,
                "bound_bkl": bound_bkl,
            }, f)
    else:
        with open('saved_fast_bkl.pkl', 'rb') as f:
            saved_bound = pickle.load(f)

            bound_fast = saved_bound["bound_fast"]
            bound_bkl = saved_bound["bound_bkl"]

    bounds = np.stack([bound_trivial, bound_fast, bound_bkl, bound_square if rnd == 0 else bound_square_rnd], axis=2)
    z = np.argmin(bounds, axis=2)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colorsList = ['w','r','g','b']
    CustomCmap = mpl.colors.ListedColormap(colorsList)

    im = ax.imshow(z, cmap=CustomCmap, alpha = 0.3)
    im.set_clim(-0.5, 3.5)

    ax.set_xlabel(r'$B$')
    ax.set_ylabel(r'$L_n$')
    ax.set_ylim([0, 1])
    ax.set_xticks(np.linspace(0,N+1,3))
    ax.set_xticklabels([r'$0$',r'$\log(2)/2$',r'$\log(2)$'])
    ax.set_yticks(np.flip(np.arange(0,N+1,N/4)))
    ax.set_yticklabels(np.linspace(1,0,5))

    if rnd == 1:
        cbar = fig.colorbar(im, ticks = [0,1,2,3])
        cbar.ax.set_yticklabels(['Trivial', 'Fast-rate', 'Binary KL', 'Square-root'])

    fig.savefig('figures/compare_%d.pdf' % rnd, format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def plot_compare_c():
    N = 1000
    x = (np.arange(N) + 0.5) / N * np.log(2)
    y = (np.arange(N) + 0.5) / N * 10

    X, Y = np.meshgrid(x, y)
    # Y = np.flip(Y, axis=0)

    bound_fast = (Y >= (-np.log(2 - np.exp(X)) / X - 1)).astype(np.int32)
    bound_linear = (-X * Y + (np.exp(X) - 1 - X) * (Y ** 2 + Y * 2 + 2) <= 0).astype(np.int32)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    im = ax.imshow(bound_fast - bound_linear, cmap=mpl.colors.ListedColormap([[0,0,0,0], [1,0,0,0.3]]), alpha = 1)
    im.set_clim(-0.5, 1.5)
    im = ax.imshow(bound_linear, cmap=mpl.colors.ListedColormap([[0,0,0,0], [0,0.5,0,0.3]]), alpha = 1)
    im.set_clim(-0.5, 1.5)

    ax.set_xlabel(r'$C_2$')
    ax.set_ylabel(r'$C_1$')
    ax.set_ylim([0, 10])
    ax.set_xticks(np.linspace(0,N+1,3))
    ax.set_xticklabels([r'$0$',r'$\log(2)/2$',r'$\log(2)$'])
    ax.set_yticks(np.flip(np.arange(0,N+1,N/4)))
    ax.set_yticklabels(np.linspace(10,0,5))

    cbar = fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(-0.5, 1.5),
                                          cmap=mpl.colors.ListedColormap([[1,0,0,0.3], [0,0.5,0,0.3]])), ticks = [0,1])
    cbar.ax.set_yticklabels(['Fast-rate', 'Linear &\nFast-rate'])

    fig.savefig('figures/compare_c.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.show()


def table_variance():
    results_dir = "results"
    exp_name = "synthetic-MLP"
    results_file_path = os.path.join(results_dir, exp_name, 'pair_results.pkl')
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    ns = [20, 40, 60, 80, 100]
    data = np.zeros((4, 5, 3))
    for m in range(4):
        for i in range(5):
            results_n = results[ns[i]][m + 1][200]
            bounds = results_n['pair_bound']

            data[m, i, 0] = results_n['gen_gap']
            data[m, i, 1] = bounds[2]
            data[m, i, 2] = bounds[3]

    for m in range(4):
        print('%d' % (m + 1), end='')
        for j in range(5):
            print(' & %.5f / %.5f' % (data[m, j, 1], data[m, j, 2]), end='')
        print(r' \\')


if __name__ == '__main__':
    # for m in range(4):
        # plot_synthetic(m + 1, False)
        # plot_synthetic(m + 1, True)

    # for e in [0, 0.05, 0.1, 0.15]:
    #     plot_mnist(e)
    # plot_cifar()
    # plot_mnist_ld()
    # plot_clip()

    # plot_compare(0)
    # plot_compare(1)

    # plot_compare_c()

    table_variance()
