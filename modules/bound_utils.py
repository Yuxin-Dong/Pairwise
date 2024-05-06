import gc

import numpy as np
import torch
from scipy import optimize

from nnlib.nnlib import utils
from methods import LangevinDynamics


def disc_mi(xs, nx, ys, ny=2):
    prob = torch.zeros((*xs.shape[1:], nx, ny), dtype=torch.float32)
    for k in range(xs.shape[0]):
        for x in range(nx):
            for y in range(ny):
                prob[..., x, y].add_((xs[k] == x).logical_and_(ys[k] == y) / xs.shape[0])
                gc.collect()
    prob.clamp_min_(torch.tensor(1e-9))

    px = torch.sum(prob, dim=-1)
    py = torch.sum(prob, dim=-2)
    mi = prob.clone().div_(px[..., :, None]).div_(py[..., None, :]).log_().mul_(prob).sum(dim=(-1, -2))
    return mi.clamp_min_(torch.tensor(0))


def disc2(xs, nx, ys, ny=2):
    prob = torch.zeros((xs.shape[1], xs.shape[2], nx, ny), dtype=torch.float32)
    for k in range(xs.shape[0]):
        for x in range(nx):
            for y in range(ny):
                prob[:, :, x, y].add_((xs[k] == x).logical_and_(ys[k] == y) / xs.shape[0])
                gc.collect()

    # prob1 = torch.zeros((xs.shape[1], xs.shape[2], nx, ny), dtype=torch.float32)
    # prob1.index_put_((torch.arange(xs.shape[1])[None, :, None], torch.arange(xs.shape[2])[None, None, :], xs.long(), ys.long()),
    #                 torch.tensor(1.0 / xs.shape[0]), accumulate=True)
    return prob.clamp_min_(torch.tensor(1e-9))


def disc2_mi(xs, nx, ys, ny=2):
    prob = disc2(xs, nx, ys, ny)

    px = torch.sum(prob, dim=3)
    py = torch.sum(prob, dim=2)
    mi = prob.clone().div_(px[:, :, :, None]).div_(py[:, :, None, :]).log_().mul_(prob).sum(dim=(2, 3))
    return mi.clamp_min_(torch.tensor(0))


def discrete_mi_est(xs, ys, nx=2, ny=2):
    prob = np.zeros((nx, ny))
    for a, b in zip(xs, ys):
        prob[a,b] += 1.0/len(xs)
    pa = np.sum(prob, axis=1)
    pb = np.sum(prob, axis=0)
    mi = 0
    for a in range(nx):
        for b in range(ny):
            if prob[a,b] < 1e-9:
                continue
            mi += prob[a,b] * np.log(prob[a,b]/(pa[a]*pb[b]))
    return max(0.0, mi)


def estimate_fcmi_bound_classification(masks, preds, num_examples, num_classes,
                                       verbose=False, return_list_of_mis=False):
    bound = 0.0
    list_of_mis = []
    for idx in range(num_examples):
        ms = [p[idx] for p in masks]
        ps = [p[2*idx:2*idx+2] for p in preds]
        for i in range(len(ps)):
            ps[i] = torch.argmax(ps[i], dim=1)
            ps[i] = num_classes * ps[i][0] + ps[i][1]
            ps[i] = ps[i].item()
        cur_mi = disc_mi(ms, 2, ps, ny=num_classes**2)
        list_of_mis.append(cur_mi)
        bound += np.sqrt(2 * cur_mi)
        if verbose and idx < 10:
            print("ms:", ms)
            print("ps:", ps)
            print("mi:", cur_mi)
    bound *= 1/num_examples

    if return_list_of_mis:
        return bound, list_of_mis

    return bound


def estimate_variance(single_mi, train_loss, train_loss_square, gamma=0.9):
    def func(c2):
        c1 = (-np.log(2 - np.exp(2 * c2)) / (2 * c2) - 1) / (gamma ** 2)
        return c1 * (train_loss - (1 - gamma ** 2) * train_loss_square) + single_mi / c2

    def grad(c2):
        dc1 = np.exp(2 * c2) / (2 - np.exp(2 * c2)) / c2 + np.log(2 - np.exp(2 * c2)) / 2 / (c2 ** 2)
        return dc1 / (gamma ** 2) * (train_loss - train_loss_square) + dc1 * train_loss_square - single_mi / (c2 ** 2)

    res = optimize.minimize(func, np.array([0.1]), jac=grad, method="L-BFGS-B", bounds=((1e-9, np.log(2) / 2 - 1e-9),))

    c2 = res.x[0]
    c1 = (-np.log(2 - np.exp(2 * c2)) / (2 * c2) - 1) / (gamma ** 2)
    return res.fun, c1, c2


def binary_kl_bound(q, c):
    q = max(q, 1e-9)

    def func(x):
        return q * np.log(2 * q / (q + x)) + (1 - q) * np.log((1 - q) / (1 - (q + x) / 2)) - c

    try:
        res = optimize.root_scalar(func, bracket=[q, 1], method='brentq')
        return res.root
    except:
        return 2


def estimate_pairwise_bound(masks, list_train_losses, list_val_losses, n):
    masks = torch.stack(masks, dim=0).to(torch.int8)
    train_losses = torch.stack(list_train_losses, dim=0).to(torch.int8)
    val_losses = torch.stack(list_val_losses, dim=0).to(torch.int8)

    del list_train_losses[:]
    del list_val_losses[:]
    gc.collect()

    loss_diff = (val_losses - train_losses) * (masks * 2 - 1)[:, :, None] + 1
    gc.collect()
    ld_mi = disc2_mi(loss_diff, 3, masks[:, :, None]).fill_diagonal_(0)
    gc.collect()
    bound_square = float(utils.to_numpy(torch.sqrt(2 * ld_mi).sum() / n / (n - 1)))

    del loss_diff
    del ld_mi
    gc.collect()

    loss_single = train_losses * masks[:, :, None] + val_losses * (1 - masks[:, :, None])
    gc.collect()
    single_mi = disc2_mi(loss_single, 2, masks[:, :, None]).fill_diagonal_(0)
    gc.collect()
    single_mi = utils.to_numpy(single_mi.sum() / n / (n - 1))
    gc.collect()
    train_loss = utils.to_numpy(train_losses.float().sum() / n / (n - 1) / masks.shape[0])
    gc.collect()
    train_loss_square = utils.to_numpy(torch.mean((torch.sum(train_losses.float(), dim=(1, 2)) / n / (n - 1)) ** 2))
    gc.collect()
    bound_bkl = binary_kl_bound(train_loss, 2 * single_mi)
    bound_variance, c1, c2 = estimate_variance(single_mi, train_loss, train_loss_square)
    bound_weighted = c1 * train_loss + single_mi / c2

    del loss_single
    gc.collect()

    return bound_square, bound_bkl, bound_weighted, bound_variance


def estimate_clip_bound(masks, train_losses, val_losses, n):
    masks = torch.stack(masks, dim=0).to(torch.int8)
    train_losses = torch.stack(train_losses, dim=0).to(torch.int8)
    val_losses = torch.stack(val_losses, dim=0).to(torch.int8)

    loss_diff = (val_losses - train_losses) * (masks * 2 - 1)[:, :, None] + 1
    ld_mi = disc2_mi(loss_diff, 3, masks[:, :, None])
    bound_square = float(utils.to_numpy(torch.sqrt(2 * ld_mi).mean()))

    loss_single = train_losses * masks[:, :, None] + val_losses * (1 - masks[:, :, None])
    single_mi = utils.to_numpy(disc2_mi(loss_single, 2, masks[:, :, None]).mean())
    train_loss = utils.to_numpy(torch.mean(train_losses.float()))
    train_loss_square = utils.to_numpy(torch.mean(torch.mean(train_losses.float(), dim=(1, 2)) ** 2))
    bound_bkl = binary_kl_bound(train_loss, 2 * single_mi)
    bound_variance, c1, c2 = estimate_variance(single_mi, train_loss, train_loss_square)
    bound_weighted = c1 * train_loss + single_mi / c2

    return bound_square, bound_bkl, bound_weighted, bound_variance


def estimate_multivariate_bound(masks, train_losses, val_losses, n, m):
    masks = torch.stack(masks, dim=0).to(torch.int8)
    train_losses = torch.stack(train_losses, dim=0).to(torch.int8)
    val_losses = torch.stack(val_losses, dim=0).to(torch.int8)

    gc.collect()

    num_losses = n
    slice_masks = masks
    for i in range(1, m):
        num_losses *= n - i
        slice_masks = slice_masks.unsqueeze(-1)

    loss_diff = (val_losses - train_losses) * (slice_masks * 2 - 1) + 1
    ld_mi = disc_mi(loss_diff, 3, slice_masks)
    bound_square = float(utils.to_numpy(torch.sqrt(2 * ld_mi).sum() / num_losses))

    loss_single = train_losses * slice_masks + val_losses * (1 - slice_masks)
    single_mi1 = utils.to_numpy(disc_mi(loss_single, 2, slice_masks).sum() / num_losses)
    loss_single = train_losses * (1 - slice_masks) + val_losses * slice_masks
    single_mi2 = utils.to_numpy(disc_mi(loss_single, 2, slice_masks).sum() / num_losses)
    single_mi = (single_mi1 + single_mi2) / 2

    train_loss = utils.to_numpy(train_losses.float().sum() / num_losses / masks.shape[0])
    train_loss_square = utils.to_numpy(torch.mean((torch.sum(train_losses.flatten(1).float(), dim=-1) / num_losses) ** 2))
    bound_bkl = binary_kl_bound(train_loss, 2 * single_mi)
    bound_variance, c1, c2 = estimate_variance(single_mi, train_loss, train_loss_square)
    bound_weighted = c1 * train_loss + single_mi / c2
    return bound_square, bound_bkl, bound_weighted, bound_variance


def estimate_sgld_bound(n, batch_size, model):
    """ Computes the bound of Negrea et al. "Information-Theoretic Generalization Bounds for
    SGLD via Data-Dependent Estimates". Eq (6) of https://arxiv.org/pdf/1911.02151.pdf.
    """
    assert isinstance(model, LangevinDynamics)
    assert model.track_grad_variance
    T = len(model._grad_variance_hist)
    assert len(model._lr_hist) == T + 1
    assert len(model._beta_hist) == T + 1
    ret = 0.0
    for t in range(1, T):  # skipping the first iteration as grad_variance was not tracked for it
        ret += model._lr_hist[t] * model._beta_hist[t] / 4.0 * model._grad_variance_hist[t-1]
    ret = np.sqrt(utils.to_numpy(ret))
    ret *= np.sqrt(n / 4.0 / batch_size / (n-1) / (n-1))
    return ret
