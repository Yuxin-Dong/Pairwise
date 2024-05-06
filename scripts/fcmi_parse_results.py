import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import pickle

import numpy as np
import torch

from nnlib.nnlib import utils
from nnlib.nnlib.data_utils.wrappers import SubsetDataWrapper
from nnlib.nnlib.matplotlib_utils import import_matplotlib
from scripts.train_synthetic import BasicMLP

matplotlib, plt = import_matplotlib()

from modules.bound_utils import estimate_fcmi_bound_classification, estimate_sgld_bound, estimate_clip_bound, estimate_pairwise_bound, estimate_multivariate_bound
from scripts.fcmi_train_classifier import mnist_ld_schedule, \
    cifar_resnet50_ld_schedule  # for pickle to be able to load LD methods
from scripts.fcmi_train_clip import image_title_dataset
import methods


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


def compute_acc(preds, mask, dataset):
    labels = [y for x, y in dataset]
    labels = torch.tensor(labels).long()
    indices = 2*np.arange(len(mask)) + mask
    acc = (preds[indices].argmax(dim=1) == labels[indices]).float().mean()
    return utils.to_numpy(acc)


def get_fcmi_results_for_fixed_z(n, epoch, seed, args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_risks = []
    val_risks = []
    train_losses = []
    val_losses = []
    masks = []

    for S_seed in range(args.n_S_seeds):
        dir_name = f'n={n},seed={seed},S_seed={S_seed}'
        dir_path = os.path.join(args.results_dir, args.exp_name, dir_name)
        if not os.path.exists(dir_path):
            print(f"Did not find results for {dir_name}")
            continue

        with open(os.path.join(dir_path, 'saved_data.pkl'), 'rb') as f:
            saved_data = pickle.load(f)

        if not os.path.exists(os.path.join(dir_path, f'saved_logits{epoch - 1}.pkl')):
            model = utils.load(path=os.path.join(dir_path, 'checkpoints', f'epoch{epoch - 1}.mdl'),
                               methods=methods, device=args.device)

            train_data = SubsetDataWrapper(saved_data['all_examples'], include_indices=saved_data['train_indices'])
            val_data = SubsetDataWrapper(saved_data['all_examples'], include_indices=saved_data['val_indices'])

            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

            def get_logits(data_loader):
                all_features = []
                all_labels = []
                with torch.no_grad():
                    for batch in tqdm(data_loader):
                        images, labels = batch

                        images = images.to(device)
                        labels = labels.to(device)

                        features = model.forward([images])
                        all_features.append(features)
                        all_labels.append(labels)

                    all_features = torch.concat(all_features, dim=0)
                    all_labels = torch.concat(all_labels, dim=0)

                all_labels = all_labels.contiguous().view(-1, 1)
                cmask = torch.eq(all_labels, all_labels.T).to(torch.uint8)
                logits = torch.matmul(all_features, all_features.T)

                return logits, cmask

            train_logits, train_cmask = get_logits(train_loader)
            val_logits, val_cmask = get_logits(val_loader)

            with open(os.path.join(dir_path, f'saved_logits{epoch - 1}.pkl'), 'wb') as f:
                pickle.dump({
                    "train_logits": train_logits,
                    "train_cmask": train_cmask,
                    "val_logits": val_logits,
                    "val_cmask": val_cmask,
                }, f)

        else:
            with open(os.path.join(dir_path, f'saved_logits{epoch - 1}.pkl'), 'rb') as f:
                saved_logits = pickle.load(f)

                train_logits = saved_logits["train_logits"]
                train_cmask = saved_logits["train_cmask"]
                val_logits = saved_logits["val_logits"]
                val_cmask = saved_logits["val_cmask"]

        def get_losses(logits, cmask):
            pos_cmask = cmask.clone().fill_diagonal_(0)
            pos_logits = logits[pos_cmask].sort().values
            pos_n = pos_cmask.sum()

            mn, mx = 0, pos_logits.shape[0]
            while mn < mx:
                md = (mn + mx) // 2
                loss = (logits >= pos_logits[md]) != cmask
                loss.fill_diagonal_(0)

                pos_risk = loss[pos_cmask].sum() / pos_n
                neg_risk = loss[1 - cmask].sum() / (n * (n - 1) - pos_n)

                if pos_risk > neg_risk:
                    mx = md
                else:
                    mn = md + 1

            return loss, utils.to_numpy(loss.sum() / n / (n - 1))

        cur_train_loss, cur_train_risk = get_losses(train_logits, train_cmask)
        cur_val_loss, cur_val_risk = get_losses(val_logits, val_cmask)
        print(cur_train_risk, cur_val_risk)

        train_risks.append(cur_train_risk)
        val_risks.append(cur_val_risk)
        train_losses.append(cur_train_loss.cpu())
        val_losses.append(cur_val_loss.cpu())
        masks.append(torch.tensor(saved_data['mask']))

    pairwise_bound = estimate_pairwise_bound(masks, train_losses, val_losses, n)

    cur_result = {
        'train_risk': np.mean(train_risks),
        'val_risk': np.mean(val_risks),
        'gen_gap': np.mean(val_risks) - np.mean(train_risks),
        'pair_bound': pairwise_bound,
    }

    print(cur_result)
    return cur_result


def get_fcmi_results_for_fixed_model(n, epoch, args):
    results = []
    for seed in range(args.n_seeds):
        cur = get_fcmi_results_for_fixed_z(n=n, epoch=epoch, seed=seed, args=args)
        results.append(cur)
    return results


def get_clip_results(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    results = NestedDict()
    for n in tqdm(args.ns):
        for epoch in tqdm(args.epochs, leave=False):
            cur_results = []

            for seed in range(args.n_seeds):
                train_risks = []
                val_risks = []
                train_losses = []
                val_losses = []
                masks = []

                for S_seed in range(args.n_S_seeds):
                    dir_name = f'n={n},seed={seed},S_seed={S_seed}'
                    dir_path = os.path.join(args.results_dir, args.exp_name, dir_name)
                    if not os.path.exists(dir_path):
                        print(f"Did not find results for {dir_name}")
                        continue

                    with open(os.path.join(dir_path, 'saved_data.pkl'), 'rb') as f:
                        saved_data = pickle.load(f)

                    if not os.path.exists(os.path.join(dir_path, 'saved_logits.pkl')):
                        checkpoint = torch.load(os.path.join(dir_path, 'checkpoints', f'epoch{epoch - 1}.mdl'), map_location=args.device)
                        model.load_state_dict(checkpoint['model'])
                        model.eval()

                        train_data = SubsetDataWrapper(saved_data['all_examples'], include_indices=saved_data['train_indices'])
                        val_data = SubsetDataWrapper(saved_data['all_examples'], include_indices=saved_data['val_indices'])

                        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
                        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

                        def get_logits(data_loader):
                            all_image_features, all_text_features = [], []
                            with torch.no_grad():
                                for batch in tqdm(data_loader):
                                    images, texts = batch

                                    images = images.to(device)
                                    texts = texts.to(device)

                                    image_features = model.encode_image(images)
                                    text_features = model.encode_text(texts)
                                    all_image_features.append(image_features / image_features.norm(dim=1, keepdim=True))
                                    all_text_features.append(text_features / text_features.norm(dim=1, keepdim=True))

                                all_image_features = torch.concat(all_image_features, dim=0)
                                all_text_features = torch.concat(all_text_features, dim=0)

                            logit_scale = model.logit_scale.exp()
                            return logit_scale * all_image_features @ all_text_features.t()

                        train_logits = get_logits(train_loader)
                        val_logits = get_logits(val_loader)

                        with open(os.path.join(dir_path, 'saved_logits.pkl'), 'wb') as f:
                            pickle.dump({"train_logits": train_logits, "val_logits": val_logits}, f)

                    else:
                        with open(os.path.join(dir_path, 'saved_logits.pkl'), 'rb') as f:
                            saved_logits = pickle.load(f)

                            train_logits = saved_logits["train_logits"]
                            val_logits = saved_logits["val_logits"]

                    def get_losses(logits):
                        diag_logits = logits.diag().sort().values
                        mn, mx = 0, n
                        while mn < mx:
                            md = (mn + mx) // 2
                            loss = (logits >= diag_logits[md]) != torch.eye(n, device=device)

                            point_loss = loss.trace() / n
                            pair_loss = (loss.sum() - loss.trace()) / n / (n - 1)

                            if point_loss > pair_loss:
                                mx = md
                            else:
                                mn = md + 1

                        return loss, utils.to_numpy(loss.sum() / (n ** 2))

                    cur_train_loss, cur_train_risk = get_losses(train_logits)
                    cur_val_loss, cur_val_risk = get_losses(val_logits)
                    print(cur_train_risk, cur_val_risk)

                    train_risks.append(cur_train_risk)
                    val_risks.append(cur_val_risk)
                    train_losses.append(cur_train_loss.cpu())
                    val_losses.append(cur_val_loss.cpu())
                    masks.append(torch.tensor(saved_data['mask']))

                clip_bound = estimate_clip_bound(masks, train_losses, val_losses, n)
                cur_result = {
                    'train_risk': np.mean(train_risks),
                    'val_risk': np.mean(val_risks),
                    'gen_gap': np.mean(val_risks) - np.mean(train_risks),
                    'clip_bound': clip_bound,
                }
                cur_results.append(cur_result)
                print(cur_result)

            results[n][epoch] = cur_results

    return results


def get_synthetic_results(args, n, m, epoch):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = BasicMLP().to(device)
    model.eval()

    train_risks = []
    val_risks = []
    train_losses = []
    val_losses = []
    masks = []

    for seed in range(args.n_seeds):
        for S_seed in range(args.n_S_seeds):
            dir_name = f'n={n},m={m},seed={seed},S_seed={S_seed}'
            dir_path = os.path.join(args.results_dir, args.exp_name, dir_name)
            if not os.path.exists(dir_path):
                print(f"Did not find results for {dir_name}")
                continue

            with open(os.path.join(dir_path, 'saved_data.pkl'), 'rb') as f:
                saved_data = pickle.load(f)

            if True: #not os.path.exists(os.path.join(dir_path, f'saved_logits{epoch - 1}.pkl')):
                checkpoint = torch.load(os.path.join(dir_path, 'checkpoints', f'epoch{epoch - 1}.mdl'), map_location=args.device)
                model.load_state_dict(checkpoint['model'])

                all_examples = saved_data['all_examples']
                all_examples = (torch.tensor(all_examples[0], dtype=torch.float, device=device), torch.tensor(all_examples[1], device=device))
                train_indices = saved_data['train_indices']
                val_indices = saved_data['val_indices']

                train_data = (all_examples[0][train_indices], all_examples[1][train_indices])
                val_data = (all_examples[0][val_indices], all_examples[1][val_indices])

                train_feats = model(train_data[0])
                val_feats = model(val_data[0])

                def get_logits(feats, labels):
                    if m == 1:
                        return model.classifier(feats), labels
                    else:
                        logits = torch.matmul(feats, feats.T)
                        cmask = torch.eq(labels[:, None], labels[None, :])

                        if m == 3:
                            logits = logits[:, :, None] - logits[:, None, :]
                            cmask = torch.logical_and(cmask[:, :, None], cmask[:, None, :].logical_not())
                        elif m == 4:
                            logits = logits[:, :, None, None] - logits[:, None, :, None] - logits[:, None, None, :]
                            cmask = torch.logical_and(torch.logical_and(cmask[:, :, None, None], cmask[:, None, :, None].logical_not()),
                                                      cmask[:, None, None, :].logical_not())

                        return logits, cmask

                train_logits, train_cmask = get_logits(train_feats, train_data[1])
                val_logits, val_cmask = get_logits(val_feats, val_data[1])

                # with open(os.path.join(dir_path, f'saved_logits{epoch - 1}.pkl'), 'wb') as f:
                #     pickle.dump({
                #         "train_logits": train_logits,
                #         "train_cmask": train_cmask,
                #         "val_logits": val_logits,
                #         "val_cmask": val_cmask,
                #     }, f)

            else:
                with open(os.path.join(dir_path, f'saved_logits{epoch - 1}.pkl'), 'rb') as f:
                    saved_logits = pickle.load(f)

                    train_logits = saved_logits["train_logits"]
                    train_cmask = saved_logits["train_cmask"]
                    val_logits = saved_logits["val_logits"]
                    val_cmask = saved_logits["val_cmask"]

            def clear_diagonal(mat):
                idx = torch.arange(n, device=device)
                if m == 2:
                    mat[idx, idx] = 0
                elif m == 3:
                    mat[idx, idx, :] = 0
                elif m == 4:
                    mat[idx, idx, :, :] = 0
                    mat[:, :, idx, idx] = 0
                return mat

            def get_losses(logits, cmask):
                if m == 1:
                    loss = logits.argmax(dim=1) != cmask
                    return loss, utils.to_numpy(loss.sum() / n)
                else:
                    pos_cmask = clear_diagonal(cmask.clone())
                    pos_logits = logits[pos_cmask].sort().values
                    pos_n = pos_cmask.sum()
                    neg_n = torch.numel(cmask) - cmask.sum()

                    mn, mx = 0, pos_logits.shape[0]
                    while mn < mx:
                        md = (mn + mx) // 2
                        loss = (logits >= pos_logits[md]) != cmask
                        clear_diagonal(loss)

                        pos_risk = loss[pos_cmask].sum() / pos_n
                        neg_risk = loss[torch.logical_not(cmask)].sum() / neg_n

                        if pos_risk > neg_risk:
                            mx = md
                        else:
                            mn = md + 1

                    return loss, utils.to_numpy(loss.sum() / (pos_n + neg_n))

            cur_train_loss, cur_train_risk = get_losses(train_logits, train_cmask)
            cur_val_loss, cur_val_risk = get_losses(val_logits, val_cmask)
            print(cur_train_risk, cur_val_risk)

            train_risks.append(cur_train_risk)
            val_risks.append(cur_val_risk)
            train_losses.append(cur_train_loss.cpu())
            val_losses.append(cur_val_loss.cpu())
            masks.append(torch.tensor(saved_data['mask']))

    pairwise_bound = estimate_multivariate_bound(masks, train_losses, val_losses, n, m)

    cur_result = {
        'train_risk': np.mean(train_risks),
        'val_risk': np.mean(val_risks),
        'gen_gap': np.mean(val_risks) - np.mean(train_risks),
        'pair_bound': pairwise_bound,
    }

    print(cur_result)
    return cur_result


def get_sgld_results_for_fixed_model(n, epoch, args):
    results = []

    for seed in range(args.n_seeds):
        for S_seed in range(args.n_S_seeds):
            if S_seed >= 4:
                continue  # these guys didn't track gradient variance to save time

            dir_name = f'n={n},seed={seed},S_seed={S_seed}'
            dir_path = os.path.join(args.results_dir, args.exp_name, dir_name)
            if not os.path.exists(dir_path):
                print(f"Did not find results for {dir_name}")
                continue

            with open(os.path.join(dir_path, 'saved_data.pkl'), 'rb') as f:
                saved_data = pickle.load(f)

            model = utils.load(path=os.path.join(dir_path, 'checkpoints', f'epoch{epoch - 1}.mdl'),
                               methods=methods, device=args.device)

            if 'all_examples_wo_data_aug' in saved_data:
                all_examples = saved_data['all_examples_wo_data_aug']
            else:
                all_examples = saved_data['all_examples']

            cur_preds = utils.apply_on_dataset(model=model, dataset=all_examples,
                                               batch_size=args.batch_size)['pred']
            cur_mask = saved_data['mask']
            cur_train_acc = compute_acc(preds=cur_preds, mask=cur_mask, dataset=all_examples)
            cur_val_acc = compute_acc(preds=cur_preds, mask=1-cur_mask, dataset=all_examples)

            cur_result = {}
            cur_result['train_acc'] = cur_train_acc
            cur_result['val_acc'] = cur_val_acc
            cur_result['gap'] = cur_val_acc - cur_train_acc

            sgld_bound = estimate_sgld_bound(n=n, batch_size=args.batch_size,
                                             model=model)
            cur_result['sgld_bound'] = sgld_bound
            results.append(cur_result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.set_defaults(parse=True)
    args = parser.parse_args()
    print(args)

    if args.exp_name == 'fcmi-mnist-4vs9-CNN-LD':
        args.n_seeds = 5
        args.n_S_seeds = 30
        args.ns = [4000]
        args.epochs = np.arange(1, 11) * 4
        args.num_classes = 2
        args.batch_size = 100
    elif args.exp_name.startswith('fcmi-mnist-4vs9-CNN'):
        args.n_seeds = 5
        args.n_S_seeds = 30
        args.ns = [75, 250, 1000, 4000]
        args.epochs = [100]
        args.num_classes = 2
    elif args.exp_name == 'cifar10-pretrained-resnet50':
        args.n_seeds = 1
        args.n_S_seeds = 40
        args.ns = [1000, 5000, 20000]
        args.epochs = [40]
        args.num_classes = 10
    elif args.exp_name == 'flickr-pretrained-clip':
        args.n_seeds = 1
        args.n_S_seeds = 40
        args.ns = [1000, 4000, 15000]
        args.epochs = [40]
    elif args.exp_name in ['synthetic-MLP', 'synthetic-MLP-noise']:
        args.n_seeds = 1
        args.n_S_seeds = 200
        args.ns = [20, 40, 60, 80, 100]
        args.ms = [1, 2, 3, 4]
        args.epochs = [200]
    else:
        raise ValueError(f"Unexpected exp_name: {args.exp_name}")

    if args.exp_name.startswith('fcmi-mnist-4vs9-CNN') or args.exp_name == 'cifar10-pretrained-resnet50':
        results = NestedDict()  # indexing with n, epoch
        for n in tqdm(args.ns):
            for epoch in tqdm(args.epochs, leave=False):
                results[n][epoch] = get_fcmi_results_for_fixed_model(n=n, epoch=epoch, args=args)
        results_file_path = os.path.join(args.results_dir, args.exp_name, 'results.pkl')
        with open(results_file_path, 'wb') as f:
            pickle.dump(results, f)

    if args.exp_name == 'flickr-pretrained-clip':
        clip_results = get_clip_results(args)
        results_file_path = os.path.join(args.results_dir, args.exp_name, 'clip_results.pkl')
        with open(results_file_path, 'wb') as f:
            pickle.dump(clip_results, f)

    if args.exp_name in ['synthetic-MLP', 'synthetic-MLP-noise']:
        results = NestedDict()
        for n in tqdm(args.ns):
            for m in tqdm(args.ms):
                for epoch in tqdm(args.epochs, leave=False):
                    results[n][m][epoch] = get_synthetic_results(args, n, m, epoch)
        results_file_path = os.path.join(args.results_dir, args.exp_name, 'pair_results.pkl')
        with open(results_file_path, 'wb') as f:
            pickle.dump(results, f)

    # parse the quantities needed for the Negrea et al. SGLD bound
    # if args.exp_name in ['fcmi-mnist-4vs9-CNN-LD', 'cifar10-pretrained-resnet50-LD',
    #                      'fcmi-mnist-4vs9-CNN-LD-shuffle_train_only_after_first_epoch']:
    #     sgld_results = NestedDict()  # indexing with n, epoch
    #     for n in tqdm(args.ns):
    #         for epoch in tqdm(args.epochs, leave=False):
    #             sgld_results[n][epoch] = get_sgld_results_for_fixed_model(n=n, epoch=epoch, args=args)
    #     results_file_path = os.path.join(args.results_dir, args.exp_name, 'sgld_results.pkl')
    #     with open(results_file_path, 'wb') as f:
    #         pickle.dump(sgld_results, f)


if __name__ == '__main__':
    main()
