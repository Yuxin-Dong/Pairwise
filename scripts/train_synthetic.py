import argparse
import json
import os
import pickle
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image
from sklearn import datasets as sk_datasets

import clip
import clip.model
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm

from nnlib.nnlib import utils
from nnlib.nnlib.data_utils.wrappers import SubsetDataWrapper


class BasicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = None

        self.encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16)
        )

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(16, 5)
        )

    @staticmethod
    def cos_dist(feats1, feats2):
        return -(feats1 * feats2).sum(dim=1)

    @staticmethod
    def euc_dist(feats1, feats2):
        return (feats1 - feats2).norm(dim=1)

    def forward(self, xs):
        feats = self.encoder(xs)
        feats = feats / feats.norm(dim=1, keepdim=True)
        return feats

    def pointwise(self, feats, labels):
        logits = self.classifier(feats)
        loss = F.cross_entropy(logits, labels)
        return loss

    def pairwise(self, feats1, feats2):
        batch_size = feats1.shape[0] // 2
        dists = self.euc_dist(feats1, feats2)
        loss = dists[:batch_size] + (0.5 - dists[batch_size:]).clamp_min_(torch.tensor(0, device=feats1.device))
        return loss.mean()

    def triplet(self, feats, pos_feats, neg_feats):
        pos_dists = self.euc_dist(feats, pos_feats)
        neg_dists = self.euc_dist(feats, neg_feats)
        loss = torch.log(1 + (pos_dists - neg_dists).exp())
        return loss.mean()

    def npair4(self, feats, pos_feats, neg_feats1, neg_feats2):
        pos_dists = self.euc_dist(feats, pos_feats)
        neg_dists1 = self.euc_dist(feats, neg_feats1)
        neg_dists2 = self.euc_dist(feats, neg_feats2)
        loss = torch.log(1 + (neg_dists1 - pos_dists).exp() + (neg_dists2 - pos_dists).exp())
        return loss.mean()


def generate_data(args):
    np.random.seed(42)
    centers = np.random.uniform(-1, 1, (5, 5))
    xs, ys = sk_datasets.make_blobs(n_samples=200, centers=centers, n_features=args.dim, shuffle=False,
                                    cluster_std=args.cluster_std, random_state=args.seed)

    indices = (np.concatenate([np.array([i * 40] * (args.n * 2 // 5)) for i in range(5)], axis=0)
               + np.tile(np.arange(args.n * 2 // 5), 5))
    xs = xs[indices]
    ys = ys[indices]

    np.random.seed(args.S_seed)
    mask = np.random.randint(2, size=(args.n,))
    train_indices = 2 * np.arange(args.n) + mask
    val_indices = 2 * np.arange(args.n) + (1 - mask)
    train_xs, train_ys = xs[train_indices], ys[train_indices]
    val_xs, val_ys = xs[val_indices], ys[val_indices]

    return mask, train_indices, val_indices, (xs, ys), (train_xs, train_ys), (val_xs, val_ys)


def sample_batch(args, data):
    batch_size = min(args.batch_size, args.n) // args.num_classes

    indices = np.random.choice(args.n, batch_size)
    if args.m == 1:
        return indices

    half_batch_size = batch_size if args.m > 2 else batch_size // 2
    indices_classes = data[1][indices]
    indices_per_class = args.n // args.num_classes

    pos_indices = np.random.choice(indices_per_class, half_batch_size)
    pos_indices += indices_classes[:half_batch_size] * indices_per_class

    neg_indices = np.random.choice(indices_per_class * (args.num_classes - 1), half_batch_size)
    neg_indices[neg_indices // indices_per_class >= indices_classes[batch_size-half_batch_size:]] += indices_per_class

    if args.m == 2:
        return indices, np.concatenate([pos_indices, neg_indices])

    if args.m == 3:
        return indices, pos_indices, neg_indices

    neg_indices2 = np.random.choice(indices_per_class * (args.num_classes - 1), batch_size)
    neg_indices2[neg_indices2 // indices_per_class >= indices_classes] += indices_per_class

    return indices, pos_indices, neg_indices, neg_indices2


def compute_loss(args, model, data, batch_indices):
    if args.m == 1:
        batch_feats = model(data[0][batch_indices])
        return model.pointwise(batch_feats, data[1][batch_indices])
    else:
        batch_feats = [model(data[0][batch_indices[i]]) for i in range(args.m)]
        return (model.pairwise, model.triplet, model.npair4)[args.m - 2](*batch_feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=50)
    parser.add_argument('--epochs', '-e', type=int, default=200)
    parser.add_argument('--save_iter', '-s', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--S_seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--cluster_std', type=float, default=0.25)
    parser.add_argument('--n', '-n', type=int, required=True, default='Number of training examples')
    parser.add_argument('--m', '-m', type=int, required=True, default='Number of variables per loss')
    args = parser.parse_args()
    print(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = BasicMLP().to(device)

    mask, train_indices, val_indices, all_examples, train_data, val_data = generate_data(args)
    train_tensor = (torch.tensor(train_data[0], dtype=torch.float, device=device), torch.tensor(train_data[1], device=device))
    val_tensor = (torch.tensor(val_data[0], dtype=torch.float, device=device), torch.tensor(val_data[1], device=device))
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    log_dir = f'results/{args.exp_name}/n={args.n},m={args.m},seed={args.seed},S_seed={args.S_seed}'
    num_batch = max(args.n // args.batch_size, 1)

    for epoch in tqdm(range(args.epochs)):
        model.train()

        train_loss = 0
        for batch in range(num_batch):
            batch_indices = sample_batch(args, train_data)
            loss = compute_loss(args, model, train_tensor, batch_indices)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Training Loss: {:0.6f}'.format(train_loss / num_batch))

        if (epoch + 1) % args.save_iter == 0:
            utils.save(model=model, path=os.path.join(log_dir, 'checkpoints', 'epoch{}.mdl'.format(epoch)))
            model.eval()

            val_loss = 0
            for batch in range(num_batch):
                batch_indices = sample_batch(args, val_data)
                loss = compute_loss(args, model, val_tensor, batch_indices)
                val_loss += loss.item()

            print('Validation Loss: {:0.6f}'.format(val_loss / num_batch))

    save_data = {
        'args': args,
        'mask': mask,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'all_examples': all_examples,
    }

    save_path = f'{log_dir}/saved_data.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(obj=save_data, file=f)


if __name__ == '__main__':
    main()

