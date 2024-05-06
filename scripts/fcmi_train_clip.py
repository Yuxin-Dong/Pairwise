import argparse
import json
import os
import pickle
import time

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image

import clip
import clip.model
from tqdm import tqdm

from nnlib.nnlib import utils
from nnlib.nnlib.data_utils.base import get_loaders_from_datasets
from nnlib.nnlib.data_utils.wrappers import SubsetDataWrapper


class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt, preprocess):
        self.image_path = list_image_path
        self.title = clip.tokenize(list_txt, truncate=True)  # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.preprocess = preprocess

        # record important attributes
        self.dataset_name = 'flickr'
        self.statistics = len(list_image_path)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        title = self.title[idx]
        return image, title


def load_data(args, preprocess):
    list_image_path, list_txt = [], []
    with open('flickr30k_images/results.csv') as captions:
        captions.readline()
        for line in captions.readlines():
            path_txt = line.split('|', maxsplit=2)
            if path_txt[1].endswith('0'):
                list_image_path.append('flickr30k_images/flickr30k_images/' + path_txt[0])
                list_txt.append(path_txt[2])

    all_examples = image_title_dataset(list_image_path, list_txt, preprocess)

    # select 2n examples (tilde{z})
    assert len(all_examples) >= 2 * args.n
    np.random.seed(args.seed)
    include_indices = np.random.choice(range(len(all_examples)), size=2 * args.n, replace=False)
    all_examples = SubsetDataWrapper(all_examples, include_indices=include_indices)

    return all_examples


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--save_iter', '-s', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--S_seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--n', '-n', type=int, required=True, default='Number of training examples')
    args = parser.parse_args()
    print(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
    model.args = args

    # Load data
    all_examples = load_data(args, preprocess)

    # select the train/val split (S)
    np.random.seed(args.S_seed)
    mask = np.random.randint(2, size=(args.n,))
    train_indices = 2*np.arange(args.n) + mask
    val_indices = 2*np.arange(args.n) + (1-mask)
    train_data = SubsetDataWrapper(all_examples, include_indices=train_indices)
    val_data = SubsetDataWrapper(all_examples, include_indices=val_indices)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    log_dir = f'results/{args.exp_name}/n={args.n},seed={args.seed},S_seed={args.S_seed}'

    # add your own code to track the training progress.
    for epoch in range(args.epochs):
        t0 = time.time()

        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            images, texts = batch

            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            # log some statistics
            t = time.time()
            log_string = 'Epoch: {}/{}'.format(epoch, args.epochs)
            log_string += ', Training Loss: {:0.6f}'.format(total_loss)
            log_string += ', Time: {:0.1f}s'.format(t - t0)
            print(log_string)

        # save the model according to our schedule
        if (epoch + 1) % args.save_iter == 0:
            utils.save(model=model, path=os.path.join(log_dir, 'checkpoints', 'epoch{}.mdl'.format(epoch)))

            t0 = time.time()

            model.eval()
            for batch in tqdm(val_loader):
                images, texts = batch

                images = images.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = model(images, texts)

                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

                # log some statistics
                t = time.time()
                log_string = 'Epoch: {}/{}'.format(epoch, args.epochs)
                log_string += ', Validation Loss: {:0.6f}'.format(total_loss)
                log_string += ', Time: {:0.1f}s'.format(t - t0)
                print(log_string)

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
