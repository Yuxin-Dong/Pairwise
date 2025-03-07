import argparse
import json
import os.path
import pickle

from torch.utils.data import DataLoader
import numpy as np

from nnlib.nnlib import training, metrics, callbacks, utils
from nnlib.nnlib.data_utils.wrappers import SubsetDataWrapper, LabelSubsetWrapper, ResizeImagesWrapper, UniformNoiseWrapper
from nnlib.nnlib.data_utils.base import get_loaders_from_datasets, get_input_shape
from modules.data_utils import load_data_from_arguments, SwitchableRandomSampler, TurnOnTrainShufflingCallback
import methods


def mnist_ld_schedule(lr, beta, iteration):
    if iteration % 100 == 0:
        lr = lr * 0.9
    beta = min(4000, max(100, 10 * np.exp(iteration / 100)))
    return lr, beta


def cifar_resnet50_ld_schedule(lr, beta, iteration):
    if iteration % 300 == 0:
        lr = lr * 0.9
    beta = min(16000, max(100, 10 * np.exp(iteration / 300)))
    return lr, beta


def load_data(args):
    all_examples, _, _, _ = load_data_from_arguments(args, build_loaders=False)

    # select labels if needed
    if args.which_labels is not None:
        all_examples = LabelSubsetWrapper(all_examples, which_labels=args.which_labels)

        if args.error_prob > 0:
            all_examples = UniformNoiseWrapper(all_examples, error_prob=args.error_prob, num_classes=2, seed=args.seed)

    # resize if needed
    if args.resize_to_imagenet:
        all_examples = ResizeImagesWrapper(all_examples, size=(224, 224))

    # select 2n examples (tilde{z})
    assert len(all_examples) >= 2 * args.n
    np.random.seed(args.seed)
    include_indices = np.random.choice(range(len(all_examples)), size=2 * args.n, replace=False)
    all_examples = SubsetDataWrapper(all_examples, include_indices=include_indices)

    return all_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--all_device_ids', nargs='+', type=str, default=None,
                        help="If not None, this list specifies devices for multiple GPU training. "
                             "The first device should match with the main device (args.device).")
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--epochs', '-e', type=int, default=400)
    parser.add_argument('--stopping_param', type=int, default=2**30)
    parser.add_argument('--save_iter', '-s', type=int, default=10)
    parser.add_argument('--vis_iter', '-v', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--S_seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--deterministic', action='store_true', dest='deterministic')
    parser.add_argument('--shuffle_train_only_after_first_epoch', action='store_true',
                        dest="shuffle_train_only_after_first_epoch")

    # data parameters
    parser.add_argument('--dataset', '-D', type=str, default='corrupt4_mnist')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_augmentation', '-A', action='store_true', dest='data_augmentation')
    parser.set_defaults(data_augmentation=False)
    parser.add_argument('--error_prob', type=float, default=0.0)
    parser.add_argument('--n', '-n', type=int, required=True, default='Number of training examples')
    parser.add_argument('--which_labels', nargs='+', default=None, type=int)
    parser.add_argument('--clean_validation', action='store_true', default=False)
    parser.add_argument('--resize_to_imagenet', action='store_true', default=False)

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='StandardClassifier')
    parser.add_argument('--load_from', type=str, default=None)

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')

    parser.add_argument('--ld_lr', type=float, help='initial learning rate of Langevin dynamics')
    parser.add_argument('--ld_beta', type=float, help='initial inverse temperature of LD')
    parser.add_argument('--ld_track_grad_variance', dest='ld_track_grad_variance', action='store_true')
    parser.add_argument('--ld_track_every_iter', type=int, default=1)
    args = parser.parse_args()
    print(args)

    log_dir = f'results/{args.exp_name}/n={args.n},seed={args.seed},S_seed={args.S_seed}'
    if os.path.exists(log_dir):
        return

    # Load data
    all_examples = load_data(args)

    # select the train/val split (S)
    np.random.seed(args.S_seed)
    mask = np.random.randint(2, size=(args.n,))
    train_indices = 2*np.arange(args.n) + mask
    val_indices = 2*np.arange(args.n) + (1-mask)
    train_data = SubsetDataWrapper(all_examples, include_indices=train_indices)
    val_data = SubsetDataWrapper(all_examples, include_indices=val_indices)

    if args.deterministic:
        num_workers = 0  # to make sure data shuffling is always done the same way
    else:
        num_workers = 1

    train_loader, val_loader, test_loader = get_loaders_from_datasets(train_data=train_data,
                                                                      val_data=val_data,
                                                                      test_data=None,
                                                                      batch_size=args.batch_size,
                                                                      num_workers=num_workers)

    switchable_random_sampler = None
    if args.shuffle_train_only_after_first_epoch:
        switchable_random_sampler = SwitchableRandomSampler(data_source=train_data, shuffle=False)
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  num_workers=0, sampler=switchable_random_sampler)

    # Options
    optimization_args = {
        'optimizer': {
            'name': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum
        }
    }

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    model_class = getattr(methods, args.model_class)

    if args.deterministic:
        utils.set_seed(42)

    ld_schedule_fn = mnist_ld_schedule
    if args.exp_name == 'cifar10-pretrained-resnet50-LD':
        ld_schedule_fn = cifar_resnet50_ld_schedule

    model = model_class(input_shape=get_input_shape(train_loader.dataset),
                        architecture_args=architecture_args,
                        device=args.device,
                        load_from=args.load_from,
                        ld_lr=args.ld_lr,
                        ld_beta=args.ld_beta,
                        ld_schedule_fn=ld_schedule_fn,
                        ld_track_grad_variance=args.ld_track_grad_variance,
                        ld_track_every_iter=args.ld_track_every_iter)

    training.train(model=model,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   epochs=args.epochs,
                   save_iter=args.save_iter,
                   vis_iter=args.vis_iter,
                   optimization_args=optimization_args,
                   log_dir=log_dir,
                   args_to_log=args,
                   device_ids=args.all_device_ids)

    save_data = {
        'args': args,
        'mask': mask,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'all_examples': all_examples,
    }

    # if data augmentation is on, add a version of the dataset where data augmentation is disabled
    if args.data_augmentation:
        args.data_augmentation = False
        all_examples_wo_data_aug = load_data(args)
        save_data['all_examples'] = all_examples_wo_data_aug

    save_path = f'{log_dir}/saved_data.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(obj=save_data, file=f)


if __name__ == '__main__':
    main()
