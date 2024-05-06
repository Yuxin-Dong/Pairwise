import argparse
import os
import subprocess
import time

import torch

from nnlib.nnlib.data_utils.base import register_parser as register_fn


local_functions = {}  # storage for registering the functions below


def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(0.1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


#######################################################################################
#
#     MNIST 4 vs 9, CNN
#
# exp_name: fcmi-mnist-4vs9-CNN
#######################################################################################

@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN')
def foo(**kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 128
    n_epochs = 100
    save_iter = 20
    exp_name = "fcmi-mnist-4vs9-CNN"
    dataset = 'mnist'
    which_labels = '4 9'

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                commands.append(command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};")
    return commands

@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN-noise')
def foo(error_prob, **kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 128
    n_epochs = 100
    save_iter = 20
    exp_name = f"fcmi-mnist-4vs9-CNN-{error_prob}"
    dataset = 'mnist'
    which_labels = '4 9'

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --error_prob {error_prob} "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                commands.append(command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};")
    return commands


@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN-LD')
def foo(**kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 100
    n_epochs = 40
    save_iter = 4
    exp_name = "fcmi-mnist-4vs9-CNN-LD"
    dataset = 'mnist'
    which_labels = '4 9'
    ld_lr = 0.004
    ld_beta = 10.0

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} -m LangevinDynamics --ld_lr {ld_lr} "\
                     f"--ld_beta {ld_beta} "

    n_seeds = 5
    n_S_seeds = 30
    ns = [4000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                commands.append(command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};")
    return commands


@register_fn(local_functions, 'cifar10-pretrained-resnet50')
def foo(**kwargs):
    config_file = 'configs/pretrained-resnet50-cifar10.json'
    batch_size = 64
    n_epochs = 40
    save_iter = n_epochs
    exp_name = "cifar10-pretrained-resnet50"
    dataset = 'cifar10'
    optimizer = 'sgd'
    lr = 0.01
    momentum = 0.9

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"-m StandardClassifier -A --resize_to_imagenet --optimizer {optimizer} "\
                     f"--lr {lr} --momentum {momentum} "

    n_seeds = 1
    n_S_seeds = 40
    ns = [1000, 5000, 20000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                commands.append(command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};")
    return commands


@register_fn(local_functions, 'flickr-pretrained-clip')
def foo(**kwargs):
    batch_size = 512
    n_epochs = 40
    save_iter = 10
    exp_name = "flickr-pretrained-clip"

    command_prefix = f"python -um scripts.fcmi_train_clip -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} --exp_name {exp_name} "

    n_seeds = 1
    n_S_seeds = 40
    ns = [1000, 4000, 15000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                commands.append(command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};")
    return commands


@register_fn(local_functions, 'synthetic-MLP')
def foo(**kwargs):
    batch_size = 50
    n_epochs = 200
    save_iter = 200
    exp_name = "synthetic-MLP"

    command_prefix = f"python -um scripts.train_synthetic -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} --exp_name {exp_name} "

    n_seeds = 1
    n_S_seeds = 200
    ns = [20, 40, 60, 80, 100]
    ms = [2, 3, 4]

    commands = []
    for n in ns:
        for m in ms:
            for seed in range(n_seeds):
                for S_seed in range(n_S_seeds):
                    commands.append(command_prefix + f"--n {n} --m {m} --seed {seed} --S_seed {S_seed};")
    return commands


@register_fn(local_functions, 'synthetic-MLP-noise')
def foo(**kwargs):
    batch_size = 50
    n_epochs = 200
    save_iter = 200
    exp_name = "synthetic-MLP-noise"

    command_prefix = f"python -um scripts.train_synthetic -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} --exp_name {exp_name} --cluster_std 0.5 "

    n_seeds = 1
    n_S_seeds = 200
    ns = [20, 40, 60, 80, 100]
    ms = [1]

    commands = []
    for n in ns:
        for m in ms:
            for seed in range(n_seeds):
                for S_seed in range(n_S_seeds):
                    commands.append(command_prefix + f"--n {n} --m {m} --seed {seed} --S_seed {S_seed};")
    return commands


def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_names', '-E', type=str, nargs='+', required=True)
    parser.add_argument('--error_prob', type=float, default=0.0)
    parser.add_argument('--skip_confirm', action='store_true')
    args = parser.parse_args()

    for exp_name in args.exp_names:
        assert exp_name in local_functions
        commands = local_functions[exp_name](**vars(args))
        for command in commands:
            print(command)
        if not args.skip_confirm:
            ask_for_confirmation()
        multi_gpu_launcher(commands)


if __name__ == '__main__':
    main()
