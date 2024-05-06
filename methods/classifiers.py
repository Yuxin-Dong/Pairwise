import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torchvision.models import ResNet

from nnlib.nnlib.utils import capture_arguments_of_init
from nnlib.nnlib import losses, utils
from nnlib.nnlib.gradients import get_weight_gradients
from methods import BaseClassifier
from modules import nn_utils


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        device = features.device

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits = dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = 1 - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        pos_mask = mask.sum(1) > 0
        mean_log_prob_pos = (mask * log_prob).sum(1)[pos_mask] / mask.sum(1)[pos_mask]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        return loss.mean()


class StandardClassifier(BaseClassifier):
    @capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, device='cuda', **kwargs) -> object:
        super(StandardClassifier, self).__init__(**kwargs)

        self.args = None  # this will be modified by the decorator
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args

        # create the network
        self.classifier, output_shape = nn_utils.parse_network_from_config(args=self.architecture_args['classifier'],
                                                                           input_shape=self.input_shape)
        if isinstance(self.classifier, ResNet):
            self.classifier.fc = nn.Sequential(
                nn.Linear(self.classifier.fc.in_features, self.classifier.fc.in_features),
                nn.ReLU(inplace=True),
                nn.Linear(self.classifier.fc.in_features, 128)
            )

        self.num_classes = output_shape[-1]
        self.classifier = self.classifier.to(device)
        self.contrastive_loss = SupConLoss()

    def forward(self, inputs, labels=None, grad_enabled=False,
                detailed_output=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        x = inputs[0].to(self.device)

        features = self.classifier(x)
        features = features / features.norm(dim=1, keepdim=True)

        return features

    def compute_loss(self, inputs, labels, outputs, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        y = labels[0].to(self.device)

        contrastive = self.contrastive_loss(outputs, y)

        batch_losses = {
            'contrastive': contrastive,
        }

        return batch_losses, outputs


class LangevinDynamics(StandardClassifier):
    @capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, device='cuda',
                 ld_lr=None, ld_beta=None, ld_schedule_fn=None,
                 ld_track_grad_variance=True, ld_track_every_iter=1, **kwargs) -> object:
        super(LangevinDynamics, self).__init__(input_shape=input_shape,
                                               architecture_args=architecture_args,
                                               device=device,
                                               **kwargs)
        self.lr = ld_lr
        self.beta = ld_beta
        self.schedule_fn = ld_schedule_fn
        self.track_grad_variance = ld_track_grad_variance
        self.track_every_iter = ld_track_every_iter

        self._iteration = 0
        self._lr_hist = [self.lr]
        self._beta_hist = [self.beta]
        self._grad_variance_hist = []

    @utils.with_no_grad
    def on_iteration_end(self, partition, tensorboard, loader, **kwargs):
        if partition != 'train':
            return
        # update lr and beta
        self._iteration += 1
        self.lr, self.beta = self.schedule_fn(lr=self.lr, beta=self.beta, iteration=self._iteration)
        self._lr_hist.append(self.lr)
        self._beta_hist.append(self.beta)
        tensorboard.add_scalar('LD_lr', self.lr, self._iteration)
        tensorboard.add_scalar('LD_beta', self.beta, self._iteration)

        # compute gradient variance
        if self.track_grad_variance:
            if (self._iteration - 1) % self.track_every_iter == 0:
                grads = get_weight_gradients(model=self, dataset=loader.dataset,
                                             max_num_examples=100,   # using 100 examples for speed
                                             use_eval_mode=True,
                                             random_selection=True)
                self.train()  # back to training mode
                grads_flattened = []
                for sample_idx in range(min(100, len(loader.dataset))):
                    cur_grad = [grads[k][sample_idx].flatten() for k in grads.keys()]
                    grads_flattened.append(torch.cat(cur_grad, dim=0))
                grads = torch.stack(grads_flattened)
                del grads_flattened
                mean_grad = torch.mean(grads, dim=0, keepdim=True)
                grad_variance = torch.sum((grads - mean_grad)**2, dim=1).mean(dim=0)
                self._grad_variance_hist.append(grad_variance)
                tensorboard.add_scalar('LD_grad_variance', grad_variance, self._iteration)
            else:
                self._grad_variance_hist.append(self._grad_variance_hist[-1])

    def before_weight_update(self, **kwargs):
        # manually doing the noisy gradient update
        for k, v in dict(self.named_parameters()).items():
            eps = torch.normal(mean=0.0, std=1/self.beta, size=v.grad.shape,
                               device=self.device, dtype=torch.float)
            update = -self.lr * v.grad + np.sqrt(2 * self.lr / self.beta) * eps
            v.data += update.data
            v.grad.zero_()

    def attributes_to_save(self):
        return {
            '_lr_hist': self._lr_hist,
            '_beta_hist': self._beta_hist,
            '_grad_variance_hist': self._grad_variance_hist
        }
