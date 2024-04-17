from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

from aigames.agent.alpha_agent_multi import BaseAlphaEvaluator
from aigames.training_manager.alpha_training_manager_multi_lightning import AlphaMultiNetwork


class Connect4Network(AlphaMultiNetwork, BaseAlphaEvaluator):

    @dataclass(kw_only=True, slots=True)
    class Hyperparameters:
        n_blocks: int = 4
        n_channels: int = 64
        n_out_channels: int = 64

    def __init__(self, hyperparams: Hyperparameters = Hyperparameters()):
        super().__init__()
        self.hyperparams = hyperparams

        self.base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.hyperparams.n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hyperparams.n_channels),
            nn.ReLU(),
            *[BasicBlock(self.hyperparams.n_channels, self.hyperparams.n_channels) for _ in range(self.hyperparams.n_blocks)],
            nn.Conv2d(in_channels=self.hyperparams.n_channels, out_channels=self.hyperparams.n_out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.Flatten()
        )

        self.n_base_out_features = self.hyperparams.n_out_channels * 6 * 7

        self.policy_logits_head = nn.Sequential(
            nn.Linear(in_features=self.n_base_out_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=7)
        )

        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.n_base_out_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2),
            nn.Tanh()
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy_logits = self.policy_logits_head(base)
        values = self.value_head(base)
        return policy_logits, values

    def process_state(self, state):
        return state[:, 1:, :].unsqueeze(1).float()

    @torch.no_grad()
    def evaluate(self, state):
        if state.shape[1] != 1:
            state = self.process_state(state)

        pi_logits, value = self.forward(state.to(self.device))

        pi = torch.softmax(pi_logits, dim=1)
        return pi, value

    def loss(self, states, pis, values, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_distn_logits, pred_values = self.forward(states)

        nan_distns = torch.isnan(pis).any(dim=1)

        value_loss = nn.functional.mse_loss(pred_values, values, reduction='none')
        distn_loss = nn.functional.cross_entropy(pred_distn_logits[~nan_distns], pis[~nan_distns],
                                                 reduction='none')

        if len(args) > 0:
            importance_sampling_weights = args[0].to(pred_distn_logits.device)
            value_loss *= importance_sampling_weights
            distn_loss *= importance_sampling_weights[~nan_distns]

        value_loss = value_loss.mean()
        distn_loss = distn_loss.mean()

        return value_loss, distn_loss

    @property
    def device(self):
        return next(self.parameters()).device
