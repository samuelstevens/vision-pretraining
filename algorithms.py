import torch
import torch.nn.functional as F
from torch import nn


class Mixup(nn.Module):
    def __init__(self, alpha, num_classes):
        super().__init__()
        self.dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha))
        self.num_classes = num_classes

    @torch.no_grad()
    def one_hot(self, y, num_classes):
        # Make sure y.shape = (B, 1)
        y = y.long().view(-1, 1)
        B, _ = y.shape
        # Fill with 0.0, then set to 1 in the target column
        empty = torch.full((B, num_classes), 0.0, device=y.device)
        return empty.scatter(1, y, 1.0)

    @torch.no_grad()
    def forward(self, inputs, targets):
        lam = self.dist.sample()
        targets = self.one_hot(targets, self.num_classes)

        inputs_flipped = inputs.flip(0).mul_(1 - lam)
        targets_flipped = targets.flip(0).mul_(1 - lam)

        # Do input ops in-place to save memory.
        inputs.mul_(lam).add_(inputs_flipped)
        targets.mul_(lam).add_(targets_flipped)

        return inputs, targets


class ProgressiveResizing(nn.Module):
    def __init__(self, init_scale, init_steps, warmup_steps, size_inc):
        super().__init__()
        self.init_scale = init_scale
        self.init_steps = init_steps
        self.warmup_steps = warmup_steps
        self.slope = (1.0 - init_scale) / warmup_steps
        self.size_inc = size_inc

        self.step = 0

    def round(self, x):
        return self.size_inc * round(x / self.size_inc)

    @torch.no_grad()
    def forward(self, inputs):
        self.step += 1
        if self.step < self.init_steps:
            scale = self.init_scale
        elif self.step < self.init_steps + self.warmup_steps:
            scale = self.init_scale + (self.step - self.init_steps) * self.slope
        else:
            return inputs, 1.0

        B, C, W, H = inputs.shape
        assert C == 3, "inputs changed their format!"
        width, height = self.round(W * scale), self.round(H * scale)
        return F.interpolate(inputs, size=(width, height), mode="nearest"), scale

    def state_dict(self):
        return dict(step=self.step)

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
