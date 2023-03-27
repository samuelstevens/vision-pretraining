import collections.abc
from itertools import repeat

import torch
import torchmetrics


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class CrossEntropy(torchmetrics.Metric):
    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, targets) -> None:
        self.sum_loss += torch.nn.functional.cross_entropy(
            outputs, targets, ignore_index=self.ignore_index, reduction="mean"
        )
        self.total_batches += 1

    def compute(self):
        """Aggregate state over all processes and compute the metric."""
        # Return average loss over entire validation dataset
        assert isinstance(self.total_batches, torch.Tensor)
        assert isinstance(self.sum_loss, torch.Tensor)
        return self.sum_loss / self.total_batches


class Map(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise ValueError(attr)
