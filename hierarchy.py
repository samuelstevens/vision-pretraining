import dataclasses
import os

import torch
import torchmetrics
import torchvision.datasets
from torch import nn

import algorithms
import utils


class MultitaskHead(nn.Module):
    """
    Hierarchical multitask head

    Adds a linear layer for each "tier" in the hierarchy.

    forward() returns a list of logits for each tier.

    Arguments:
        num_features (int): number of features from the backbone
        num_classes (tuple[int, ...]): a tuple of each number of classes in the hierarchy.
    """

    def __init__(self, n_features, num_classes):
        super().__init__()

        self.heads = nn.ModuleList()
        for n_cls in num_classes:
            assert n_cls > 0
            self.heads.append(nn.Linear(n_features, n_cls))

    def forward(self, x):
        # we do not want to use self.heads(x) because that would feed them through
        # each element in the list sequentially, whereas we want x through each head
        # individually.
        return [head(x) for head in self.heads]


def multitask_surgery(model, head: str, num_classes):
    """
    Replaces the head with a MultitaskHead.
    """
    if not hasattr(model, head):
        raise RuntimeError(f"model has no attribute {head}!")

    # We use max because we know the number of classes is 2 (in train_hierarchy.py).
    # So we can pick the bigger one because all the models we're using will
    # always have more than 2 features.
    num_features = max(getattr(model, head).weight.shape)

    setattr(model, head, MultitaskHead(num_features, num_classes))


class MultitaskCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, *args, coeffs=(1.0,), **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.clone().detach().type(torch.float)
        else:
            coeffs = torch.tensor(coeffs, dtype=torch.float)

        self.register_buffer("coeffs", coeffs)

    def forward(self, inputs, targets):
        assert isinstance(targets, list)
        assert (
            len(inputs) == len(targets) == len(self.coeffs)
        ), f"{len(inputs)} != {len(targets)} != {len(self.coeffs)}"

        losses = [
            # Need to specify arguments to super() because of some
            # bug with super() in list comprehensions (unclear).
            super(MultitaskCrossEntropy, self).forward(i, t)
            for i, t in zip(inputs, targets)
        ]
        losses = torch.stack(losses)
        return torch.dot(self.coeffs, losses)


class FineGrainedAccuracy(torchmetrics.Metric):
    is_differentiable = False
    higher_is_better = True
    # Try turning this off and see if it improves performance while producing the same results.
    # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html#internal-implementation-details
    full_state_update = True

    def __init__(self, top_k=1):
        super().__init__()
        self.top_k = top_k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: list[torch.Tensor], targets: torch.Tensor):
        assert isinstance(outputs, list)
        assert targets.size(1) == 7, f"Should be 7 tiers, not {targets.size(1)}"
        targets = targets[:, -1]

        # B x K
        preds = fine_grained_predictions(outputs, top_k=self.top_k).view(-1, self.top_k)
        # B x K
        targets = targets.view(-1, 1).expand(preds.shape)

        self.correct += torch.sum(preds == targets)
        self.total += targets.numel() // self.top_k  # B

    def compute(self):
        return self.correct.float() / self.total


class FineGrainedCrossEntropy(utils.CrossEntropy):
    """
    A cross-entropy used with hierarchical inputs and targets and only
    looks at the finest-grained tier (the last level).
    """

    def update(self, preds: list[torch.Tensor], targets: torch.Tensor):
        if not isinstance(preds, list):
            raise RuntimeError("FineGrainedCrossEntropy needs a list of predictions")
        preds, targets = preds[-1], targets[:, -1]
        super().update(preds, targets)


class ImageFolder(torchvision.datasets.ImageFolder):
    """
    Parses an image folder where the hierarchy is represented as follows:

    00000_top_middle_..._bottom
    00001_top_middle_..._other
    ...
    """

    num_classes = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_classes(self, directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )

        tier_lookup = {}
        class_to_idxs = {}

        for cls in classes:
            tiers = HierarchicalLabel.parse(cls).clean_tiers

            for tier, value in enumerate(tiers):
                if tier not in tier_lookup:
                    tier_lookup[tier] = {}

                if value not in tier_lookup[tier]:
                    tier_lookup[tier][value] = len(tier_lookup[tier])

            class_to_idxs[cls] = torch.tensor(
                [tier_lookup[tier][value] for tier, value in enumerate(tiers)]
            )

        # Set self.num_classes
        self.num_classes = tuple(len(tier) for tier in tier_lookup.values())

        return classes, class_to_idxs


@dataclasses.dataclass(frozen=True)
class HierarchicalLabel:
    raw: str
    number: int
    kingdom: str
    phylum: str
    cls: str
    order: str
    family: str
    genus: str
    species: str

    @classmethod
    def parse(cls, name):
        """
        Sometimes the tree is not really a tree. For example, sometimes there are repeated orders.
        This function fixes that by repeating the upper tier names in the lower tier names.

        Suppose we only had order-level classification. This would be the class for a bald eagle's
        order:

        00001_animalia_chordata_aves_accipitriformes

        Suppose then that we had a another class:

        00002_animalia_chordata_reptilia_accipitriformes

        These accipitriformes refer to different nodes in the tree. To fix this, we do:

        00001_animalia_chordata_aves_accipitriformes ->
            00001, animalia, animalia-chordata, animalia-chordata-aves, animalia-chordata-aves-accipitriformes

        00002_animalia_chordata_reptilia_accipitriformes ->
            00002, animalia, animalia-chordata, animalia-chordata-reptilia, animalia-chordata-reptilia-accipitriformes

        Now each bit of text refers to the same nodes.
        It's not pretty but it does get the job done.

        Arguments:
            name (str): the complete taxonomic name, separated by '_'
        """

        # index is a number
        # top is kingdom
        index, top, *tiers = name.split("_")
        number = int(index)

        cleaned = [top]

        complete = top
        for tier in tiers:
            complete += f"-{tier}"
            cleaned.append(complete)

        assert len(cleaned) == 7, f"{len(cleaned)} != 7"

        return cls(name, number, *cleaned)

    @property
    def cleaned(self):
        return "_".join(
            [
                str(self.number).rjust(5, "0"),
                self.kingdom,
                self.phylum,
                self.cls,
                self.order,
                self.family,
                self.genus,
                self.species,
            ]
        )

    @property
    def clean_tiers(self):
        return [
            self.kingdom,
            self.phylum,
            self.cls,
            self.order,
            self.family,
            self.genus,
            self.species,
        ]


def fine_grained_predictions(output, top_k=1, hierarchy_level=-1):
    """
    Computes the top k predictions for a hierarchical output

    Copied from rwightman/pytorch-image-models/timm/utils/metrics.py and modified
    to work with hierarchical outputs as well.

    When the output is hierarchical, only returns the accuracy for `hierarchy_level`
    (default -1, which is the fine-grained level).
    """
    if isinstance(output, list):
        output = output[hierarchy_level]

    batch_size, num_classes = output.shape

    maxk = min(top_k, num_classes)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    return pred


class Mixup(algorithms.Mixup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.num_classes) == 7

    @torch.no_grad()
    def forward(self, inputs, targets):
        """
        inputs (B C W H): input images
        targets (B 7): target indices
        """
        assert len(inputs) % 2 == 0, "Batch size should be even"
        assert targets.size(1) == 7, f"Should be 7 tiers, not {targets.size(1)}"

        targets = [self.one_hot(t, n) for t, n in zip(targets.T, self.num_classes)]

        lam = self.dist.sample()

        inputs_flipped = inputs.flip(0).mul_(1 - lam)
        targets_flipped = [t.flip(0).mul_(1 - lam) for t in targets]

        # Do input ops in-place to save memory.
        inputs.mul_(lam).add_(inputs_flipped)
        for t, t_flipped in zip(targets, targets_flipped):
            t.mul_(lam).add_(t_flipped)

        return inputs, targets
