"""
Simple training script for computer vision deep learning models.

* Minimizes cross-entropy loss for a pytorch vision model on an image dataset.
* Logs training statistics to wandb
* Saves only the latest and best checkpoints.
* Minimizes configuration.
"""
import os

import torch
import torchmetrics
import torchvision
from torch import nn
from torchvision import transforms

import utils
import wandb
from convnext import ConvNeXt
from swinv2 import SwinTransformerV2

# ------
# Config
# ------

num_classes = 10000

log_every = 10

data_path = "/local/scratch/cv_datasets/inat21/raw"
resize_size = (256, 256)
crop_size = (224, 224)
global_batch_size = 2048
gradient_accumulation_steps = 1
channel_mean = (0.4632, 0.4800, 0.3762)
channel_std = (0.2375, 0.2291, 0.2474)
rand_augment = utils.Map(
    num_ops=2, magnitude=9, interpolations=torchvision.InterpolationMode.BILINEAR
)
mixup_alpha = 1.0

# Tiny variants
swin = utils.Map(
    img_size=crop_size,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=7,
    drop_path_rate=0.2,
)
convnext = utils.Map(
    depths=(3, 3, 9, 3),
    dims=(96, 192, 384, 768),
)

# Training
label_smoothing = 0.1
learning_rate = 3e-4
weight_decay = 0.01
epochs = 300
warmup_steps = 1000

dtype = "bfloat16"
# PyTorch dtype
ptdtype = getattr(torch, dtype)
# If we change from bfloat16 to float16, we might need a grad scaler
ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------
# Distributed setup
# -----------------

is_ddp = int(os.environ.get("LOCAL_RANK", -1)) != -1  # is this a ddp run?
if is_ddp:
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_master = rank == 0
    seed_offset = rank
    torch.distributed.barrier()
else:
    rank = 0
    world_size = 0
    is_master = True
    seed_offset = 0

device = f"cuda:{rank}"
torch.cuda.set_device(device)

local_batch_size = global_batch_size // world_size // gradient_accumulation_steps

step = 0

# -------
# Logging
# -------

val_metrics = torchmetrics.MetricCollection(
    {
        "val/acc1": torchmetrics.Accuracy(
            task="multiclass", top_k=1, num_classes=num_classes
        ),
        "val/acc5": torchmetrics.Accuracy(
            task="multiclass", top_k=5, num_classes=num_classes
        ),
        "val/cross-entropy": utils.CrossEntropy(),
    }
).to(device)
train_metrics = torchmetrics.MetricCollection(
    {
        "train/acc1": torchmetrics.Accuracy(
            task="multiclass", top_k=1, num_classes=num_classes
        ),
        "train/acc5": torchmetrics.Accuracy(
            task="multiclass", top_k=5, num_classes=num_classes
        ),
        "train/cross-entropy": utils.CrossEntropy(),
    }
).to(device)


def log_batch(metrics, **kwargs):
    if not is_master:
        return

    wandb.log({**metrics.compute(), **kwargs})


def log_epoch(metrics):
    if not is_master:
        return

    wandb.log({**metrics.compute(), "epoch": epoch})


# ----
# Data
# ----


def build_dataloader(*, train):
    split = "train" if train else "val"
    drop_last = train
    shuffle = train

    # Transforms
    transform = torch.nn.Sequential()
    transform.append(transforms.Resize(resize_size, antialias=True))
    if train:
        transform.append(
            transforms.RandAugment(
                num_ops=rand_augment.num_ops,
                magnitude=rand_augment.magnitude,
                num_magnitude_bins=10,
                interpolation=rand_augment.interpolation,
            )
        )
        transform.append(
            transforms.RandomResizedCrop(
                crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0), antialias=True
            )
        )
        transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.CenterCrop(crop_size))
    transform.append(transforms.Normalize(mean=channel_mean, std=channel_std))
    # TODO: transform = torch.jit.script(transform)
    transform = transforms.Compose([transforms.ToTensor(), transform])

    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, split), transform
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        drop_last=drop_last,
        shuffle=shuffle,
        num_replicas=world_size,
        rank=rank,
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )


class Mixup(nn.Module):
    def __init__(self, alpha):
        self.dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha))

    @classmethod
    @torch.no_grad()
    def one_hot(self, y):
        # Make sure y.shape = (B, 1)
        y = y.long().view(-1, 1)
        batch, _ = y.shape
        # Fill with 0.0, then set to 1 in the target column
        return torch.full(
            (batch, num_classes), 0.0, device=y.device, dtype=y.dtype
        ).scatter(1, y, 1.0)

    @torch.no_grad()
    def forward(self, inputs, targets):
        lam = self.dist.sample()
        targets = self.one_hot(targets)

        inputs_flipped = inputs.flip(0).mul_(1 - lam)
        targets_flipped = targets.flip(0).mul_(1 - lam)

        # Do these ops in-place to save memory.
        inputs.mul_(lam).add_(inputs_flipped)
        targets.mul_(lam).add_(targets_flipped)

        return inputs, targets


# --------
# Training
# --------


def get_lr():
    # Linear warmup followed by constant LR.
    if step < warmup_steps:
        return learning_rate * step / warmup_steps

    return learning_rate


@torch.no_grad()
def val():
    print(f"Starting epoch {epoch} validation."
    model.eval()
    val_metrics.reset()

    for inputs, targets in val_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        with ctx:
            outputs = model(inputs)
        val_metrics(outputs, targets)

    val_metrics.compute()
    log_epoch(val_metrics)
    print(f"Finished epoch {epoch} validation."


def train():
    print(f"Starting epoch {epoch} training."
    global step
    model.train()
    mixup = Mixup(alpha=mixup_alpha)

    for batch, (inputs, targets) in enumerate(train_dataloader):
        # Whether we will actually do an optimization step
        will_step = batch % gradient_accumulation_steps == 0
        # Only need to sync if we're doing an optimization step
        model.require_backward_grad_sync = will_step

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = mixup(inputs, targets)

        with ctx:
            outputs = model(inputs)
        
        # Reset every step to calculate batch metrics
        train_metrics.reset()
        train_metrics(outputs, targets)

        loss = loss_fn(outputs, targets)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Update the learning rate while we wait for the forward pass
        lr = get_lr()
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if will_step:
            optimizer.step()
            # This has to be after optimizer.step().
            optimizer.zero_grad()
            step += 1

        train_metrics.compute()

        if batch % log_every == 0:
            # Need to use **{} because train/lr isn't a valid variable name
            log_batch(
                train_metrics,
                **{
                    "train/lr": lr,
                    "train/step": step,
                    "perf/total-images": step * global_batch_size,
                    "perf/batches": batch,
                },
            )

    print(f"Finished epoch {epoch} training."


if __name__ == "__main__":
    train_dataloader = build_dataloader(train=True)
    val_dataloader = build_dataloader(train=False)

    model = SwinTransformerV2(
        img_size=swin.img_size,
        num_classes=num_classes,
        drop_path_rate=swin.drop_path_rate,
        embed_dim=swin.embed_dim,
        depths=swin.depths,
        num_heads=swin.num_heads,
        window_size=swin.window_size,
    )
    # model = ConvNeXt(
    # num_classes=num_classes,
    # depths=convnext.depths,
    # dims=convnext.dims,
    # )
    model.to(device)
    model = torch.compile(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # TODO: decouple LR and weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if is_master:
        wandb.init(
            project="computer-vision-pretraining",
            entity="samuelstevens",
            config=dict(
                img_size=crop_size,
                num_classes=num_classes,
                # model options
                model="swinv2",
                drop_path_rate=swin.drop_path_rate,
                embed_dim=swin.embed_dim,
                depths=swin.depths,
                num_heads=swin.num_heads,
                window_size=swin.window_size,
                # depths=convnext.depths,
                # embed_dim=convnext.dims,
                global_batch_size=global_batch_size,
                local_batch_size=local_batch_size,
                world_size=world_size,
                # Data options
                data_path=data_path,
                resize_size=resize_size,
                crop_size=crop_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                channel_mean=channel_mean,
                channel_std=channel_std,
                rand_augment_num_ops=rand_augment.num_ops,
                rand_augment_magnitude=rand_augment.magnitude,
                mixup_alpha=mixup_alpha,
                # Training options
                epochs=epochs,
                label_smoothing=label_smoothing,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                amp=dtype,
            ),
            job_type="pretrain",
        )

    print("Starting training.")
    for epoch in range(epochs):
        train()
        val()

    print("Done.")
