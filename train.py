"""
Simple training script for computer vision deep learning models.

* Minimizes cross-entropy loss for a pytorch vision model on an image dataset.
* Logs training statistics to wandb
* Saves only the latest and best checkpoints.
* Minimizes configuration.
"""
import heapq
import logging
import os

import torch
import torchmetrics
import torchvision
from torchvision import transforms

import algorithms
import utils
import wandb
from convnext import ConvNeXt
from swinv2 import SwinTransformerV2

# ------
# Config
# ------

num_classes = 10_000

data_root = "/local/scratch/cv_datasets/inat21/raw"
resize_size = (256, 256)
crop_size = (224, 224)
global_batch_size = 2048
gradient_accumulation_steps = 1
channel_mean = (0.4632, 0.4800, 0.3762)
channel_std = (0.2375, 0.2291, 0.2474)
rand_augment_num_ops = 2
rand_augment_magnitude = 9
mixup_alpha = 1.0

# Tiny variants
swin = utils.DotDict(
    img_size=crop_size,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=7,
    drop_path_rate=0.2,
)
convnext = utils.DotDict(
    depths=(3, 3, 9, 3),
    embed_dim=(96, 192, 384, 768),
    drop_path_rate=0.1,
)

# Training
label_smoothing = 0.1
learning_rate = 3e-4
weight_decay = 0.01
grad_clip = 2.0
epochs = 300
warmup_steps = 1000
# Other option is torch.contiguous_format; use channels_last for convnets
memory_format = torch.channels_last

pr_init_scale = 0.5
pr_init_steps = 100_000
pr_warmup_steps = 100_000
pr_size_inc = 4

dtype = "float16"
# PyTorch dtype
ptdtype = getattr(torch, dtype)
ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
# Don't need scaler for bfloat16 because it's higher range, lower precision.
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

save_root = "/local/scratch/stevens.994/vision/checkpoints"
log_every = 10
n_latest_checkpoints = 3
n_best_checkpoints = 2
# Needs to be a key in val_metrics and a value we maximize (not minimize)
best_metric = "val/acc1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------
# Distributed setup
# -----------------

is_ddp = int(os.environ.get("LOCAL_RANK", -1)) != -1
if is_ddp:
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_master = rank == 0
    seed_offset = rank
    torch.distributed.barrier()
else:
    rank = 0
    world_size = 1
    is_master = True
    seed_offset = 0

device = f"cuda:{rank}"
torch.cuda.set_device(device)

local_batch_size = global_batch_size // world_size // gradient_accumulation_steps
step = 0

# -------
# Logging
# -------

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(f"Rank {rank}")

task = "multiclass"
val_metrics = torchmetrics.MetricCollection(
    {
        "val/acc1": torchmetrics.Accuracy(task=task, top_k=1, num_classes=num_classes),
        "val/acc5": torchmetrics.Accuracy(task=task, top_k=5, num_classes=num_classes),
        "val/cross-entropy": utils.CrossEntropy(),
    }
).to(device)
train_metrics = torchmetrics.MetricCollection(
    {
        "train/acc1": torchmetrics.Accuracy(
            task=task, top_k=1, num_classes=num_classes
        ),
        "train/acc5": torchmetrics.Accuracy(
            task=task, top_k=5, num_classes=num_classes
        ),
    }
).to(device)

assert best_metric in val_metrics, f"Can't optimize for {best_metric}!"


# ----
# Data
# ----


def build_dataloader(*, train):
    split = "train" if train else "val"
    drop_last = train
    shuffle = train

    # Transforms
    transform = []
    transform.append(transforms.Resize(resize_size, antialias=True))
    if train:
        train_transforms = [
            transforms.RandomResizedCrop(
                crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(
                num_ops=rand_augment_num_ops,
                magnitude=rand_augment_magnitude,
                num_magnitude_bins=10,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
        transform.extend(train_transforms)
    else:
        transform.append(transforms.CenterCrop(crop_size))
    common_transforms = [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=channel_mean, std=channel_std),
    ]
    transform.extend(common_transforms)
    transform = transforms.Compose(transform)

    dataset_path = os.path.join(data_root, split)
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform)
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
        # TODO: Evaluate whether this helps throughput or not on A100 and on A6000.
        pin_memory=True,
        persistent_workers=True,
    )


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
    logger.info(f"Starting epoch {epoch} validation.")
    model.eval()
    val_metrics.reset()

    for inputs, targets in val_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        with ctx:
            outputs = model(inputs)
        val_metrics(outputs, targets)

    val_metrics.compute()
    if is_master:
        wandb.log({**val_metrics.compute(), "perf/epoch": epoch})
    logger.info(f"Finished epoch {epoch} validation.")


def train():
    logger.info(f"Starting epoch {epoch} training.")
    global step
    model.train()
    mixup = algorithms.Mixup(alpha=mixup_alpha, num_classes=num_classes)

    for batch, (inputs, targets) in enumerate(train_dataloader):
        # Whether we will actually do an optimization step
        will_step = batch % gradient_accumulation_steps == 0
        # Only need to sync if we're doing an optimization step
        model.require_backward_grad_sync = will_step

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, soft_targets = mixup(inputs, targets)
        inputs, scale = resizer(inputs)

        with ctx:
            outputs = model(inputs)

        # Reset every step to calculate batch metrics
        train_metrics.reset()
        train_metrics(outputs, targets)

        loss = loss_fn(outputs, soft_targets)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        # Update the learning rate while we wait for the forward pass
        lr = get_lr()
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if will_step:
            # Clip gradients
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Step optimizer
            scaler.step(optimizer)
            scaler.update()
            # This has to be after optimizer.step() or scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            step += 1

        train_metrics.compute()

        if batch % (log_every * gradient_accumulation_steps) == 0 and is_master:
            # Need to use **{} because train/lr isn't a valid variable name
            # We can use loss.item() because we don't need to sync it across
            # all processes since it's going to be noisy anyways.
            metrics = {
                **train_metrics.compute(),
                "train/lr": lr,
                "train/cross-entropy": loss.item(),
                "train/step": step,
                "perf/total-images": step * global_batch_size,
                "perf/batches": batch,
                "train/grad": grad,
                "train/img-scale": scale,
            }
            wandb.log(metrics)

    logger.info(f"Finished epoch {epoch} training.")


def save():
    if not is_master:
        return

    logger.info("Saving checkpoint.")
    ckpt = {
        "model": model_without_ddp.state_dict(),
        "optim": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        **val_metrics.compute(),
    }
    directory = os.path.join(save_root, run_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"ep{epoch}.pt")
    torch.save(ckpt, path)
    logger.info("Saved to %s.", path)

    # Prune checkpoints: keep n_best_checkpoints and n_latest_checkpoints
    # ------------------

    # Load all checkpoints and select which ones to keep.
    all_ckpts = {path: ckpt}
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        all_ckpts[path] = torch.load(path, map_location="cpu")

    # Get the best and latest checkpoints
    best_ckpts = heapq.nlargest(
        n_best_checkpoints, all_ckpts, key=lambda c: all_ckpts[c][best_metric]
    )
    best_ckpts = set(best_ckpts)
    latest_ckpts = heapq.nlargest(
        n_latest_checkpoints, all_ckpts, key=lambda c: all_ckpts[c]["epoch"]
    )
    latest_ckpts = set(latest_ckpts)

    # If not one of the best or latest, remove it
    for path in all_ckpts:
        if path in best_ckpts:
            logger.info(
                "Keep %s. It's the top %d for %s.",
                path,
                n_best_checkpoints,
                best_metric,
            )
            continue

        if path in latest_ckpts:
            logger.info("Keep %s. It's in %d latest.", path, n_latest_checkpoints)
            continue

        os.remove(path)
        logger.info("Removed %s.", path)


def restore():
    # Restores the latest checkpoint.
    latest_ckpt = None
    directory = os.path.join(save_root, run_id)
    if not os.path.isdir(directory):
        raise RuntimeError(f"No checkpoint directory at {directory}.")

    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        ckpt = torch.load(path, map_location=device)
        if not latest_ckpt or ckpt["epoch"] > latest_ckpt["epoch"]:
            latest_ckpt = ckpt

    if not latest_ckpt:
        raise RuntimeError(f"No saved checkpoints in {directory}.")

    global model, optimizer, start, step
    model_without_ddp.load_state_dict(latest_ckpt["model"])
    optimizer.load_state_dict(latest_ckpt["optim"])
    # Add one because we save after we finish an epoch
    start = latest_ckpt["epoch"] + 1
    step = latest_ckpt["step"]


# ---------------
# Training Script
# ---------------


if __name__ == "__main__":
    # First epoch
    start = 0

    train_dataloader = build_dataloader(train=True)
    val_dataloader = build_dataloader(train=False)

    model = ConvNeXt(
        num_classes=num_classes,
        drop_path_rate=convnext.drop_path_rate,
        embed_dim=convnext.embed_dim,
        depths=convnext.depths,
    )
    model_without_ddp = model
    model.to(device, memory_format=memory_format)
    # compile doesn't support bfloat16: https://github.com/pytorch/pytorch/issues/97016
    if dtype != "bfloat16":
        model = torch.compile(model)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        # fused doesn't work with channels_last (bug? not sure)
        fused=(memory_format != torch.channels_last),
    )

    resumed_and_id = [False, None]
    if is_master:
        run = wandb.init(
            project="computer-vision-pretraining",
            entity="samuelstevens",
            config=dict(
                img_size=crop_size,
                num_classes=num_classes,
                # model options
                model="convnext",
                depths=convnext.depths,
                embed_dim=convnext.embed_dim,
                drop_path_rate=convnext.drop_path_rate,
                global_batch_size=global_batch_size,
                local_batch_size=local_batch_size,
                world_size=world_size,
                # data options
                data_root=data_root,
                resize_size=resize_size,
                crop_size=crop_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                channel_mean=channel_mean,
                channel_std=channel_std,
                rand_augment_num_ops=rand_augment_num_ops,
                rand_augment_magnitude=rand_augment_magnitude,
                mixup_alpha=mixup_alpha,
                # training options
                epochs=epochs,
                label_smoothing=label_smoothing,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                # progressive resizing options
                pr_init_scale=pr_init_scale,
                pr_init_steps=pr_init_steps,
                pr_warmup_steps=pr_warmup_steps,
                pr_size_inc=pr_size_inc,
                amp=dtype,
                grad_clip=grad_clip,
                memory_format=str(memory_format),
                debug=False,
            ),
            job_type="pretrain",
            resume=False,
        )
        resumed_and_id = run.resumed, run.id

    # Now non-master processes have resumed and run_id
    # Refer to https://github.com/pytorch/pytorch/issues/56142
    # for why we need a variable instead of an anonymous list
    if is_ddp:
        torch.distributed.broadcast_object_list(resumed_and_id)
    resumed, run_id = resumed_and_id

    if resumed:
        assert run_id is not None, "If we resume, we need a run.id"
        restore()

    resizer = algorithms.ProgressiveResizing(
        init_scale=pr_init_scale,
        init_steps=pr_init_steps,
        warmup_steps=pr_warmup_steps,
        size_inc=pr_size_inc,
    )

    logger.info("Starting training from epoch %s.", start)
    for epoch in range(start, epochs):
        train()
        val()
        save()

    logger.info("Done.")
