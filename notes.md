
To Add:

* [done] Mixed precision
* [done] LR scheduler
* [done] Gradient accumulation
* [done] Data augmentation
* Decoupled weight decay
  * Decided not to do this because AdamW supports fused.
* [done] Checkpoints
* Progressive image resizing
* torch.compile (needs float16, which needs a scaler)
* Gradient clipping
* Channels last
* Sharpness-aware minimization


Augmentation Notes from SwinV2:
* Not using color jitter because we use autoaugment (see https://github.com/huggingface/pytorch-image-models/blob/56b90317cd9db1038b42ebdfc5bd81b1a2275cc1/timm/data/transforms_factory.py#L82-L86)

rand-m9-mstd0.5-inc1 means:

* rand -> use rand augmentation
* m9 -> magnitude 9
* mstd0.5 -> use 0.5 std dev for random noise
* inc1 -> use augmentations that increase in severity with magnitude

But the timm rand augmentation is so different from other implementations that it almost doesn't matter.
