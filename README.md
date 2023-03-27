# Vision Model Pretraining

Basic code for pretraining a large(ish) vision model with classification.
Extremely simple, minimal configuration.

## Install

```
pyenv local 3.10.9
python -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

## Train Models

With one GPU:

```
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 train.py
```

With four GPUs:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 train.py
```

On a Slurm system:

```
sbatch --output=logs/%j-pretrain.txt --job-name=pretrain slurm/train.bash
```

If you modify this command, change the command in train.bash (because it recursively calls itself).
