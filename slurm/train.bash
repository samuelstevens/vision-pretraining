#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=48
#SBATCH --signal=B:USR1@60

source $HOME/projects/vision/.venv/bin/activate

function submit_again() {
  sbatch \
    --output=logs/%j-pretrain.txt \
    --job-name=pretrain \
    slurm/train.bash
}

trap submit_again USR1
trap submit_again TERM

# Launch in background and then `wait` for it to finish.
# Then we can catch interrupt signals and submit another job.
# See https://www.osc.edu/supercomputing/batch-processing-at-osc/job-scripts
# specifically the section on Signal Handling for details.
torchrun --standalone --nproc-per-node=4 train.py &
wait
