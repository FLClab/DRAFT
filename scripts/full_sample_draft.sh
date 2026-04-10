#!/bin/bash 

#SBATCH --time=24:00:00 
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=8
#SBATCH --mem=16Gb
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-4

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export TORCH_NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

#### PARAMETERS
module load python/3.12 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


SEEDS=(
    9
    42
    87
    97
    99
)

CHECKPOINTS=(
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-full-sample-9.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-full-sample-42.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-full-sample-87.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-full-sample-97.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-full-sample-99.pth"
)

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Beginning..."
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
checkpoint=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

python train_draft.py --K 1 --ddim-ckpt $checkpoint --subsample None --seed $seed

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"