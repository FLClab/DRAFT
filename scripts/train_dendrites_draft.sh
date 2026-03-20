#!/bin/bash 

#SBATCH --time=24:00:00 
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=8
#SBATCH --mem=16Gb
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=frbea320@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-39

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export TORCH_NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

#### PARAMETERS
module load python/3.12 scipy-stack
module load cuda cudnn
source ~/phd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SUBSAMPLES=(
    50
    50
    50
    50
    50
    100
    100
    100
    100
    100
    250
    250
    250
    250
    250
    300
    300
    300
    300
    300
    500
    500
    500
    500
    500
    1000
    1000
    1000
    1000
    1000
    2000
    2000
    2000
    2000
    2000
    3000
    3000
    3000
    3000
    3000
)

SEEDS=(
    9
    42
    87
    97
    99
    9
    42
    87
    97
    99
    9
    42
    87
    97
    99
    9
    42
    87
    97
    99
    9
    42
    87
    97
    99
    9
    42
    87
    97
    99
    9
    42
    87
    97
    99
    9
    42
    87
    97
    99
)

CHECKPOINTS=(
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-50-sample-9.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-50-sample-42.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-50-sample-87.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-50-sample-97.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-50-sample-99.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-100-sample-9.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-100-sample-42.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-100-sample-87.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-100-sample-97.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-100-sample-99.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-250-sample-9.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-250-sample-42.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-250-sample-87.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-250-sample-97.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-250-sample-99.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-300-sample-9.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-300-sample-42.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-300-sample-87.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-300-sample-97.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-300-sample-99.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-500-sample-9.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-500-sample-42.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-500-sample-87.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-500-sample-97.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-500-sample-99.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-1000-sample-9.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-1000-sample-42.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-1000-sample-87.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-1000-sample-97.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-1000-sample-99.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-2000-sample-9.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-2000-sample-42.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-2000-sample-87.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-2000-sample-97.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-2000-sample-99.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-3000-sample-9.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-3000-sample-42.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-3000-sample-87.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-3000-sample-97.pth"
    "/home/frbea320/scratch/baselines/DRAFT/DendriticFActin/DDPM_DendriticFActinDataset-3000-sample-99.pth"
)

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Beginning..."
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

subsample=${SUBSAMPLES[$SLURM_ARRAY_TASK_ID]}
seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
checkpoint=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

python train_draft.py --K 1 --ddim-ckpt $checkpoint --subsample $subsample --seed $seed

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"