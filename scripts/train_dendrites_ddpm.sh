#!/bin/bash 

#SBATCH --time=4:00:00 
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

# SUBSAMPLES=(
#     3000
# )

SEEDS=(
    9
    42
    87
    97
    99
)

# opts=()
# for subsample in "${SUBSAMPLES[@]}"
# do
#     for seed in "${SEEDS[@]}"
#     do
#             opts+=("$subsample;$seed")
#     done
# done

# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# echo "% Beginning..."
# echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# IFS=';' read -r -a opt <<< "${opts[${SLURM_ARRAY_TASK_ID}]}"
# subsample="${opt[0]}"
# seed="${opt[1]}"

#python train_ddpm.py --dataset DendriticFActinDataset --subsample 1000 --seed 97
python train_ddpm.py --dataset DendriticFActinDataset --seed ${SEEDS[${SLURM_ARRAY_TASK_ID}]}

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% DONE %"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"