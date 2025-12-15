# DRAFT with STED-FM:

A partial PyTorch implementation of the DRAFT paper (*Directly Fine-Tuning Diffusion Models on Differentiable Rewards*, Clark et al., ICLR 2024) that uses [STED-FM](https://github.com/FLClab/STED-FM) as a reward encoder. DRAFT backpropagates through the entire sampling process to directly optimize a reward function—in this case, perceptual similarity computed using STED-FM embeddings.

### Key Features

- **Reward-based fine-tuning**: Directly optimize generated images for perceptual similarity using STED-FM embeddings
- **LoRA integration**: Parameter-efficient fine-tuning (~95% parameter reduction)
- **Gradient checkpointing**: Memory-efficient training through the sampling chain
- **Multiple variants**: Support for DRaFT (full) and DRaFT-K 
- **DDIM sampling**: Differentiable deterministic sampling for stable training

## Project Structure

```
DRAFT/
├── diffusion_model_draft.py   # DRaFT_DDPM class and RewardEncoder
├── diffusion_model.py         # Base DDPM implementation
├── denoising_unet.py          # UNet architecture for denoising
├── lora_layers.py             # LoRA implementation for parameter-efficient training
├── train_draft.py             # Main training script
├── train_dm.py                # Standard diffusion model training
├── test_draft.py              # Unit tests
├── evaluate_axons.py          # Evaluation utilities
├── segmentation_unet.py       # Segmentation model
├── datasets/                  # Dataset loaders
│   ├── axons_dataset.py
│   ├── dendrites_dataset.py
│   └── synaptic_protein_dataset.py
├── scripts/
│   └── train_draft.sh         # SLURM training script
└── validation/                # Generated validation samples
```

## Installation

This module depends on the `stedfm` package (https://github.com/FLClab/STED-FM). Ensure it is installed:

```bash
cd ../STED-FM
pip install -e .
```

## Usage

### Training with DRaFT

```bash
python train_draft.py \
    --dataset-path /path/to/dataset \
    --pretrained-checkpoint /path/to/pretrained_ddpm.pth \
    --batch-size 8 \
    --num-epochs 100 \
    --K 5 \
    --num-sampling-steps 100 \
    --reward-weight 1.0 \
    --use-lora \
    --lora-rank 4 \
    --learning-rate 1e-5 \
    --use-gradient-checkpointing
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--K` | Number of sampling steps to backpropagate through (`None` = all steps) | 5 |
| `--num-sampling-steps` | Number of DDIM sampling steps | 100 |
| `--reward-weight` | Weight for the reward loss | 1.0 |
| `--denoising-weight` | Weight for auxiliary denoising loss | 0.0 |
| `--use-lora` | Enable LoRA for parameter-efficient training | True |
| `--lora-rank` | LoRA rank (lower = more efficient) | 4 |
| `--use-gradient-checkpointing` | Enable gradient checkpointing (recommended) | True |



