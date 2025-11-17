"""
LoRA (Low-Rank Adaptation) implementation for diffusion models.

Based on "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
and used in "Directly Fine-Tuning Diffusion Models on Differentiable Rewards" (Clark et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


class LoRALayer(nn.Module):
    """
    LoRA layer that can be injected into any Linear or Conv2d layer.
    
    Instead of fine-tuning W, we keep W frozen and learn:
    ΔW = B @ A
    where A is (in_features, r) and B is (r, out_features) with r << min(in_features, out_features)
    
    Forward pass: y = W·x + (B @ A)·x
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with kaiming_uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA output: (B @ A) @ x
        """
        # x: (..., in_features)
        x = self.dropout(x)
        result = x @ self.lora_A @ self.lora_B
        result = result * self.scaling
        return result


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.
    """
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Freeze original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


class Conv2dWithLoRA(nn.Module):
    """
    Conv2d layer with LoRA adaptation.
    
    For convolutional layers, we reshape the weight matrix to apply LoRA.
    For 1x1 convolutions (common in UNets), this is equivalent to a Linear layer.
    """
    def __init__(
        self,
        conv: nn.Conv2d,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = conv
        self.is_1x1 = (conv.kernel_size == (1, 1))
        
        # For LoRA, treat as (out_channels, in_channels * k * k)
        in_features = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
        out_features = conv.out_channels
        
        self.lora = LoRALayer(
            in_features,
            out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Freeze original conv layer
        for param in self.conv.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original convolution
        h = self.conv(x)
        
        # LoRA adaptation
        if self.is_1x1:
            # For 1x1 convs, we can treat it as a linear transformation
            # Input: (B, C_in, H, W) -> (B, H*W, C_in)
            batch_size, in_channels, height, width = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, in_channels)
            
            # Apply LoRA: (B*H*W, C_in) -> (B*H*W, C_out)
            lora_out = self.lora(x_flat)
            
            # Reshape back: (B*H*W, C_out) -> (B, C_out, H, W)
            lora_out = lora_out.reshape(batch_size, height, width, self.conv.out_channels)
            lora_out = lora_out.permute(0, 3, 1, 2)
        else:
            # For larger kernels, use unfold/fold
            batch_size, in_channels, height, width = x.shape
            x_unfold = F.unfold(
                x,
                kernel_size=self.conv.kernel_size,
                padding=self.conv.padding,
                stride=self.conv.stride,
            )  # (B, in_channels * K * K, L)
            
            # Apply LoRA
            x_unfold = x_unfold.transpose(1, 2)  # (B, L, in_channels * K * K)
            lora_out = self.lora(x_unfold)  # (B, L, out_channels)
            lora_out = lora_out.transpose(1, 2)  # (B, out_channels, L)
            
            # Fold back
            out_h = (height + 2 * self.conv.padding[0] - self.conv.kernel_size[0]) // self.conv.stride[0] + 1
            out_w = (width + 2 * self.conv.padding[1] - self.conv.kernel_size[1]) // self.conv.stride[1] + 1
            lora_out = lora_out.view(batch_size, self.conv.out_channels, out_h, out_w)
        
        return h + lora_out


def inject_lora_into_unet(
    unet: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, int]:
    """
    Inject LoRA layers into a UNet model.
    
    Specifically targets key layers in the UNet:
    - Attention projection layers (to_qkv, to_out)
    - ResNet block convolutions (proj, res_conv)
    - Time/conditioning MLPs
    
    Args:
        unet: The UNet model to inject LoRA into
        rank: LoRA rank (r) - typically 4-16 for diffusion models
        alpha: LoRA scaling parameter
        dropout: Dropout rate for LoRA layers
        target_modules: List of module name patterns to target (None = sensible defaults)
        
    Returns:
        unet: Modified UNet with LoRA layers
        num_lora_params: Number of trainable LoRA parameters
    """
    # Default targets for UNet architecture
    if target_modules is None:
        # Target attention and projection layers primarily
        target_modules = [
            "to_qkv",      # Attention Q/K/V projections
            "to_out",      # Attention output projections  
            "proj",        # ResNet block projections
            "time_mlp",    # Time embedding MLP
            "cond_mlp",    # Conditioning MLP
        ]
    
    num_lora_params = 0
    num_original_params = 0
    num_injected = 0
    
    def should_inject(module_name: str, module: nn.Module) -> bool:
        """Determine if this module should get LoRA."""
        # Must be Linear or Conv2d
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            return False
        
        # Check against target patterns
        for pattern in target_modules:
            if pattern in module_name:
                return True
        
        return False
    
    # Collect modules to inject
    modules_to_inject = []
    for name, module in unet.named_modules():
        if should_inject(name, module):
            modules_to_inject.append((name, module))
            for p in module.parameters():
                num_original_params += p.numel()
    
    print(f"\nFound {len(modules_to_inject)} modules to inject LoRA into")
    
    # Inject LoRA into each target module
    for module_name, module in modules_to_inject:
        # Get parent and attribute name
        *parent_names, attr_name = module_name.split('.')
        
        parent = unet
        for parent_name in parent_names:
            parent = getattr(parent, parent_name)
        
        # Create LoRA wrapper
        if isinstance(module, nn.Linear):
            lora_module = LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout)
        elif isinstance(module, nn.Conv2d):
            lora_module = Conv2dWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout)
        else:
            continue
        
        # Count LoRA parameters
        num_lora_params += sum(p.numel() for p in lora_module.lora.parameters())
        
        # Replace module
        setattr(parent, attr_name, lora_module)
        num_injected += 1
    
    print(f"\nLoRA injection complete:")
    print(f"  Modules modified: {num_injected}")
    print(f"  Original trainable parameters: {num_original_params:,}")
    print(f"  LoRA parameters: {num_lora_params:,}")
    print(f"  Parameter reduction: {100 * (1 - num_lora_params / max(num_original_params, 1)):.1f}%")
    print(f"  Memory savings: ~{(num_original_params - num_lora_params) * 4 / 1024 / 1024:.1f} MB")
    
    return unet, num_lora_params


def inject_lora_into_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, int]:
    """
    Generic LoRA injection for any model.
    
    For UNet models, prefer inject_lora_into_unet() for better targeting.
    """
    # Try to use UNet-specific injection if it looks like a UNet
    if hasattr(model, 'downs') and hasattr(model, 'ups') and hasattr(model, 'mid_block1'):
        print("Detected UNet architecture, using UNet-specific LoRA injection...")
        return inject_lora_into_unet(model, rank, alpha, dropout, target_modules)
    
    # Otherwise, fall back to generic injection
    return inject_lora_into_unet(model, rank, alpha, dropout, target_modules)


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, (LinearWithLoRA, Conv2dWithLoRA)):
            lora_params.extend(module.lora.parameters())
    return lora_params


def merge_lora_into_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into original weights for inference.
    
    This eliminates the LoRA overhead during inference by permanently
    adding ΔW to W.
    """
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            # Merge: W_new = W + B @ A * scaling
            with torch.no_grad():
                delta_w = (module.lora.lora_B @ module.lora.lora_A.T) * module.lora.scaling
                module.linear.weight.data += delta_w.T
            
            # Replace with original linear (now containing merged weights)
            parent_name, child_name = name.rsplit('.', 1)
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, module.linear)
            
        elif isinstance(module, Conv2dWithLoRA):
            # Merge conv weights
            with torch.no_grad():
                # Reshape LoRA matrices to conv shape
                in_features = module.conv.in_channels * module.conv.kernel_size[0] * module.conv.kernel_size[1]
                delta_w_flat = (module.lora.lora_B @ module.lora.lora_A.T) * module.lora.scaling  # (out, in)
                delta_w = delta_w_flat.view(
                    module.conv.out_channels,
                    module.conv.in_channels,
                    module.conv.kernel_size[0],
                    module.conv.kernel_size[1]
                )
                module.conv.weight.data += delta_w
            
            # Replace with original conv
            parent_name, child_name = name.rsplit('.', 1)
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, module.conv)
    
    return model


def save_lora_weights(model: nn.Module, path: str):
    """
    Save only LoRA weights (not the full model).
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (LinearWithLoRA, Conv2dWithLoRA)):
            lora_state_dict[name + ".lora_A"] = module.lora.lora_A
            lora_state_dict[name + ".lora_B"] = module.lora.lora_B
    
    torch.save(lora_state_dict, path)
    print(f"Saved LoRA weights to {path}")


def load_lora_weights(model: nn.Module, path: str):
    """
    Load LoRA weights into a model.
    """
    lora_state_dict = torch.load(path)
    
    for name, module in model.named_modules():
        if isinstance(module, (LinearWithLoRA, Conv2dWithLoRA)):
            if name + ".lora_A" in lora_state_dict:
                module.lora.lora_A.data = lora_state_dict[name + ".lora_A"]
                module.lora.lora_B.data = lora_state_dict[name + ".lora_B"]
    
    print(f"Loaded LoRA weights from {path}")

