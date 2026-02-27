import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Callable, List
from diffusion_model import DDPM, _extract_into_tensor
from torch import nn
from lora_layers import inject_lora_into_model, get_lora_parameters, merge_lora_into_weights, save_lora_weights, load_lora_weights
import copy


class DRaFT_DDPM(DDPM):
    """
    Direct Reward Fine-Tuning (DRaFT) extension of DDPM.
    
    This model implements the DRaFT procedure from "Directly Fine-Tuning Diffusion Models 
    on Differentiable Rewards" (Clark et al., ICLR 2024).
    
    Instead of using standard denoising loss, this model optimizes a differentiable reward
    function by backpropagating through the sampling process.
    """
    
    def __init__(
        self,
        denoising_model: nn.Module,
        reward_encoder: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        K: Optional[int] = None,  # Number of steps to backprop through (None = all steps)
        use_low_variance: bool = False,  # Use DRaFT-LV variant
        reward_weight: float = 1.0,  # Weight for reward loss
        denoising_weight: float = 0.1,  # Weight for standard denoising loss (for stability)
        num_sampling_steps: int = 100,  # Number of DDIM steps for sampling
        eta: float = 0.0,  # DDIM eta parameter (0 = deterministic)
        use_lora: bool = True,  # Use LoRA for parameter-efficient fine-tuning
        lora_rank: int = 4,  # LoRA rank
        lora_alpha: float = 1.0,  # LoRA scaling
        lora_dropout: float = 0.0,  # LoRA dropout
        lora_target_modules: Optional[List[str]] = None,  # Target modules for LoRA
        use_gradient_checkpointing: bool = True,  # Use gradient checkpointing
        **kwargs
    ):
        """
        Args:
            denoising_model: The UNet denoising model
            reward_encoder: Pre-trained encoder for extracting embeddings (frozen)
            timesteps: Number of diffusion timesteps
            beta_schedule: Beta schedule type
            K: Number of final sampling steps to backprop through (None = all steps, DRaFT)
            use_low_variance: If True and K=1, use DRaFT-LV for lower variance
            reward_weight: Weight for the reward loss
            denoising_weight: Weight for standard denoising loss (helps stability)
            num_sampling_steps: Number of steps for DDIM sampling
            eta: DDIM stochasticity parameter
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_rank: LoRA rank (r) - lower is more efficient
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout rate for LoRA layers
            lora_target_modules: Which modules to apply LoRA to (None = default)
            use_gradient_checkpointing: Use gradient checkpointing to reduce memory
        """
        super().__init__(
            denoising_model=denoising_model,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            **kwargs
        )
        
        self.reward_encoder = reward_encoder
        # Freeze reward encoder
        for p in self.reward_encoder.parameters():
            p.requires_grad = False
        self.reward_encoder.eval()
        
        self.K = K  # None means backprop through all steps (full DRaFT)
        self.use_low_variance = use_low_variance
        self.reward_weight = reward_weight
        self.denoising_weight = denoising_weight
        self.num_sampling_steps = num_sampling_steps
        self.eta = eta
        self.use_lora = use_lora
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # EMA model for stable evaluation (optional)
        self.use_ema = False  # Can be enabled via method
        self.ema_model = None
        self.ema_decay = 0.9999

        # Inject LoRA into the denoising model
        if use_lora:
            print(f"\n{'='*60}")
            print("Injecting LoRA into denoising model...")
            print(f"  Rank: {lora_rank}")
            print(f"  Alpha: {lora_alpha}")
            print(f"  Dropout: {lora_dropout}")
            
            # Freeze all parameters first
            for p in self.model.parameters():
                p.requires_grad = False
            
            # Inject LoRA (this will unfreeze LoRA parameters)
            self.model, self.num_lora_params = inject_lora_into_model(
                self.model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            
            print(f"{'='*60}\n")
        else:
            self.num_lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def compute_perceptual_loss_reward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss between generated and target images.

        The reward is negative MSE, so higher (less negative) is better.
        """
        self.reward_encoder.eval()

        # Normalize images to expected range for the encoder
        # Most pretrained models expect images in [0, 1] or [-1, 1]
        # Clamp to ensure valid range after diffusion process
        gen_normalized = torch.clamp(generated, 0, 1)
        target_normalized = torch.clamp(target, 0, 1)

        # Extract features
        gen_features = self.reward_encoder(gen_normalized)
        with torch.no_grad():
            target_features = self.reward_encoder(target_normalized)

        # Flatten if needed
        if len(gen_features.shape) > 2:
            gen_features = gen_features.flatten(start_dim=1)
        if len(target_features.shape) > 2:
            target_features = target_features.flatten(start_dim=1)

        # Compute negative MSE as reward (higher is better)
        reward = -F.mse_loss(gen_features, target_features, reduction='none').mean(dim=1)
        return reward
        
    def compute_cosine_similarity_reward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings of generated and target images.

        Args:
            generated: Generated images [B, C, H, W]
            target: Target (real) images [B, C, H, W]

        Returns:
            Cosine similarity reward [B] in range [-1, 1]
        """
        # Keep encoder in eval mode
        # Encoder parameters are frozen (requires_grad=False), but we need
        # gradients w.r.t. the input to backprop through sampling
        self.reward_encoder.eval()

        # Normalize images to expected range for the encoder
        gen_normalized = torch.clamp(generated, 0, 1)
        target_normalized = torch.clamp(target, 0, 1)

        # Extract embeddings
        gen_embedding = self.reward_encoder(gen_normalized)

        # Target embedding can be computed without gradients (it's fixed)
        with torch.no_grad():
            target_embedding = self.reward_encoder(target_normalized)

        # If embeddings are multi-dimensional, flatten them
        if len(gen_embedding.shape) > 2:
            gen_embedding = gen_embedding.flatten(start_dim=1)
        if len(target_embedding.shape) > 2:
            target_embedding = target_embedding.flatten(start_dim=1)

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(gen_embedding, target_embedding, dim=-1)

        return cos_sim  # Shape: [B], range [-1, 1]
    
    def ddim_sample_differentiable(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
        cond: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        clip_denoised: bool = True,
        use_low_variance: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable DDIM sampling step (no torch.no_grad()).

        This is the key difference from standard sampling - we need gradients
        to flow through the sampling process for DRaFT.

        Args:
            use_low_variance: If True, use DRaFT-LV gradient estimator (only for K=1)
                             This detaches x_t to reduce gradient variance
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Handle two-channel input (image + segmentation)
        x_for_eps = x
        if x.shape[1] == 2:
            x_for_eps = x[:, [0], :, :]

        # DRaFT-LV: detach x_t to compute lower-variance gradients
        # Only gradients flow through epsilon_theta(x_t, t), not through x_t
        if use_low_variance:
            x_for_model = x.detach()
        else:
            x_for_model = x

        # Get model prediction (with gradients)
        out = self.p_mean_variance(
            x_for_model, t, clip_denoised=clip_denoised,
            denoised_fn=None, model_kwargs=model_kwargs, cond=cond
        )

        pred_x0 = out["pred_x0"]

        # Get alphas
        alphas_cumprod = torch.from_numpy(self.alphas_cumprod).to(x.device, x.dtype)
        a_t = alphas_cumprod[t].view(-1, *([1] * (x_for_eps.dim() - 1)))
        a_prev = alphas_cumprod[t_prev].view(-1, *([1] * (x_for_eps.dim() - 1)))

        # Compute epsilon (predicted noise)
        # Add epsilon for numerical stability
        eps = (x_for_eps - a_t.sqrt() * pred_x0) / ((1 - a_t).sqrt() + 1e-8)

        # DDIM update
        # Add epsilon to prevent sqrt of negative numbers due to numerical errors
        sigma_squared = eta**2 * (1 - a_prev) / (1 - a_t + 1e-8) * (1 - a_t / (a_prev + 1e-8))
        sigma_squared = torch.clamp(sigma_squared, min=0.0)
        sigma = sigma_squared.sqrt()

        # For DRaFT, we typically use deterministic sampling (eta=0)
        # But allow stochasticity if needed
        if eta > 0:
            noise = torch.randn_like(x_for_eps)
        else:
            noise = 0.0

        # Compute coefficient for epsilon with numerical stability
        eps_coef_squared = 1 - a_prev - sigma_squared
        eps_coef_squared = torch.clamp(eps_coef_squared, min=0.0)
        eps_coef = eps_coef_squared.sqrt()

        mean_pred = pred_x0 * a_prev.sqrt() + eps * eps_coef
        x_prev = mean_pred + sigma * noise

        return x_prev, pred_x0
    
    def _ddim_step_with_checkpointing(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
        cond: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        clip_denoised: bool = True,
        use_low_variance: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM sampling step with gradient checkpointing.

        This wraps the standard DDIM step with PyTorch's gradient checkpointing,
        which only stores the input and recomputes activations during backprop.
        This is the key memory optimization from the DRaFT paper.
        """
        # Create a wrapper function that can be checkpointed
        def checkpoint_fn(x_in, t_in, t_prev_in):
            """
            Wrapper function for checkpointing.
            Must be a pure function of tensors (no kwargs).
            """
            # This function will be called twice:
            # 1. Forward pass: compute output
            # 2. Backward pass: recompute to get gradients
            return self.ddim_sample_differentiable(
                x_in, t_in, t_prev_in,
                eta=eta,
                cond=cond,
                model_kwargs=model_kwargs,
                clip_denoised=clip_denoised,
                use_low_variance=use_low_variance
            )

        # Use PyTorch's gradient checkpointing
        # This only stores (x, t, t_prev) and recomputes everything else during backprop
        from torch.utils.checkpoint import checkpoint

        # Call checkpoint with use_reentrant=False (recommended for modern PyTorch)
        x_prev, pred_x0 = checkpoint(
            checkpoint_fn,
            x, t, t_prev,
            use_reentrant=False
        )

        return x_prev, pred_x0
    
    def sample_with_reward(
        self,
        shape: Tuple[int, int, int, int],
        target_images: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        segmentation: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform sampling with gradient computation for reward optimization.
        
        Args:
            shape: Shape of samples to generate
            target_images: Real images for computing reward
            cond: Conditioning information
            segmentation: Segmentation to concatenate
            model_kwargs: Additional model arguments
            noise: Initial noise (if None, sample from N(0,I))
            clip_denoised: Whether to clip denoised predictions
            
        Returns:
            generated_images: Final generated samples
            reward: Cosine similarity reward
        """
        device = next(self.model.parameters()).device
        
        if model_kwargs is None:
            model_kwargs = {}
        
        # Initialize noise
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        
        # Create time schedule
        times = np.linspace(self.T - 1, 0, self.num_sampling_steps, dtype=int)
        times_next = np.append(times[1:], 0)
        
        # Determine where to start computing gradients
        if self.K is None:
            # DRaFT: backprop through all steps
            start_grad_idx = 0
        else:
            # DRaFT-K: backprop through last K steps
            start_grad_idx = max(0, len(times) - self.K)
        
        # Sampling loop
        for idx, (t_i, t_prev_i) in enumerate(zip(times, times_next)):
            t = torch.full((shape[0],), t_i, device=device, dtype=torch.long)
            t_prev = torch.full((shape[0],), t_prev_i, device=device, dtype=torch.long)

            # Concatenate segmentation if needed
            if segmentation is not None:
                img_in = torch.cat((img, segmentation), dim=1)
            else:
                img_in = img

            # Determine if this is the last step (for DRaFT-LV)
            is_last_step = (idx == len(times) - 1)
            use_lv = self.use_low_variance and self.K == 1 and is_last_step

            # Use gradient computation for last K steps (or all if K is None)
            if idx >= start_grad_idx:
                # Differentiable sampling with optional gradient checkpointing
                if self.use_gradient_checkpointing:
                    # Use gradient checkpointing: only store input, recompute activations during backprop
                    # This dramatically reduces memory at the cost of ~30% more compute
                    img, _ = self._ddim_step_with_checkpointing(
                        img_in, t, t_prev, eta=self.eta,
                        cond=cond, model_kwargs=model_kwargs,
                        clip_denoised=clip_denoised,
                        use_low_variance=use_lv
                    )
                else:
                    # No checkpointing: store all activations (uses more memory)
                    img, _ = self.ddim_sample_differentiable(
                        img_in, t, t_prev, eta=self.eta,
                        cond=cond, model_kwargs=model_kwargs,
                        clip_denoised=clip_denoised,
                        use_low_variance=use_lv
                    )
            else:
                # No gradient for early steps (memory efficient)
                with torch.no_grad():
                    img, _ = self.ddim_sample_differentiable(
                        img_in, t, t_prev, eta=self.eta,
                        cond=cond, model_kwargs=model_kwargs,
                        clip_denoised=clip_denoised,
                        use_low_variance=False
                    )
        
        # Compute reward
        # reward = self.compute_cosine_similarity_reward(img, target_images)
        reward = self.compute_perceptual_loss_reward(img, target_images)
        return img, reward
    
    def draft_training_step(
        self,
        target_images: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        segmentation: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        DRaFT training step: sample and optimize for reward.
        
        Args:
            target_images: Real images to match
            cond: Conditioning
            segmentation: Segmentation mask
            model_kwargs: Additional model arguments
            noise: Initial noise
            
        Returns:
            Dictionary with losses and metrics
        """
        batch_size = target_images.shape[0]
        shape = target_images.shape
        
        # Generate samples with gradient tracking
        generated, reward = self.sample_with_reward(
            shape=shape,
            target_images=target_images,
            cond=cond,
            segmentation=segmentation,
            model_kwargs=model_kwargs,
            noise=noise,
        )
        
        # Reward loss: we want to maximize reward, so minimize negative reward
        reward_loss = -reward.mean()
        
        # Optional: Add standard denoising loss for stability
        if self.denoising_weight > 0:
            device = target_images.device
            t = torch.randint(0, self.T, (batch_size,), device=device).long()
            noise_std = torch.randn_like(target_images)
            
            # Standard denoising objective
            denoising_losses, _ = self.forward(
                x_0=target_images,
                t=t,
                cond=cond,
                segmentation=segmentation,
                model_kwargs=model_kwargs,
                noise=noise_std,
            )
            denoising_loss = denoising_losses["loss"].mean()
        else:
            denoising_loss = torch.tensor(0.0, device=target_images.device)
        
        # Combined loss
        total_loss = (
            self.reward_weight * reward_loss + 
            self.denoising_weight * denoising_loss
        )
        
        return {
            "total_loss": total_loss,
            "reward_loss": reward_loss,
            "denoising_loss": denoising_loss,
            "reward": reward.mean(),
            "generated": generated,
        }
    
    def training_step(self, batch, batch_idx):
        """
        PyTorch Lightning training step for DRaFT.
        """
        imgs, segmentations, ground_truth, _ = batch
        device = imgs.device
        
        # Move to device and ensure contiguous
        imgs = imgs.contiguous()
        if segmentations is not None:
            segmentations = segmentations.contiguous()
        
        # Prepare conditioning
        if self.condition_type == "latent":
            latent_code, context = self.latent_encoder.forward_features(
                imgs, return_patches=True
            )
            cond = latent_code
        else:
            cond = None
        
        # Condition that gets concatenated to input noise (spatial)
        if not self.concat_segmentation:
            segmentations = None
        
        # DRaFT training step
        results = self.draft_training_step(
            target_images=imgs,
            cond=cond,
            segmentation=segmentations,
            model_kwargs={},
        )
        
        loss = results["total_loss"]
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("reward", results["reward"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("reward_loss", results["reward_loss"], on_step=True, on_epoch=True, sync_dist=True)
        
        if self.denoising_weight > 0:
            self.log("denoising_loss", results["denoising_loss"], on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer for DRaFT training.
        
        Note: DRaFT may benefit from lower learning rates than standard training.
        When using LoRA, only LoRA parameters are optimized.
        """
        if self.use_lora:
            # Only optimize LoRA parameters
            lora_params = get_lora_parameters(self.model)
            optimizer = torch.optim.Adam(lora_params, lr=1e-5, betas=(0.9, 0.99))
            print(f"Optimizing {len(lora_params)} LoRA parameter groups")
        else:
            # Optimize all trainable parameters
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, betas=(0.9, 0.99))
        return [optimizer]
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights (much smaller than full model)."""
        if self.use_lora:
            save_lora_weights(self.model, path)
        else:
            raise ValueError("Model was not trained with LoRA")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        if self.use_lora:
            load_lora_weights(self.model, path)
        else:
            raise ValueError("Model does not have LoRA layers")
    
    def merge_lora_weights(self):
        """Merge LoRA weights into base model for inference."""
        if self.use_lora:
            self.model = merge_lora_into_weights(self.model)
            self.use_lora = False
            print("LoRA weights merged into base model")

    def enable_ema(self, decay: float = 0.9999):
        """
        Enable Exponential Moving Average of model weights.

        This is recommended in the DRaFT paper for more stable training
        and better final performance.

        Args:
            decay: EMA decay rate (default: 0.9999)
        """
        self.use_ema = True
        self.ema_decay = decay
        self.ema_model = copy.deepcopy(self.model)
        # Freeze EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
        print(f"EMA enabled with decay={decay}")

    def update_ema(self):
        """Update EMA model weights."""
        if not self.use_ema or self.ema_model is None:
            return

        with torch.no_grad():
            # Update EMA parameters
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    def swap_to_ema(self):
        """Swap to EMA model for evaluation."""
        if not self.use_ema:
            return
        self.model, self.ema_model = self.ema_model, self.model

    def swap_back_from_ema(self):
        """Swap back to training model."""
        if not self.use_ema:
            return
        self.model, self.ema_model = self.ema_model, self.model


class RewardEncoder(nn.Module):
    """
    Example reward encoder wrapper.
    
    This wraps any pre-trained model to extract embeddings for reward computation.
    """
    
    def __init__(self, backbone: nn.Module, pooling: str = "adaptive"):
        """
        Args:
            backbone: Pre-trained model (e.g., ResNet, ViT, etc.)
            pooling: How to pool features ('adaptive', 'mean', 'flatten')
        """
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from images.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Embeddings [B, D]
        """
        # Keep in eval mode to disable dropout, batchnorm updates, etc.
        # No torch.no_grad() because we need gradients to flow
        self.backbone.eval()
        
        # Get features from backbone 
        # Parameters are frozen (requires_grad=False) but gradients flow through forward pass
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        # # Pool features if needed
        # if len(features.shape) > 2:
        #     if self.pooling == "adaptive":
        #         features = F.adaptive_avg_pool2d(features, (1, 1))
        #         features = features.flatten(start_dim=1)
        #     elif self.pooling == "mean":
        #         features = features.mean(dim=[-2, -1])
        #     elif self.pooling == "flatten":
        #         features = features.flatten(start_dim=1)
        
        return features

