"""
Simple test script to verify DRaFT implementation works correctly.
"""

import torch
import torch.nn as nn
from diffusion_model_draft import DRaFT_DDPM, RewardEncoder
from denoising_unet import UNet
from lora_layers import get_lora_parameters


def create_simple_encoder():
    """Create a simple CNN encoder for testing."""
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 128)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.flatten(start_dim=1)
            x = self.fc(x)
            return x
    
    return SimpleEncoder()


def test_reward_computation():
    """Test that reward computation works."""
    print("Testing reward computation...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple encoder
    encoder = create_simple_encoder()
    reward_encoder = RewardEncoder(encoder, pooling="adaptive")
    reward_encoder.to(device)
    
    # Create dummy images
    generated = torch.randn(2, 1, 64, 64).to(device)
    target = torch.randn(2, 1, 64, 64).to(device)
    
    # Create DRaFT model (minimal)
    denoising_model = UNet(
        dim=32,
        channels=1,
        out_dim=1,
        dim_mults=(1, 2),
        condition_type=None,
    )
    
    model = DRaFT_DDPM(
        denoising_model=denoising_model,
        reward_encoder=reward_encoder,
        timesteps=100,
        K=5,
        num_sampling_steps=10,
    )
    model.to(device)
    
    # Compute reward
    reward = model.compute_cosine_similarity_reward(generated, target)
    
    assert reward.shape == (2,), f"Expected shape (2,), got {reward.shape}"
    assert -1 <= reward.min() <= 1, "Cosine similarity should be in [-1, 1]"
    assert -1 <= reward.max() <= 1, "Cosine similarity should be in [-1, 1]"
    
    print(f"✓ Reward computation works! Rewards: {reward}")
    return True


def test_differentiable_sampling():
    """Test that sampling is differentiable."""
    print("\nTesting differentiable sampling...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple encoder
    encoder = create_simple_encoder()
    reward_encoder = RewardEncoder(encoder, pooling="adaptive")
    reward_encoder.to(device)
    
    # Create model
    denoising_model = UNet(
        dim=32,
        channels=1,
        out_dim=1,
        dim_mults=(1, 2),
        condition_type=None,
    )
    
    model = DRaFT_DDPM(
        denoising_model=denoising_model,
        reward_encoder=reward_encoder,
        timesteps=100,
        K=5,
        num_sampling_steps=10,
    )
    model.to(device)
    model.train()
    
    # Create dummy target
    target = torch.randn(1, 1, 64, 64).to(device)
    
    # Sample with reward computation
    generated, reward = model.sample_with_reward(
        shape=(1, 1, 64, 64),
        target_images=target,
    )
    
    # Check that generated has gradients
    assert generated.requires_grad, "Generated samples should require gradients!"
    
    # Compute loss and backprop
    loss = -reward.mean()
    loss.backward()
    
    # Check that model parameters have gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "Model parameters should have gradients after backprop!"
    
    print(f"✓ Differentiable sampling works! Generated shape: {generated.shape}, Reward: {reward.item():.4f}")
    return True


def test_training_step():
    """Test a full training step."""
    print("\nTesting full training step...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple encoder
    encoder = create_simple_encoder()
    reward_encoder = RewardEncoder(encoder, pooling="adaptive")
    reward_encoder.to(device)
    
    # Create model
    denoising_model = UNet(
        dim=32,
        channels=1,
        out_dim=1,
        dim_mults=(1, 2),
        condition_type=None,
    )
    
    model = DRaFT_DDPM(
        denoising_model=denoising_model,
        reward_encoder=reward_encoder,
        timesteps=100,
        K=5,
        num_sampling_steps=10,
        reward_weight=1.0,
        denoising_weight=0.1,
    )
    model.to(device)
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy batch
    target_images = torch.randn(2, 1, 224, 224).to(device)
    
    # Training step
    optimizer.zero_grad()
    
    results = model.draft_training_step(
        target_images=target_images,
    )
    
    loss = results["total_loss"]
    reward = results["reward"]
    
    # Check results
    assert "total_loss" in results, "Results should contain total_loss"
    assert "reward_loss" in results, "Results should contain reward_loss"
    assert "reward" in results, "Results should contain reward"
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step works!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Reward: {reward.item():.4f}")
    print(f"  Reward Loss: {results['reward_loss'].item():.4f}")
    print(f"  Denoising Loss: {results['denoising_loss'].item():.4f}")
    
    return True


def test_draft_k_variants():
    """Test different DRaFT variants (full, K, LV)."""
    print("\nTesting DRaFT variants...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple encoder
    encoder = create_simple_encoder()
    reward_encoder = RewardEncoder(encoder, pooling="adaptive")
    reward_encoder.to(device)
    
    configs = [
        ("Full DRaFT", {"K": None}),
        ("DRaFT-K (K=5)", {"K": 5}),
        ("DRaFT-LV (K=1)", {"K": 1, "use_low_variance": True}),
    ]
    
    for name, config in configs:
        print(f"\n  Testing {name}...")
        
        denoising_model = UNet(
            dim=32,
            channels=1,
            out_dim=1,
            dim_mults=(1, 2),
            condition_type=None,
        )
        
        model = DRaFT_DDPM(
            denoising_model=denoising_model,
            reward_encoder=reward_encoder,
            timesteps=100,
            num_sampling_steps=10,
            **config
        )
        model.to(device)
        
        # Test sampling
        target = torch.randn(1, 1, 64, 64).to(device)
        generated, reward = model.sample_with_reward(
            shape=(1, 1, 64, 64),
            target_images=target,
        )
        
        print(f"    ✓ {name} works! Reward: {reward.item():.4f}")
    
    return True


def test_lora_integration():
    """Test that LoRA integration works correctly."""
    print("\nTesting LoRA integration...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple encoder
    encoder = create_simple_encoder()
    reward_encoder = RewardEncoder(encoder, pooling="adaptive")
    reward_encoder.to(device)
    
    # Create model WITH LoRA
    denoising_model = UNet(
        dim=32,
        channels=1,
        out_dim=1,
        dim_mults=(1, 2),
        condition_type=None,
    )
    
    model_with_lora = DRaFT_DDPM(
        denoising_model=denoising_model,
        reward_encoder=reward_encoder,
        timesteps=100,
        K=5,
        num_sampling_steps=10,
        use_lora=True,
        lora_rank=4,
    )
    model_with_lora.to(device)
    
    # Check that only LoRA parameters are trainable
    lora_params = get_lora_parameters(model_with_lora.model)
    assert len(lora_params) > 0, "Should have LoRA parameters"
    
    # Count trainable vs total parameters
    trainable = sum(p.numel() for p in model_with_lora.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_with_lora.model.parameters())
    
    reduction = 100 * (1 - trainable / total)
    assert reduction > 50, f"LoRA should reduce trainable params by >50%, got {reduction:.1f}%"
    
    print(f"✓ LoRA integration works!")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable (LoRA): {trainable:,}")
    print(f"  Reduction: {reduction:.1f}%")
    
    # Test that training works with LoRA
    target = torch.randn(1, 1, 64, 64).to(device)
    results = model_with_lora.draft_training_step(target_images=target)
    
    loss = results["total_loss"]
    loss.backward()
    
    # Check that LoRA parameters have gradients
    has_lora_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_params)
    assert has_lora_grad, "LoRA parameters should have gradients!"
    
    print(f"✓ LoRA training works!")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running DRaFT Implementation Tests")
    print("="*60)
    
    tests = [
        ("Reward Computation", test_reward_computation),
        ("Differentiable Sampling", test_differentiable_sampling),
        ("Training Step", test_training_step),
        ("DRaFT Variants", test_draft_k_variants),
        ("LoRA Integration", test_lora_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} failed with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 All tests passed! DRaFT implementation is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

