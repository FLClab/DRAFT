import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

def main():
    base_path = "/home-local/Frederic/baselines/SR-baselines/DRAFT"

    checkpoints = [
        os.path.join(base_path, f"DRaFT_{i}.pth") for i in range(1, 11)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rewards = []
    for c in checkpoints:
        print(c)
        data = torch.load(c, map_location="cpu", weights_only=False)
        reward = data["reward"]
        rewards.append(reward)
    ax.plot(rewards)
    plt.savefig("reward.pdf")
    plt.close()

if __name__ == "__main__":
    main()