import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import os
from itertools import combinations
from scipy.stats import mannwhitneyu

USERS = ["Flavie", "Cabou", "jchabbert", "Andréanne", "Kamylle", "LD", "Adeline", "Antoine"]
SUBSAMPLES = [50, 100, 300, 500, 1000, 2000, 3000, "full"]

class User:
    def __init__(self, name):
        self.name = name


def load_data(user: str, subsample: int):
    pix2pix_choices = 0
    ddpm_choices = 0
    draft_choices = 0
    with open(os.path.join("data", "diffusion-super-resolution", f"{subsample}-sample", f"{user}.pkl"), "rb") as f:
        data = pickle.load(f)
    user_choices = list(data["user_choices"].values()) 

    for u in user_choices:
        if "pix2pix" in u:
            pix2pix_choices += 1
        elif "ddpm" in u:
            ddpm_choices += 1
        elif "draft" in u:
            draft_choices += 1
        else:
            raise ValueError(f"Unknown choice: {u}") 


    print(f"User {user} made {len(user_choices)}/26 choices")

    pix2pix_fraction = pix2pix_choices / len(user_choices)
    ddpm_fraction = ddpm_choices / len(user_choices)
    draft_fraction = draft_choices / len(user_choices)
    return pix2pix_fraction, ddpm_fraction, draft_fraction

def plot_results(pix2pix_fractions, ddpm_fractions, draft_fractions):
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    pix2pix_avg = np.nanmean(pix2pix_fractions, axis=0)
    ddpm_avg = np.nanmean(ddpm_fractions, axis=0)
    draft_avg = np.nanmean(draft_fractions, axis=0)
    n = pix2pix_fractions.size 
    pix2pix_std = np.nanstd(pix2pix_fractions, axis=0, ddof=1) / np.sqrt(n)
    n = ddpm_fractions.size 
    ddpm_std = np.nanstd(ddpm_fractions, axis=0, ddof=1) / np.sqrt(n)
    n = draft_fractions.size 
    draft_std = np.nanstd(draft_fractions, axis=0, ddof=1) / np.sqrt(n)
    # draft_std = np.nanstd(draft_fractions, axis=0)
    ax.plot(np.arange(pix2pix_avg.shape[0]), pix2pix_avg, marker="o", label="Pix2Pix", color="tab:green")
    ax.fill_between(np.arange(pix2pix_avg.shape[0]), pix2pix_avg - pix2pix_std, pix2pix_avg + pix2pix_std, color="tab:green", alpha=0.2)
    ax.plot(np.arange(ddpm_avg.shape[0]), ddpm_avg, marker="o", label="DDPM", color="tab:blue")
    ax.fill_between(np.arange(ddpm_avg.shape[0]), ddpm_avg - ddpm_std, ddpm_avg + ddpm_std, color="tab:blue", alpha=0.2)
    ax.plot(np.arange(draft_avg.shape[0]), draft_avg, marker="o", label="Draft", color="#CC503E")
    ax.fill_between(np.arange(draft_avg.shape[0]), draft_avg - draft_std, draft_avg + draft_std, color="#CC503E", alpha=0.2)
    ax.set_xticks(np.arange(pix2pix_avg.shape[0]))
    ax.set_xticklabels(SUBSAMPLES)
    ax.set_xlabel("Subsample size")
    ax.set_ylabel("Fraction of choices")
    # ax.legend()
    # plt.tight_layout()
    fig.set_size_inches(3,3)
    fig.savefig("./data/user_study_results.pdf", transparent=True)
    plt.close(fig)

def plot_results_per_user(pix2pix_results, ddpm_results, draft_results):
    fig, axs = plt.subplots(1, 3, figsize=(9,3))
    cmap = plt.get_cmap("tab10")
    for i, user in enumerate(USERS): 
        pix2pix_curve = pix2pix_results[i]
        ddpm_curve = ddpm_results[i]
        draft_curve = draft_results[i]
        axs[0].plot(np.arange(pix2pix_curve.shape[0]), pix2pix_curve, marker="o", label=user, color=cmap(i), alpha=0.6)
        axs[1].plot(np.arange(ddpm_curve.shape[0]), ddpm_curve, marker="o", label=user, color=cmap(i), alpha=0.6)
        axs[2].plot(np.arange(draft_curve.shape[0]), draft_curve, marker="o", label=user, color=cmap(i), alpha=0.6)
        for ax in axs:
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(np.arange(pix2pix_curve.shape[0]))
            ax.set_xticklabels(SUBSAMPLES, rotation=45)
            ax.set_xlabel("Subsample size")
            ax.set_ylabel("Fraction of choices")
        handles, labels = axs[0].get_legend_handles_labels()
        for legend in fig.legends:
            legend.remove()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout(rect=[0, 0, 0.86, 1])
        fig.savefig(f"./data/user_study_results_per_user_{i}.pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)

def compute_stats(pix2pix_data: np.ndarray, ddpm_data: np.ndarray, draft_data: np.ndarray, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(SUBSAMPLES)):
        data = {
            "pix2pix": pix2pix_data[i],
            "ddpm": ddpm_data[i],
            "draft": draft_data[i]
        }
        pairs = combinations(data.keys(), 2)
        out_path = os.path.join(save_dir, f"user_study_stats_{SUBSAMPLES[i]}_sample.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for (p1, p2) in pairs:
                if i == 7:
                    print(f"p1: {p1}, p2: {p2}")
                    print(data[p1])
                    print(data[p2])
                    print("\n")
                data_p1 = [float(item) for item in data[p1] if not np.isnan(item)]
                data_p2 = [float(item) for item in data[p2] if not np.isnan(item)]
                w, p = mannwhitneyu(data_p1, data_p2)
                f.write(f"{p1} vs {p2}: w={w:.4f}, p={p:.6f}\n")

    

def main():
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pix2pix_results = np.zeros((len(USERS), len(SUBSAMPLES)))
    ddpm_results = np.zeros((len(USERS), len(SUBSAMPLES)))
    draft_results = np.zeros((len(USERS), len(SUBSAMPLES)))
    for i, user in enumerate(USERS):
        user_pix2pix = [] 
        user_ddpm = [] 
        user_draft = [] 
        for subsample in SUBSAMPLES:
            print(f"[---] Subsample: {subsample} [---]")
            try:
                pix2pix_fraction, ddpm_fraction, draft_fraction = load_data(user, subsample)
            except:
                print(f"[---] User {user} has not yet annotated subsample {subsample} [---]")
                user_pix2pix.append(np.nan)
                user_ddpm.append(np.nan)
                user_draft.append(np.nan)
                continue
            user_pix2pix.append(pix2pix_fraction)
            user_ddpm.append(ddpm_fraction)
            user_draft.append(draft_fraction)
            print(f"\tPix2Pix fraction: {pix2pix_fraction:.3f}")
            print(f"\tDDPM fraction: {ddpm_fraction:.3f}")
            print(f"\tDraft fraction: {draft_fraction:.3f}")
            print(f"\tTotal choices: {pix2pix_fraction + ddpm_fraction + draft_fraction:.3f}\n")

        pix2pix_results[i] = user_pix2pix
        ddpm_results[i] = user_ddpm
        draft_results[i] = user_draft

    compute_stats(pix2pix_results, ddpm_results, draft_results, save_dir="./data/user_study_stats")
    plot_results(pix2pix_results, ddpm_results, draft_results)
    plot_results_per_user(pix2pix_results, ddpm_results, draft_results)

if __name__=="__main__":
    main()