import pickle
import os
import glob
import numpy 
import argparse

from matplotlib import pyplot
from scipy.stats import wilcoxon, mannwhitneyu


from stedfm.DEFAULTS import COLORS
from stedfm.utils import savefig

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="diffusion-super-resolution")
args = parser.parse_args()

CLASSES = [
    "ddim",
    "draft",

]
NAMES = {
    "ddim" : "DDIM",
    "draft" : "DRAFT"
}

class User:
    def __init__(self, name):
        self.name = name

def get_color(model):
    for key, value in COLORS.items():
        if key.lower() in model.lower():
            return value
    return COLORS[model]

def get_class(filename):
    basename = os.path.basename(filename)
    for c in CLASSES:
        if c in basename:
            return c
    return None

def get_user_choices():
    files = glob.glob(f"data/{args.dataset}/*.pkl")
    files = [f for f in files if "fred" not in f]
    files = [f for f in files if "anthony" not in f]
    per_user_scores = {}
    for file in files:
        scores = {c : 0 for c in CLASSES}
        with open(file, "rb") as f:
            data = pickle.load(f)
            user_choices = data["user_choices"]
        if len(user_choices) == 0:
            continue
        for key, value in user_choices.items():
            scores[get_class(value)] += 1
        per_user_scores[file] = scores
    return per_user_scores

def merge_dicts(dicts):
    merged = {}
    for i, d in enumerate(dicts):
        # print(i)
        for key, value in d.items():
            # print(os.path.basename(key) in value)
            if key not in merged:
                merged[key] = [value]
            else:
                merged[key].append(value)
    return merged

def get_selections():
    files = glob.glob(f"data/{args.dataset}/*.pkl")
    per_user_data = []
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)

            # Makes sure that the user choices are consistent
            to_remove = []
            for key, value in data["user_choices"].items():
                if not (os.path.basename(key) in value):
                    to_remove.append(key)
            if to_remove:
                print(f"Removing {len(to_remove)} inconsistent selections for user {data['user'].name}")
            for key in to_remove:
                del data["user_choices"][key]

            per_user_data.append(data["user_choices"])
    
    merged = merge_dicts(per_user_data)

    largest_set = max([len(set(values)) for values in merged.values()])
    for key, values in merged.items():
        print(len(set(values)), set(values))

        # Every user selected the same image
        if len(set(values)) == 1:
            print(os.path.basename(key), NAMES[get_class(values[0])])
        elif len(set(values)) == largest_set:
            print(os.path.basename(key), [NAMES[get_class(v)] for v in values])

    # Rank by disagreement
    disagreements = {key : len(set(values)) for key, values in merged.items()}
    values = []
    for key, value in sorted(disagreements.items(), key=lambda x: x[1]):
        values.append(value)

    fig, ax = pyplot.subplots(figsize=(3, 3))
    ax.plot(values)
    ax.set(
        ylabel="Disagreement (-)"
    )
    savefig(fig, f"./results/{args.dataset}/disagreement", save_white=True)
        
def main():

    numpy.random.seed(42)

    get_selections()

    per_user_scores = get_user_choices()
    print("\n\n")
    all_values = []
    per_class_scores = {c : [] for c in CLASSES}
    for user, scores in per_user_scores.items():
        print(f"{os.path.basename(user)}: {scores}")
        for c in CLASSES:
            per_class_scores[c].append(scores[c])
        values = numpy.array([scores[c] for c in CLASSES])
        all_values.append(values)
    all_values = numpy.array(all_values)

    all_values = all_values / all_values.sum(axis=1, keepdims=True)

    c1 = numpy.array(per_class_scores[CLASSES[0]])
    c2 = numpy.array(per_class_scores[CLASSES[1]]) 

    w, p = mannwhitneyu(c1, c2)
    print(f"Wilcoxon test: w={w}, p={p}")

    fig, ax = pyplot.subplots(figsize=(3, 3))
    for i in range(all_values.shape[1]):
        mean = numpy.mean(all_values[:, i])
        std = numpy.std(all_values[:, i])
        ax.scatter(numpy.random.normal(i, 0., size=all_values.shape[0]), all_values[:, i], facecolor="none", edgecolor="black", zorder=100)
        ax.bar(i, mean, yerr=std, width=0.8, label=CLASSES[i], align="center", color=COLORS[CLASSES[i]])
    
    ax.set_xticks(numpy.arange(len(scores.keys())))
    ax.set_xticklabels([NAMES[c] for c in CLASSES], rotation=45)
    ax.set(
        ylabel="Proportion (-)", ylim=(0, 1)
    )
    savefig(fig, f"./results/{args.dataset}/choices", save_white=True)

if __name__ == "__main__":
    main()