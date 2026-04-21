import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def visualize():

    messages = torch.load("messages.pt")
    labels = torch.load("labels.pt")
    rewards = torch.load("rewards.pt")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(messages.numpy())

    labels = np.array(labels)
    rewards = np.array(rewards)

    # -------- Plot 1: Color by Agent --------
    plt.figure()

    for agent, color in zip(["Pre", "Intra", "Post"], ["red", "blue", "green"]):
        idx = labels == agent
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=agent, alpha=0.5)

    plt.title("Communication by Agent Type")
    plt.legend()
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

    # -------- Plot 2: Color by Reward --------
    plt.figure()

    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=rewards,
        cmap="viridis",
        alpha=0.5
    )

    plt.colorbar(scatter, label="Reward")
    plt.title("Communication Colored by Reward")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


if __name__ == "__main__":
    visualize()