import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize():

    messages = torch.load("messages.pt")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(messages.numpy())

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    plt.title("Agent Communication Space (PCA)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


if __name__ == "__main__":
    visualize()