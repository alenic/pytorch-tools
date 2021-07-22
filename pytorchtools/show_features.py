'''
Simple features exploration tool
'''
# edited by Alessandro Nicolosi - https://github.com/alenic

import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import distance_matrix


def show_features(
    x, y, method="tsne", perplexity=15, path_images=None, show_closest=True, original_emb=None, metric="cosine", random_state=42
):
    matplotlib.use('TkAgg')
    # 2d reduce method
    if method == "tsne":
        dim_red = TSNE(2, perplexity=perplexity, learning_rate=200, random_state=random_state)
        x_2d = dim_red.fit_transform(x)
    elif method == "pca":
        dim_red = PCA(2)
        x_2d = dim_red.fit_transform(x)
    elif method is None:
        x_2d = x
    else:
        raise NotImplementedError("2d reduction method is not valid")

    def on_pick(event):
        if path_images is None:
            print("no images is associated to ", event.ind)
            return

        current_index = event.ind[0]
        # change point color
        if show_closest:
            proj_points._facecolors[current_index, :] = (1, 1, 0, 1)
            global_fig.canvas.draw()

        plt.figure()
        path = path_images[current_index]
        plt.imshow(Image.open(path))
        plt.title(path)
        plt.show()

        if show_closest:
            # proj_points._facecolors = copy.deepcopy(original_colors)
            sort_indices = np.argsort(dist_matrix[current_index])[1:]
            for i in range(2):
                proj_points._facecolors[sort_indices[i], :] = (1, 0, 0, 1)
                global_fig.canvas.draw()
                plt.figure()
                path = path_images[sort_indices[i]]
                plt.imshow(Image.open(path))
                plt.title(f"{dist_matrix[current_index][sort_indices[i]]}")
                plt.show()

    if show_closest:
        assert original_emb is not None
        if metric == "cosine":
            original_emb_norm = original_emb / np.linalg.norm(
                original_emb, axis=1
            ).reshape(-1, 1)
            dist_matrix = -np.matmul(original_emb_norm, original_emb_norm.T)
        else:
            dist_matrix = distance_matrix(original_emb, original_emb)

    y_unique = np.unique(y)
    n_c = len(y_unique)

    i = 0
    colors = np.empty((n_c, 4))
    colors[0, :] = np.random.rand(4) * 0.8
    while i < n_c - 1:
        color = np.random.rand(4) * 0.8

        if np.any(np.matmul(colors[: i + 1, :], color) > 0.85):
            continue

        colors[i + 1, :] = color
        i += 1

    colors[:, -1] = 1

    global_fig, ax = plt.subplots()

    for i, class_label in enumerate(y_unique):
        select_ind = y==class_label
        ax.scatter(
            x_2d[select_ind, 0],
            x_2d[select_ind, 1],
            color=colors[i],
            picker=True,
            label=str(class_label)
        )

    ax.legend()

    global_fig.canvas.mpl_connect("pick_event", on_pick)

    plt.show()