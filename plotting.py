import matplotlib.pyplot as plt  # type: ignore [import-not-found]
from mpl_toolkits.mplot3d import Axes3D  # type: ignore [import-not-found]
import numpy as np


def plot_torus(vertices_coords, e2v):
    # print("len(vertices_coords) : {}".format(len(vertices_coords)))
    # print("e2v : {}".format(e2v))
    # print("c2v : {}".format(c2v))
    # Plot the torus
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot vertices_coords
    ax.scatter(
        vertices_coords[:, 0],
        vertices_coords[:, 1],
        vertices_coords[:, 2],
        color="r",
        s=100,
    )

    # Plot vertices indices
    for idx, vertex in enumerate(vertices_coords):
        ax.text(vertex[0], vertex[1], vertex[2], str(idx), color="blue", fontsize=12)

    num_edges = len(e2v)
    colors = plt.cm.viridis(np.linspace(0, 1, num_edges))
    # Plot e2v
    for i, edge in enumerate(e2v):
        v1 = vertices_coords[edge[0]]
        v2 = vertices_coords[edge[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color=colors[i])
        mid_point = v1 + (v2 - v1) * 0.25
        ax.text(
            mid_point[0],
            mid_point[1],
            mid_point[2],
            str(i),
            color=colors[i],
            fontsize=10,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig("torus.png")