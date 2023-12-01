'''
the code for visualize mesh in matplotlib.

author
    zhangsihao yang

logs
    2023-10-06
        file created
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..skeleton_obj_writer import get_limit_from_joints  # ====================
from ..skeleton_obj_writer import set_axes_equal  # ===========================


def _create_poly3d(vertices, faces, facecolor='cyan'):
    ''' Create polygons from vertices and faces for 3D plotting '''
    polygons = [[vertices[face[j]] for j in range(3)] for face in faces]
    poly3d = Poly3DCollection(
        polygons,
        edgecolor='k',
        facecolor=facecolor,
        alpha=0.25,
        linewidth=0.5
    )
    return poly3d


def _fig_to_image(fig):
    ''' Convert a Matplotlib figure to an image array '''
    fig.canvas.draw()
    data = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8  # type: ignore
    )

    # Reshape the data into an image
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


def _set_axes_properties(ax, x_limit, y_limit, z_limit):
    ''' Set properties for 3D plot axes '''
    # set the ratio
    set_axes_equal(x_limit, y_limit, z_limit, ax)

    # axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False  # Transparent panes

    plt.tight_layout()


def plot_obj(vertices, faces):
    '''
    plot the obj file
    '''
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': '3d'})

    poly3d = _create_poly3d(vertices, faces)
    ax.add_collection3d(poly3d)

    x_limit, y_limit, z_limit = get_limit_from_joints(vertices)
    _set_axes_properties(ax, x_limit, y_limit, z_limit)

    image = _fig_to_image(fig)

    plt.close()

    return image


def save_plot(vertices, faces, save_path: str):
    '''
    wrap of function plot_obj. with save function.
    '''
    image = plot_obj(vertices, faces)
    cv2.imwrite(save_path, image)


def plot_compare_obj(vertices_0, faces_0, vertices_1, faces_1):
    '''
    plot two obj files in one figure
    '''
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': '3d'})

    poly3d_0 = _create_poly3d(vertices_0, faces_0, facecolor='cyan')
    ax.add_collection3d(poly3d_0)

    poly3d_1 = _create_poly3d(vertices_1, faces_1, facecolor='r')
    ax.add_collection3d(poly3d_1)

    x_limit, y_limit, z_limit = get_limit_from_joints(
        np.concatenate([vertices_0, vertices_1], axis=0)
    )

    _set_axes_properties(ax, x_limit, y_limit, z_limit)

    image = _fig_to_image(fig)

    plt.close()

    return image
