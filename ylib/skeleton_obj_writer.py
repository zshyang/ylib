'''
author:
    zhangsihao yang

logs:
    2023-09-06
        file created
'''
import imageio
import numpy as np
import torch
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from .skeleton_prior import t2m_kinematic_chain


# ------------------------------ 2D Image Drawing --------------------------- #
def get_limit_from_joints(joints):
    '''
    get the limit of the joints

    inputs:
    -------
    joints: (n, 3)

    return:
    -------
    x_limit: (2,)

    y_limit: (2,)

    z_limit: (2,)

    '''
    assert len(joints.shape) == 2

    x_limit = joints[:, 0].min(), joints[:, 0].max()
    y_limit = joints[:, 1].min(), joints[:, 1].max()
    z_limit = joints[:, 2].min(), joints[:, 2].max()

    return x_limit, y_limit, z_limit


def set_axes_equal(x_limits, y_limits, z_limits, ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def draw_skeleton(
    joints, kinematic_chain, limits=None, y_is_up=True
):
    ''' Draw the first frame of the motion
    # joints: (n_frames, n_joints, 3)
    # kinematic_chain: [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], ...]
    '''

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Skeleton')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if limits is None:
        x_limit, y_limit, z_limit = get_limit_from_joints(joints)
    else:
        x_limit, y_limit, z_limit = limits

    set_axes_equal(x_limit, y_limit, z_limit, ax)

    for chain in kinematic_chain:
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i + 1]
            ax.plot(
                [joints[j1, 0], joints[j2, 0]],
                [joints[j1, 1], joints[j2, 1]],
                [joints[j1, 2], joints[j2, 2]],
                color='b', marker='o', markersize=2, linestyle='-'
            )

    for i, joint in enumerate(joints):
        ax.text(joint[0], joint[1], joint[2], f'{i}')

    if y_is_up:
        ax.view_init(elev=100, azim=-90)

    # plt.savefig('./preprocess/skeleton.png')
    fig.canvas.draw()
    data = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore

    # Reshape the data into an image
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # clean the figure
    plt.close()

    return image


def draw_skeleton_with_parent(joints, parent, limits=None, y_is_up=True):
    '''
    draw the skeleton with parent
    '''
    # trimmed_joints: [35, 3]
    trimmed_joints = joints[:len(parent)]
    # trimmed_joints = trimmed_joints.data.cpu().numpy()

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Skeleton')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if limits is None:
        x_limit, y_limit, z_limit = get_limit_from_joints(joints)
    else:
        x_limit, y_limit, z_limit = limits

    set_axes_equal(x_limit, y_limit, z_limit, ax)

    for child, parent in enumerate(parent):
        if parent == -1:
            continue
        ax.plot(
            [trimmed_joints[child, 0], trimmed_joints[parent, 0]],
            [trimmed_joints[child, 1], trimmed_joints[parent, 1]],
            [trimmed_joints[child, 2], trimmed_joints[parent, 2]],
            color='b', marker='o', markersize=2, linestyle='-'
        )

    for i, joint in enumerate(trimmed_joints):
        ax.text(joint[0], joint[1], joint[2], f'{i}')

    ax.view_init(elev=100, azim=-90)

    fig.canvas.draw()
    data = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore

    # Reshape the data into an image
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # closs to prevent memory leak
    plt.close(fig)

    return image


def batch_draw_skeleton_animation(
    frame_joints,
    parent: list,
    verbose: bool = False,
    stop_frame: int = -1,
):
    '''
    draw the skeleton

    inputs:
    -------
    frame_joints: (batch_size, n_frames, n_joints, 3)
        joints positions accross frames
    save_gif_path
        the file path to save the gif
    '''
    if isinstance(frame_joints, torch.Tensor) and frame_joints.is_cuda:
        frame_joints = frame_joints.detach().data.cpu().numpy()
    elif isinstance(frame_joints, torch.Tensor):
        frame_joints = frame_joints.data.detach().numpy()

    images = []

    limits = get_limit_from_joints(frame_joints.reshape(-1, 3))

    batch_size, n_frames, n_joints, _ = frame_joints.shape

    if stop_frame > 0:
        frame_joints = frame_joints[:, :stop_frame]

    if verbose:
        iterable = tqdm(
            frame_joints.reshape((-1, n_joints, 3)), desc="generating",
        )
    else:
        iterable = frame_joints.reshape((-1, n_joints, 3))

    for i, joints in enumerate(iterable):
        if parent is None:
            joint_image = draw_skeleton(
                joints, t2m_kinematic_chain, limits=limits
            )
        else:
            joint_image = draw_skeleton_with_parent(
                joints, parent, limits=limits
            )

        images.append(joint_image)

    # TODO reshape to batch_size, n_frames, c, h, w
    # TODO make the image size smaller.

    return images


def draw_skeleton_animation(
    frame_joints,
    parent: list,
    save_gif_path: str = '',
    verbose: bool = False,
    stop_frame: int = -1,
):
    '''
    draw the skeleton

    inputs:
    -------
    frame_joints: (n_frames, n_joints, 3)
        joints positions accross frames
    save_gif_path
        the file path to save the gif
    '''
    images = []

    limits = get_limit_from_joints(frame_joints.reshape(-1, 3))

    if verbose:
        iterable = tqdm(
            frame_joints, desc=f"generating '{save_gif_path}'",
        )
    else:
        iterable = frame_joints

    for i, joints in enumerate(iterable):
        if parent is None:
            joint_image = draw_skeleton(
                joints, t2m_kinematic_chain, limits=limits
            )
        else:
            joint_image = draw_skeleton_with_parent(
                joints, parent, limits=limits
            )

        images.append(joint_image)

        if stop_frame > 0 and i == stop_frame:
            break

    imageio.mimsave(save_gif_path, images)


# ------------------------------ 3D Mesh Drawing ---------------------------- #
def find_perpendicular_vectors(unit_direction):
    # Create a candidate vector that is not parallel to unit_direction
    if unit_direction[0] == 0 and unit_direction[1] == 0:
        candidate_vector = np.array([1, 0, 0])
    else:
        candidate_vector = np.array([0, 0, 1])

    # Calculate the first perpendicular vector using cross product
    perpendicular1 = np.cross(unit_direction, candidate_vector)

    # Normalize the vector
    perpendicular1 = perpendicular1 / np.linalg.norm(perpendicular1)

    # Calculate the second perpendicular vector using cross product
    perpendicular2 = np.cross(unit_direction, perpendicular1)

    # Normalize the vector
    perpendicular2 = perpendicular2 / np.linalg.norm(perpendicular2)

    return perpendicular1, perpendicular2


def create_square_cone(start_point, end_point, base_width):
    # Calculate direction vector from start to end
    direction = np.array(end_point) - np.array(start_point)

    # Calculate length and unit vector
    length = np.linalg.norm(direction)
    unit_direction = direction / length

    # Generate a perpendicular vector for base
    # perpendicular = np.array([-unit_direction[1], unit_direction[0], 0])
    perpendicular1, perpendicular2 = find_perpendicular_vectors(unit_direction)

    # Calculate half_width for convenience
    half_width = base_width / 2

    # Define the square base vertices
    base_points = np.array(
        [
            start_point + half_width * perpendicular1,
            start_point + half_width * perpendicular2,
            start_point - half_width * perpendicular1,
            start_point - half_width * perpendicular2,
        ]
    )

    # Combine base and apex to create the full set of vertices
    vertices = np.vstack([base_points, end_point])

    # Define faces by vertex indices
    faces = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [0, 2, 1],
            [2, 0, 3],
        ]
    )

    # Create the mesh
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def get_box_diagonal_length(joints):
    assert len(joints.shape) == 2

    diagonal_vector = joints.max(axis=0) - joints.min(axis=0)

    diagonal_length = np.linalg.norm(diagonal_vector)

    return diagonal_length


def draw_skeleton_obj(joints, parents: list):
    '''
    draw the skeleton using trimesh
    '''
    # trimmed_joints: [35, 3]
    trimmed_joints = joints[:len(parents)]
    if isinstance(trimmed_joints, torch.Tensor):
        trimmed_joints = trimmed_joints.data.cpu().numpy()

    # determine the base radius and base width
    box_diagonal_length = get_box_diagonal_length(trimmed_joints)
    base_radius = box_diagonal_length / 150
    base_width = box_diagonal_length / 45

    list_joints = []
    for joint in trimmed_joints:
        sphere = trimesh.creation.icosphere(  # type: ignore
            subdivisions=1,
            radius=base_radius
        )
        sphere.apply_translation(joint)
        list_joints.append(sphere)

    list_bones = []
    for child, parent in enumerate(parents):
        if parent == -1:
            continue
        cone = create_square_cone(
            base_width=base_width,
            end_point=trimmed_joints[child],
            start_point=trimmed_joints[parent],
        )
        list_bones.append(cone)

    # merge list of bones and list of joints
    list_meshes = list_joints + list_bones

    # Combine all the cones into a single mesh
    combined = trimesh.util.concatenate(list_meshes)

    # Export the combined mesh to an OBJ file
    # combined.export('skeleton.obj')  # type: ignore

    return combined
