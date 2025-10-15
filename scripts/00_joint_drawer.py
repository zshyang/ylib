"""
author
    zhangsihao yang
logs
    2025-10-15
        file created
"""

from ylib.skeleton_obj_writer import draw_skeleton_obj
import numpy as np
import os


def main():
    """
    Main function to draw skeleton from joint and parent information.
    """

    # 1. load joints
    joints = np.load("demo/joints.npy")
    # process the joints shape to (J, 3)
    joints = joints.reshape(-1, 3)

    # 2. load parents
    parents = np.load("demo/parents.npy")
    # process the parents shape to (J, ) and convert to list
    parents = parents.reshape(-1).tolist()

    # 3. draw skeleton and export to obj file
    tmesh = draw_skeleton_obj(joints=joints, parents=parents)
    os.makedirs("demo_output", exist_ok=True)
    tmesh.export("demo_output/skeleton.obj")


if __name__ == "__main__":
    main()
