'''
author
    zhangsihao yang

logs
    2023-10-03
        file created
'''
import os
from typing import Tuple

import numpy as np
import trimesh


def save_obj(vert_data, face_data, filename, color=None):
    '''
    save obj file
    '''
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create a mesh object
    mesh = trimesh.Trimesh(vertices=vert_data, faces=face_data, process=False)

    if color is not None:
        # Assign colors to the mesh
        mesh.visual.vertex_colors = color

    # Export to .obj format
    mesh.export(filename, file_type='obj')


def load_obj(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
    load obj file
    '''
    mesh = trimesh.load(filename, process=False)
    return mesh.vertices, mesh.faces
