'''
copied from the dataset

author
    zhangsihao yang

logs
    2023-10-03
        file created
'''
import numpy as np


def anime_read(filename):
    """
    filename: .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: [nv, 3], vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: [nt, 3], riangle face data of the 1st frame
        offset_data: [nf-1,nv,3], 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data
