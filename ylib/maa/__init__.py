'''
author
    zhangsihao yang

logs    
    2023-10-08
        file created
'''
import numpy as np
import torch

from ..quaternion import quaternion_to_cont6d_np  # ===========================
from ..quaternion import cont6d_to_matrix_np
from ..rotation_conversions import matrix_to_quaternion
from ..skeleton import Skeleton  # ============================================
from ..smpl_format.pre_processing import get_quaternion_with_skel  # ==========


def convert_motion_to_maa(
    # forward_kinematics: ForwardKinematics,
    positions: torch.Tensor,
    offset: np.ndarray,
    kinematic_chain,
    face_joint_index
) -> np.ndarray:
    '''
    convert the motion from bvh format to maa use inverse kinematics. Here, the
    maa format is not the 263 data format used in mdm. but the 6d ratation of
    each joint and the root positions.
    for the root position, we put three zero at the end of the root position to
    make it 6d
    maa stands for Make-An-Animation: Large-Scale Text-conditional 3D Human 
    Motion Generation.

    inputs:
    -------
    forward_kinematics
        the ForwardKinematics object
    motion : (num_frames, (num_joints - 1) * 3 + 3)
        the motion data in bvh format
    offset : (num_joints, 3)
        the offset of the bvh file

    return:
    -------
    ndarray : (num_frames, (num_joints + 1), 6)
        the motion data in maa format
    '''
    # positions : [1, num_frames, num_joints, 3]
    # positions = forward_kinematics.forward_from_raw(
    #     torch.from_numpy(motion.T[None]).float(),
    #     torch.from_numpy(offset[None]).float(),
    #     world=True,
    # )

    # num_joints = positions.shape[2]
    # if num_joints == 28:
    #     face_joint_index = dme_28_face_joint_index
    #     kinematic_chain = dme_28_kinematic_chain
    # elif num_joints == 23:
    #     valid_joint_index = [i for i in range(23) if i not in [9]]
    #     # remove the 9 th joint
    #     positions = positions[:, :, valid_joint_index, :]
    #     offset = offset[valid_joint_index, :]
    #     face_joint_index = dme_22_face_joint_index
    #     kinematic_chain = dme_22_kinematic_chain
    # else:
    #     raise ValueError('the number of joints is not supported')

    # convert the positions into maa format
    # get the quaternions from the smpl motion
    # quat_params : (num_frames, num_joints, 4)
    quat_params, _ = get_quaternion_with_skel(
        face_joint_index,
        kinematic_chain,
        offset,
        positions,
    )

    # rot_6d : (num_frames, num_joints, 6)
    rot_6d = quaternion_to_cont6d_np(quat_params)

    # append zero to the root position
    # root_positions : (num_frames, 1, 3)
    root_positions = positions[:, :1, :]
    root_positions_6d = np.concatenate(
        [root_positions, np.zeros((root_positions.shape[0], 1, 3))], axis=-1
    )

    # group the rotaion and root position together
    # maa_motion : (num_frames, (num_joints + 1) * 6)
    maa_motion = np.concatenate(
        [rot_6d, root_positions_6d], axis=1
    )

    return maa_motion


def convert_maa_motion_to_positions(
    maa_motion, offsets, kinematic_chain
):
    '''
    convert the maa motion to positions
    TODO
    inputs:
    -------
    maa_motion : (num_frames, (num_joints + 1), 6)
        the motion data in maa format
    '''
    rot_6d = maa_motion[:, :-1, :]
    root_positions = maa_motion[:, -1:, :3]

    skel = Skeleton(offsets, kinematic_chain, "cpu")
    skel._offset = torch.tensor(offsets).float()

    positions = skel.forward_kinematics_np(
        quat_params=matrix_to_quaternion(
            torch.from_numpy(cont6d_to_matrix_np(rot_6d)).float()
        ).numpy(),
        root_pos=root_positions[:, 0, :],
    )

    return positions
