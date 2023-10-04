'''
orginaze the code into more readable format

author
    zhangsihao yang

logs
    2023-09-25
        file created
'''
import torch

from ..quaternion import qinv  # ==============================================
from ..quaternion import qrot  # ==============================================
from ..skeleton import Skeleton  # ============================================
from ..skeleton_prior import smpl_22_face_joint_indx  # =======================
from ..skeleton_prior import t2m_kinematic_chain  # ===========================
from ..skeleton_prior import t2m_raw_offsets  # ===============================


def recover_root_rot_pos(data: torch.Tensor):
    '''
    Recover global angle and positions for rotation dataset.
    all the operations are done in pytorch

    inputs:
    -------
    data (..., 263)

    return:
    -------
    root_rot_velocity (B, seq_len, 1)

    root_linear_velocity (B, seq_len, 2)

    root_y (B, seq_len, 1)

    ric_data (B, seq_len, (joint_num - 1)*3)

    rot_data (B, seq_len, (joint_num - 1)*6)

    local_velocity (B, seq_len, joint_num*3)

    foot contact (B, seq_len, 4)
    '''
    # TODO clear the docstring and add dimension description
    # rot_vel : [...]
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)

    # Get Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num: int):
    '''
    recover global positions from rotation invariant coordinates

    inputs:
    -------
    data (..., 263)
        the data from mdm
    joints_num
        the number of joints, include global one
    '''
    # TODO addd dimension description
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # add Y-axis rotation to local joints
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions
    )

    # add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # concate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def get_quaternion(positions):
    '''
    get the quaternion from positions

    inputs:
    -------
    positions (..., joints_num, 3)
    '''
    skel = Skeleton(t2m_raw_offsets, t2m_kinematic_chain, "cpu")
    # TODO fix this
    skel.get_offsets_joints(positions[0])
    # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(
        positions.numpy(), smpl_22_face_joint_indx, smooth_forward=False
    )
    return quat_params, skel


def get_quaternion_with_skel(
    face_joint_index,
    kinematic_chain,
    offsets,
    positions,
):
    '''
    get the quaternion from positions

    inputs:
    -------
    positions [num_frames, joints_num, 3]
        the positions of the joints
    '''
    skel = Skeleton(offsets, kinematic_chain, "cpu")
    skel._offset = torch.tensor(offsets).float()

    # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(
        positions.numpy(), face_joint_index, smooth_forward=False
    )
    return quat_params, skel
