'''
author
    zhangsihao yang

logs
    2023-09-18
        file created
'''
import numpy as np
import torch
from ylib.quaternion import qbetween_np, qinv_np, qmul_np, qrot_np

from .quaternion import cont6d_to_matrix, qbetween, qrot


def normlize_np(vector):
    '''
    inputs:
    -------
    vector : (batch_size, 3)
    '''
    return vector / np.sqrt((vector**2).sum(axis=-1))[:, np.newaxis]


def get_forward_direction(face_joint_idx, joints):
    '''
    get the forward direction vector of the smal model

    inputs:
    -------
    joints : (batch_size, joints_num, 3)
        the joints position of the smal model
    face_joint_idx : (4,)

    return:
    -------
    return a forward direction vector that is on the XZ plane
    '''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_idx

    # across1 : (batch_size, 3)
    across1 = joints[:, r_hip] - joints[:, l_hip]
    # across2 : (batch_size, 3)
    across2 = joints[:, sdr_r] - joints[:, sdr_l]

    across = across1 + across2

    # across : (batch_size, 3)
    across = normlize_np(across)

    # forward : (batch_size, 3)
    forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)

    # forward : (batch_size, 3)
    forward = normlize_np(forward)

    return forward


def safe_l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    '''
    safely performs L2 normalization on a tensor at the last dimension

    inputs:
    -------
    tensor (torch.Tensor): 
        The input tensor to be normalized.

    return:
    -------
    torch.Tensor: 
        The L2-normalized tensor. Returns the original tensor if its L2 
        norm is zero.
    '''
    l2_norm = torch.norm(tensor, dim=-1, keepdim=True, p=2)

    # Create a mask for vectors with non-zero L2 norm
    mask = (l2_norm != 0.)

    # Handle division by zero by using masked division
    normalized_tensor = torch.where(mask, tensor / l2_norm, tensor)

    return normalized_tensor


def compute_forward(face_joint_index, joints: torch.tensor) -> torch.tensor:
    '''
    compute the forward direction of the joints

    inputs:
    -------
    joints : torch.tensor (batch_size, num_joints, 3)
        the joints that has been moved to the origin

    return:
    -------
    forward : torch.tensor (batch_size, 3)
        the forward direction of the joints
    '''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_index

    # across1 across2 across : [batch_size, 3]
    across1 = joints[:, r_hip] - joints[:, l_hip]
    across2 = joints[:, sdr_r] - joints[:, sdr_l]
    across = across1 + across2
    across = safe_l2_normalize(across)

    # y_axis : [1, 3]
    y_axis = torch.tensor([[0, 1, 0]], dtype=torch.float32).to(joints.device)

    # forward : [batch_size, 3]
    forward = torch.cross(y_axis, across, dim=-1)
    forward = safe_l2_normalize(forward)

    return forward


def from_offset_to_joints(offset, parents) -> np.ndarray:
    '''
    convert the offset to joints

    inputs:
    -------
    offset : (joints_num, 3)
    parents : (joints_num,)
    '''
    # check the parents is valid
    for i, parent in enumerate(parents):
        if parent == -1:
            continue
        assert parent < i, 'the parents is not valid'

    # joints : (joints_num, 3)
    joints = np.zeros(offset.shape)

    # assign the root position
    joints[0] = offset[0]

    for i in range(1, len(parents)):
        joints[i] = offset[i] + joints[parents[i]]
    return joints


def make_joints_face_z(face_joint_index, joints: torch.tensor) -> torch.tensor:
    '''
    make the joints face towards z+ direction

    inputs:
    -------
    face_joint_index : [4]
        the index of the joints that are used to compute the forward direction
    joints : torch.tensor [batch_size, num_joints, 3]
        the joints that has been moved to the origin

    return:
    -------
    the new joint positions that face towards z+ direction
    '''
    # forward : (batch_size, 3)
    forward = compute_forward(face_joint_index, joints)

    # target : [1, 3]
    target = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(joints.device)

    # root_quat : [batch_size, 4]
    root_quat = qbetween(forward, target)

    # expand the second dimension and repeat 22 times
    # root_quat : [batch_size, num_joints, 4]
    root_quat = root_quat.unsqueeze(1).repeat(1, joints.shape[1], 1)

    # the joints that all face towards z+ direction
    joints = qrot(root_quat, joints)

    return joints


class Skeleton:
    '''
    Skeleton class
    '''

    def __init__(self, offset, kinematic_tree, device):
        self.device = device
        if isinstance(offset, np.ndarray):
            offset = torch.tensor(offset).float()
        # if isinstance(offset, torch.tensor):
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def get_offsets_joints(self, joints):
        '''
        Get the offsets from the joints

        inputs:
        -------
        joints : (joints_num, 3)
        '''
        assert len(joints.shape) == 2

        _offsets = self._raw_offset.clone()

        for i in range(1, self._raw_offset.shape[0]):
            _offsets[i] = torch.norm(
                joints[i] - joints[self._parents[i]],
                p=2,
                dim=0
            ) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    def inverse_kinematics_np(
        self, joints, face_joint_idx, smooth_forward=False
    ):
        '''
        inputs:
        -------
        joints : (batch_size, joints_num, 3)
            the joints xyz position
        '''
        assert len(face_joint_idx) == 4
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(
                forward, 20, axis=0, mode='nearest'
            )
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_offset_np[chain[j+1]][np.newaxis, ...].repeat(
                    len(joints), axis=0
                )
                # print(u.shape)
                # (batch, 3)
                v = joints[:, chain[j+1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                # print(u.shape, v.shape)
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    def forward_kinematics_np(
        self, quat_params, root_pos, skel_joints=None, do_root_R=True
    ):
        '''
        inputs:
        -------
        quat_params (batch_size, joints_num, 4)
        joints (batch_size, joints_num, 3)
        root_pos (batch_size, 3)
        '''

        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)

        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            R = quat_params[:, 0]
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])

                offset_vec = offsets[:, chain[i]]

                joints[:, chain[i]] = qrot_np(
                    R, offset_vec
                ) + joints[:, chain[i - 1]]

        return joints

    @classmethod
    def forward_kinematics_cont6d(
        cls, cont6d_params, offsets=None,
        kinematic_tree=None,
        skel_joints=None,
        do_root_R=True,
        world=True,
    ):
        '''
        forward kinematics with cont6d parameters

        inputs:
        -------
        cont6d_params (batch_size, num_frames, joints_num + 1, 6)
        joints (batch_size, joints_num, 3)
        root_pos (batch_size, 3)
        '''

        # if skel_joints is not None:
        # skel_joints = torch.from_numpy(skel_joints)
        # offsets = self.get_offsets_joints_batch(skel_joints)
        # if len(self._offset.shape) == 2:
        # offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)

        # position : [batch_size, num_frames, 1, 3]
        position = cont6d_params[:, :, -1:, 0:3]

        # rotation : [batch_size, num_frames, num_joints, 6]
        rotation = cont6d_params[:, :, :-1, :]

        # joints : [batch_size, num_frames, num_joints, 3]
        joints = torch.zeros(
            rotation.shape[:-1] + (3,)
        ).to(cont6d_params.device)

        joints[..., :1, :] = position

        for chain in kinematic_tree:
            # if do_root_R:
            matR = cont6d_to_matrix(cont6d_params[..., 0, :])
            # else:
            #     matR = torch.eye(3).expand(
            #         (len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                # matR : [batch_size, num_frames, 3, 3]
                matR = torch.matmul(
                    matR,
                    cont6d_to_matrix(cont6d_params[..., chain[i], :])
                )
                # offset_vec : [batch_size, 1, 3, 1]
                offset_vec = offsets[:, chain[i], :].unsqueeze(-1).unsqueeze(1)
                # print(matR.shape, offset_vec.shape)
                joints[..., chain[i], :] = torch.matmul(
                    matR, offset_vec
                ).squeeze(-1)

                if world:
                    joints[..., chain[i], :] = joints[..., chain[i], :] + \
                        joints[..., chain[i-1], :]
        return joints
