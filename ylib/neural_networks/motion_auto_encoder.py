'''
motion_auto_encoder

author
    zhangsihao yang

logs
    2023-09-25
        file created
'''
from typing import Tuple

import torch
from torch import nn

from .transformer_backbone import TransformerEncoder


class TemporalPooler(nn.Module):
    '''
    isolated pooler at temporal dimension
    '''

    def __init__(
        self,
        temporal_pool_ratio: int = 4,
    ) -> None:
        '''
        init function
        '''
        super().__init__()

        self.temporal_pooler = nn.AvgPool1d(
            kernel_size=temporal_pool_ratio, stride=temporal_pool_ratio
        )

    def forward(self, temporal_codes):
        '''
        inputs:
        -------
        temporal_codes : [batch_size, num_frames, num_joints, dim]
            the temporal codes

        return:
        -------
        tensor : [batch_size, pooled_num_frames, pooled_num_joints, dim]
            the pooled temporal_codes
        '''
        batch_size, num_frames, num_joints, dim = temporal_codes.shape

        # reshape to apply the 1D pooling on the temporal dimension
        # temporal_codes : [batch_size, num_frames, num_joints * dim]
        temporal_codes = temporal_codes.view(batch_size, num_frames, -1)
        # temporal_codes_bcl : [batch_size, num_joints * dim, num_frames]
        temporal_codes_bcl = temporal_codes.permute(0, 2, 1)
        # pooled_temporal : [batch_size, num_joints * dim, pooled_num_frames]
        pooled_temporal = self.temporal_pooler(temporal_codes_bcl)
        # pooled_temporal : [batch_size, pooled_num_frames, num_joints * dim]
        pooled_temporal = pooled_temporal.permute(0, 2, 1)
        # reshape back to the original format
        pooled_temporal = pooled_temporal.view(batch_size, -1, num_joints, dim)

        return pooled_temporal


class MotionPooler(nn.Module):
    '''
    the pooler for motion
    '''

    def __init__(
        self,
        key_joint_index: Tuple[int, ...],
        temporal_pool_ratio: int = 4,
    ) -> None:
        '''
        init function
        '''
        super().__init__()

        self.temporal_pooler = nn.AvgPool1d(
            kernel_size=temporal_pool_ratio, stride=temporal_pool_ratio
        )

        self.key_joint_index = key_joint_index

    def forward(self, temporal_codes):
        '''
        inputs:
        -------
        temporal_codes : [batch_size, num_frames, num_joints, dim]
            the temporal codes

        return:
        -------
        tensor : [batch_size, pooled_num_frames, pooled_num_joints, dim]
            the pooled temporal_codes
        '''
        batch_size, num_frames, num_joints, dim = temporal_codes.shape

        # reshape to apply the 1D pooling on the temporal dimension
        # temporal_codes : [batch_size, num_frames, num_joints * dim]
        temporal_codes = temporal_codes.view(batch_size, num_frames, -1)
        # temporal_codes_bcl : [batch_size, num_joints * dim, num_frames]
        temporal_codes_bcl = temporal_codes.permute(0, 2, 1)
        # pooled_temporal : [batch_size, num_joints * dim, pooled_num_frames]
        pooled_temporal = self.temporal_pooler(temporal_codes_bcl)
        # pooled_temporal : [batch_size, pooled_num_frames, num_joints * dim]
        pooled_temporal = pooled_temporal.permute(0, 2, 1)
        # reshape back to the original format
        pooled_temporal = pooled_temporal.view(batch_size, -1, num_joints, dim)

        pooled_temporal = pooled_temporal[:, :, self.key_joint_index, :]

        return pooled_temporal


class TemporalUnpooler(nn.Module):
    '''
    the unpooler for temporal
    '''

    def __init__(self, temporal_pool_ratio: int):
        super().__init__()
        self.temporal_pool_ratio = temporal_pool_ratio

    def forward(self, latent):
        """
        Unpool the latent tensor along the temporal dimension.

        inputs:
        -------
        latent : [batch_size, pooled_num_frames, len(key_joint_index), dim]
            tensor

        Returns:
        [batch_size, pooled_num_frames * temporal_pool_ratio, len(key_joint_index), dim]
            tensor
        """

        # Introduce a new dimension for repetition
        # Shape: [batch_size, pooled_num_frames, 1, len(key_joint_index), dim]
        latent_with_new_dim = latent.unsqueeze(2)

        # Expand along the new temporal dimension
        # Shape: [batch_size, pooled_num_frames, temporal_pool_ratio, len(key_joint_index), dim]
        expanded_latent = latent_with_new_dim.expand(
            -1, -1, self.temporal_pool_ratio, -1, -1
        )

        # Reshape to collapse the pooled frames with the repetition dimension
        # Shape: [batch_size, pooled_num_frames * temporal_pool_ratio, len(key_joint_index), dim]
        unpooled_temporal = expanded_latent.reshape(
            latent.size(0), -1, latent.size(2), latent.size(3)
        )

        return unpooled_temporal


class MotionUnPooler(nn.Module):
    '''
    the unpooler for motion
    '''

    def __init__(
        self,
        num_joints: int,
        key_joint_index: Tuple[int, ...],
        temporal_pool_ratio: int = 4,
    ) -> None:
        '''
        init function

        inputs:
        -------
        num_joints : int
            the number of joints to be unpooled to
        '''
        super().__init__()

        self.num_joints = num_joints
        self.temporal_unpooler = TemporalUnpooler(temporal_pool_ratio)
        self.key_joint_index = key_joint_index

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        '''
        inputs:
        -------
        latent : [batch_size, pooled_num_frames, pooled_num_joints, dim]
            the latent tensor

        return:
        -------
        tensor : [batch_size, num_frames, num_joints + 1, dim]
            the unpooled tensor
        '''

        # Temporal Unpooling: Repeat each frame by the temporal_pool_ratio
        # latent: [batch_size, pooled_num_frames, len(key_joint_index), dim]
        # unpooled_temporal: [batch_size, num_frames, len(key_joint_index), dim]
        unpooled_temporal = self.temporal_unpooler(latent)

        # Joints Unpooling:
        # Create a tensor of zeros to hold the unpooled joint data
        # zero_tensor: [batch_size, num_frames, num_joints, dim]
        zero_tensor = torch.zeros(
            unpooled_temporal.shape[0],
            unpooled_temporal.shape[1],
            self.num_joints + 1,
            unpooled_temporal.shape[3],
            device=unpooled_temporal.device
        )

        # Place the data from latent into the appropriate joint positions
        zero_tensor[:, :, self.key_joint_index, :] = unpooled_temporal

        return zero_tensor


class OffsetEncoder(nn.Module):
    '''
    offset encoder
    '''

    def __init__(self, joint_codes_dim, joints_dim) -> None:
        super().__init__()

        self.offset_joint_encoder = TransformerEncoder(
            in_out_dim=(3, joints_dim), d_model=joints_dim, nhead=1,
        )
        self.offset_joint_code_encoder = TransformerEncoder(
            in_out_dim=(joints_dim, joint_codes_dim), d_model=joint_codes_dim,
        )

    def forward(self, offsets):
        '''
        inputs:
        -------
        offsets : [batch_size, num_joints, 3]
            the offset vector for each joint.

        outputs:
        --------
        offset_codes_0 : [batch_size, num_joints, joints_dim]
            the offset codes for each joint
        offset_codes_1 : [batch_size, num_joints, joint_codes_dim]
            the offset codes for each joint codes
        '''
        offset_codes_0 = self.offset_joint_encoder(offsets)
        offset_codes_1 = self.offset_joint_code_encoder(offset_codes_0)
        return offset_codes_0, offset_codes_1


class MotionAutoEncoder(nn.Module):
    '''
    auto encoder for motion
    # TODO write the input and output dimension of the class
    '''

    def __init__(
        self,
        joint_codes_dim: int = 16,
        joints_dim: int = 6,
        key_joint_index: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8),
        num_joints: int = 28,
    ) -> None:
        '''
        inputs:
        -------
        joint_codes_dim
            the dimension of the joint codes
        '''
        super().__init__()

        self._record_config(
            concat_joints_dim=joints_dim * 2,
            joint_codes_dim=joint_codes_dim,
            joints_dim=joints_dim,
            key_joint_index=key_joint_index,
            num_joints=num_joints,
        )

        self.offset_encoder = OffsetEncoder(joint_codes_dim, joints_dim)

        self.joint_encoder = TransformerEncoder(
            in_out_dim=(joints_dim * 2, joint_codes_dim),
            d_model=joint_codes_dim,
        )

        self.temporal_encoder = TransformerEncoder(
            in_out_dim=(
                joint_codes_dim * 2 * (num_joints + 1),
                joint_codes_dim * (num_joints + 1),
            ),
            d_model=joint_codes_dim * (num_joints + 1),
        )

        self.pooler = MotionPooler(
            key_joint_index, temporal_pool_ratio=4
        )

        self.unpooler = MotionUnPooler(
            num_joints=num_joints,
            key_joint_index=key_joint_index,
            temporal_pool_ratio=4
        )

        self.temporal_decoder = TransformerEncoder(
            in_out_dim=(
                joint_codes_dim * 2 * (num_joints + 1),
                joint_codes_dim * (num_joints + 1),
            ),
            d_model=joint_codes_dim * (num_joints + 1),
        )

        self.joint_decoder = TransformerEncoder(
            in_out_dim=(joint_codes_dim * 2, joints_dim),
            d_model=joint_codes_dim,
        )
        self.joint_decoder = nn.Linear(joint_codes_dim * 2, joints_dim)

    def _record_config(self, **kwargs):
        '''
        record the config
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _concat_offset(self, motions, offset_codes):
        '''
        concat the offset to the motion

        inputs:
        -------
        motions : [batch_size, num_frames, num_joints, dim]
            the batch of motions
        offset_codes : [batch_size, num_joints, dim]
            the batch of offset codes

        return:
        -------
        motions : [batch_size, num_frames, num_joints, dim * 2]
            the batch of motions with offset codes
        '''
        batch_size, num_frames, num_joints, dim = motions.shape

        offset_codes = offset_codes.view(batch_size, 1,  num_joints, dim)
        offset_codes = offset_codes.repeat(1, num_frames, 1, 1)

        # motions : [batch_size, num_frames, num_joints, dim * 2]
        motions = torch.cat((motions, offset_codes), dim=-1)

        return motions

    def _encode_offsets(self, offsets):
        '''
        encode the offsets

        inputs:
        -------
        offsets : [batch_size, num_joints, 3]
            the offset of batched skeleton

        return:
        -------
        offset_codes_0 : [batch_size, num_joints, joints_dim]
            for compatibility with the joint encoder
        offset_codes_1 : [batch_size, num_joints, joint_codes_dim]
            for deeper feature of joints
        '''
        # enlarge offset with zero in the end
        # offsets : [batch_size, num_joints, joints_dim]
        # augment_offsets : [batch_size, num_joints + 1, joints_dim]
        augment_offsets = torch.cat(
            (offsets, torch.zeros_like(offsets[:, 0:1])), dim=-2
        )
        offset_codes_0, offset_codes_1 = self.offset_encoder(augment_offsets)
        return offset_codes_0, offset_codes_1

    def _encode_joints(self, motions, offset_codes_0):
        '''
        encode the joints

        inputs:
        -------
        motions : [batch_size, num_frames, num_joints, joints_dim]
            the batch of motions
        offset_codes_0 : [batch_size, num_joints, joints_dim]
            the offset codes for each joint

        return:
        -------
        joint_codes : [batch_size, num_frames, num_joints, joint_codes_dim]
            the joint codes for each joint
        '''
        batch_size, num_frames, num_joints, _ = motions.shape

        # motions : [batch_size, num_frames, num_joints, concat_joints_dim]
        motions = self._concat_offset(motions, offset_codes_0)

        motions = motions.view(batch_size * num_frames, num_joints, -1)

        # forward motions though pose encoder
        # joint_codes : [batch_size * num_frames, num_joints, joint_codes_dim]
        joint_codes = self.joint_encoder(motions)

        joint_codes_dim = joint_codes.shape[2]

        joint_codes = joint_codes.view(
            batch_size, num_frames, num_joints, joint_codes_dim
        )

        return joint_codes

    def _encode_temporal(self, joint_codes, offset_codes_1):
        '''
        encode the temporal

        inputs:
        -------
        joint_codes : [batch_size, num_frames, num_joints, joint_codes_dim]
            the joint codes for each joint
        offset_codes_1 : [batch_size, num_joints, joint_codes_dim]
            the offset codes for each joint codes

        return:
        -------
        temporal_codes : [batch_size, pooled_num_frames, temporal_codes_dim]
            the temporal codes for each frame
        '''
        batch_size, num_frames, num_joints, joint_codes_dim = joint_codes.shape

        # joint_codes : [batch_size, num_frames, num_joints, joint_codes_dim * 2]
        joint_codes = self._concat_offset(joint_codes, offset_codes_1)
        # joint_codes : [batch_size, num_frames, num_joints * joint_codes_dim * 2]
        joint_codes = joint_codes.view(batch_size, num_frames, -1)

        # temporal_codes = [batch_size, num_frames, temporal_codes_dim]
        # temporal_codes_dim = num_joints * joint_codes_dim
        temporal_codes = self.temporal_encoder(joint_codes)

        temporal_codes = temporal_codes.view(
            batch_size, num_frames, num_joints, joint_codes_dim
        )
        return temporal_codes

    def _pool_and_flatten(self, temporal_codes):
        ''' 
        pool and flatten the temporal codes

        inputs:
        -------
        temporal_codes : [batch_size, num_frames, num_joints, joint_codes_dim]
            the temporal codes for each frame

        return:
        -------
        latent : [batch_size, pooled_num_frames, pooled_num_joints, joint_codes_dim]
            the pooled temporal codes
        flatten_latent : [batch_size, pooled_num_frames * pooled_num_joints * joint_codes_dim]
            the flatten latent
        '''
        batch_size, _, _, _ = temporal_codes.shape

        # latent : [batch_size, pooled_num_frames, pooled_num_joints, joint_codes_dim]
        # pooled_temporal_codes_dim = pooled_num_joints * joint_codes_dim
        latent = self.pooler(temporal_codes)
        flatten_latent = latent.view(batch_size, -1)

        return latent, flatten_latent

    def _unpool_and_decode(self, latent, offset_codes_1):
        ''' 
        unpool and decode the latent

        inputs:
        -------
        latent : [batch_size, pooled_num_frames, pooled_num_joints, joint_codes_dim]
            the pooled temporal codes
        offset_codes_1 : [batch_size, num_joints, joint_codes_dim]
            the offset codes for each joint codes
        '''
        # temporal_codes : [batch_size, num_frames, num_joints, joint_codes_dim]
        temporal_codes = self.unpooler(latent)

        batch_size, num_frames, num_joints, _ = temporal_codes.shape

        # temporal_codes : [batch_size, num_frames, num_joints, joint_codes_dim * 2]
        temporal_codes = self._concat_offset(temporal_codes, offset_codes_1)

        temporal_codes = self.temporal_decoder(
            temporal_codes.view(batch_size, num_frames, -1)
        ).view(batch_size, num_frames, num_joints, -1)

        temporal_codes = self._concat_offset(temporal_codes, offset_codes_1)

        rec_motions = self.joint_decoder(
            temporal_codes.view(batch_size * num_frames, num_joints, -1)
        ).view(batch_size, num_frames, num_joints, -1)

        return rec_motions

    def forward(
        self, motions: torch.Tensor, offsets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        forward motion and offset to get latent and rec_motions

        inputs:
        -------
        motions: [batch_size, num_frames, num_joints + 1, joints_dim]
            the batch of motions
        offsets: [batch_size, num_joints, 3]
            the batch of offsets

        return:
        -------
        latent : [batch_size, pooled_num_frames, pooled_num_joints, joint_codes_dim]
            the flatten latent
        rec_motions : [batch_size, num_frames, num_joints + 1, joints_dim]
            the batch of motions
        '''
        offset_codes_0, offset_codes_1 = self._encode_offsets(offsets)
        joint_codes = self._encode_joints(motions, offset_codes_0)
        temporal_codes = self._encode_temporal(joint_codes, offset_codes_1)
        latent, _ = self._pool_and_flatten(temporal_codes)
        rec_motions = self._unpool_and_decode(
            latent, offset_codes_1
        )
        return latent, rec_motions

    def enc_forward(
        self, motions: torch.Tensor, offsets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        forward motion and offset to get latent and rec_motions

        inputs:
        -------
        motions: [batch_size, num_frames, num_joints + 1, joints_dim]
            the batch of motions
        offsets: [batch_size, num_joints, 3]
            the batch of offsets

        return:
        -------
        latent : [batch_size, pooled_num_frames, pooled_num_joints, joint_codes_dim]
            the latent
        rec_motions : [batch_size, num_frames, num_joints + 1, joints_dim]
            the batch of motions
        '''
        offset_codes_0, offset_codes_1 = self._encode_offsets(offsets)
        joint_codes = self._encode_joints(motions, offset_codes_0)
        temporal_codes = self._encode_temporal(joint_codes, offset_codes_1)
        latent, _ = self._pool_and_flatten(temporal_codes)

        return latent, offset_codes_1

    def dec_forward(self, latent, offsets):
        '''
        inputs:
        -------
        '''
        offset_codes_1 = self.offset_forward(offsets)
        rec_motions = self._unpool_and_decode(
            latent, offset_codes_1
        )

        return rec_motions

    def offset_forward(self, offsets: torch.Tensor):
        '''
        inputs:
        -------
        '''
        _, offset_codes_1 = self._encode_offsets(offsets)
        return offset_codes_1


class MotionEncoder(nn.Module):
    '''
    the encoder of the motion autoencoder.
    '''

    def __init__(
        self,
        joint_codes_dim: int = 16,
        joints_dim: int = 6,
        key_joint_index: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8),
        num_joints: int = 28,
    ) -> None:
        '''
        inputs:
        -------
        joint_codes_dim
            the dimension of the joint codes
        '''
        super().__init__()

        self._record_config(
            concat_joints_dim=joints_dim * 2,
            joint_codes_dim=joint_codes_dim,
            joints_dim=joints_dim,
            key_joint_index=key_joint_index,
            num_joints=num_joints,
        )

        self.offset_encoder = OffsetEncoder(joint_codes_dim, joints_dim)

        self.joint_encoder = TransformerEncoder(
            in_out_dim=(joints_dim * 2, joint_codes_dim),
            d_model=joint_codes_dim,
        )

        self.temporal_encoder = TransformerEncoder(
            in_out_dim=(
                joint_codes_dim * 2 * (num_joints + 1),
                joint_codes_dim * (num_joints + 1),
            ),
            d_model=joint_codes_dim * (num_joints + 1),
        )

        self.pooler = MotionPooler(
            key_joint_index, temporal_pool_ratio=4
        )

    def _record_config(self, **kwargs):
        '''
        record the config
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _concat_offset(self, motions, offset_codes):
        '''
        concat the offset to the motion

        inputs:
        -------
        motions : [batch_size, num_frames, num_joints, dim]
            the batch of motions
        offset_codes : [batch_size, num_joints, dim]
            the batch of offset codes

        return:
        -------
        motions : [batch_size, num_frames, num_joints, dim * 2]
            the batch of motions with offset codes
        '''
        batch_size, num_frames, num_joints, dim = motions.shape

        offset_codes = offset_codes.view(batch_size, 1,  num_joints, dim)
        offset_codes = offset_codes.repeat(1, num_frames, 1, 1)

        # motions : [batch_size, num_frames, num_joints, dim * 2]
        motions = torch.cat((motions, offset_codes), dim=-1)

        return motions

    def _encode_offsets(self, offsets):
        '''
        encode the offsets

        inputs:
        -------
        offsets : [batch_size, num_joints, 3]
            the offset of batched skeleton

        return:
        -------
        offset_codes_0 : [batch_size, num_joints, joints_dim]
            for compatibility with the joint encoder
        offset_codes_1 : [batch_size, num_joints, joint_codes_dim]
            for deeper feature of joints
        '''
        # enlarge offset with zero in the end
        # offsets : [batch_size, num_joints, joints_dim]
        # augment_offsets : [batch_size, num_joints + 1, joints_dim]
        augment_offsets = torch.cat(
            (offsets, torch.zeros_like(offsets[:, 0:1])), dim=-2
        )
        offset_codes_0, offset_codes_1 = self.offset_encoder(augment_offsets)
        return offset_codes_0, offset_codes_1

    def _encode_joints(self, motions, offset_codes_0):
        '''
        encode the joints

        inputs:
        -------
        motions : [batch_size, num_frames, num_joints, joints_dim]
            the batch of motions
        offset_codes_0 : [batch_size, num_joints, joints_dim]
            the offset codes for each joint

        return:
        -------
        joint_codes : [batch_size, num_frames, num_joints, joint_codes_dim]
            the joint codes for each joint
        '''
        batch_size, num_frames, num_joints, _ = motions.shape

        # motions : [batch_size, num_frames, num_joints, concat_joints_dim]
        motions = self._concat_offset(motions, offset_codes_0)

        motions = motions.view(batch_size * num_frames, num_joints, -1)

        # forward motions though pose encoder
        # joint_codes : [batch_size * num_frames, num_joints, joint_codes_dim]
        joint_codes = self.joint_encoder(motions)

        joint_codes_dim = joint_codes.shape[2]

        joint_codes = joint_codes.view(
            batch_size, num_frames, num_joints, joint_codes_dim
        )

        return joint_codes

    def _encode_temporal(self, joint_codes, offset_codes_1):
        '''
        encode the temporal

        inputs:
        -------
        joint_codes : [batch_size, num_frames, num_joints, joint_codes_dim]
            the joint codes for each joint
        offset_codes_1 : [batch_size, num_joints, joint_codes_dim]
            the offset codes for each joint codes

        return:
        -------
        temporal_codes : [batch_size, pooled_num_frames, temporal_codes_dim]
            the temporal codes for each frame
        '''
        batch_size, num_frames, num_joints, joint_codes_dim = joint_codes.shape

        # joint_codes : [batch_size, num_frames, num_joints, joint_codes_dim * 2]
        joint_codes = self._concat_offset(joint_codes, offset_codes_1)
        # joint_codes : [batch_size, num_frames, num_joints * joint_codes_dim * 2]
        joint_codes = joint_codes.view(batch_size, num_frames, -1)

        # temporal_codes = [batch_size, num_frames, temporal_codes_dim]
        # temporal_codes_dim = num_joints * joint_codes_dim
        temporal_codes = self.temporal_encoder(joint_codes)

        temporal_codes = temporal_codes.view(
            batch_size, num_frames, num_joints, joint_codes_dim
        )
        return temporal_codes

    def _pool_and_flatten(self, temporal_codes):
        ''' 
        pool and flatten the temporal codes

        inputs:
        -------
        temporal_codes : [batch_size, num_frames, num_joints, joint_codes_dim]
            the temporal codes for each frame

        return:
        -------
        latent : [batch_size, pooled_num_frames, pooled_num_joints, joint_codes_dim]
            the pooled temporal codes
        flatten_latent : [batch_size, pooled_num_frames * pooled_num_joints * joint_codes_dim]
            the flatten latent
        '''
        batch_size, _, _, _ = temporal_codes.shape

        # latent : [batch_size, pooled_num_frames, pooled_num_joints, joint_codes_dim]
        # pooled_temporal_codes_dim = pooled_num_joints * joint_codes_dim
        latent = self.pooler(temporal_codes)
        flatten_latent = latent.view(batch_size, -1)

        return latent, flatten_latent

    def forward(
        self, motions: torch.Tensor, offsets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        forward motion and offset to get latent and rec_motions

        inputs:
        -------
        motions: [batch_size, num_frames, num_joints + 1, joints_dim]
            the batch of motions
        offsets: [batch_size, num_joints, 3]
            the batch of offsets

        return:
        -------
        latent : [batch_size, pooled_num_frames, pooled_num_joints, joint_codes_dim]
            the flatten latent
        rec_motions : [batch_size, num_frames, num_joints + 1, joints_dim]
            the batch of motions
        '''
        offset_codes_0, offset_codes_1 = self._encode_offsets(offsets)
        joint_codes = self._encode_joints(motions, offset_codes_0)
        temporal_codes = self._encode_temporal(joint_codes, offset_codes_1)
        latent, _ = self._pool_and_flatten(temporal_codes)

        return latent
