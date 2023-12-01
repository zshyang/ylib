'''
the encoder and semantic head for motion evaluation

author
    zhangsihao yang

logs
    2023-11-22
        file created
'''
import clip
import torch
from torch import nn
from ylib.neural_networks.motion_auto_encoder import MotionAutoEncoder  # =====
from ylib.skeleton import Skeleton  # =========================================
from ylib.skeleton import compute_forward  # ==================================
from ylib.skeleton_prior import smal_35_face_joint_indx  # ====================
from ylib.skeleton_prior import smal_ee_idx  # ================================
from ylib.skeleton_prior import smal_key_joint_index  # =======================
from ylib.skeleton_prior import smpl_22_face_joint_indx  # ====================
from ylib.skeleton_prior import smpl_ee_idx  # ================================
from ylib.skeleton_prior import smpl_key_joint_index  # =======================
from ylib.skeleton_prior import smal_35_kinematic_chain as smal_kc  # =========
from ylib.skeleton_prior import t2m_kinematic_chain as smpl_kc  # =============


class MotionEncoder(nn.Module):
    '''
    only the encoder for the motion recall network.
    '''

    def __init__(self, args):
        super().__init__()

        # smal part
        self.smal_ae = MotionAutoEncoder(
            key_joint_index=tuple(smal_key_joint_index),
            num_joints=35,
        )

        # clip
        clip_version = 'ViT-B/32'
        self.clip_model = self._load_and_freeze_clip(clip_version)

        # the text to motion matching head
        self.motion_text_matching_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(49 * 7 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 512)  # 512 is hard coded for clip output size
        )

    def _load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(
            clip_version, device='cpu',
            jit=False
        )  # Must set jit=False for training

        clip.model.convert_weights(  # type: ignore
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad = False

        return clip_model

    def _encode_tokens(self, tokens):
        return self.clip_model.encode_text(tokens).float()

    def forward(self, batch_data):
        '''
        forward function for this network.
        '''
        text_emd = self._encode_tokens(
            batch_data['clip_token'],
        )

        latent, _ = self.smal_ae.enc_forward(
            batch_data['motions'],
            batch_data['offsets'],
        )

        text_pred = self.motion_text_matching_head(
            latent
        )

        return {
            'text_pred': text_pred,
            'text_emd': text_emd,
        }


class LossCalculator(nn.Module):
    '''
    the loss calculator for the motion encoder model
    '''

    def __init__(self, args) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.l2_loss = lambda a, b: (a - b) ** 2
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _compute_cos_loss(self, in_feat, out_feat):
        '''
        compute the cosine similarity loss
        '''
        features_norm = in_feat / in_feat.norm(dim=-1, keepdim=True)
        cos = self.cos_sim(features_norm, out_feat)
        cos_loss = (1 - cos).mean()
        return cos_loss

    def forward(self, batch_data, batch_output):
        '''
        forward function.
        '''
        text_loss = self._compute_cos_loss(
            batch_output['text_pred'],
            batch_output['text_emd'],
        )

        total_loss = text_loss

        return total_loss, {
            'text_loss': text_loss,
        }
