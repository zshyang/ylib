''' modification of SMALify/smal_model/smal_torch.py

author:
    zhangsihao yang

logs:
    2023-09-03
        file created
'''
import pickle as pkl
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .batch_lbs import batch_global_rigid_transformation

SMAL_FILE = '/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/smalify/SMALify/data/SMALST/smpl_models/my_smpl_00781_4_all.pkl'
SMAL_SYM_FILE = '/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/smalify/SMALify/data/SMALST/smpl_models/symIdx.pkl'


def align_smal_template_to_symmetry_axis(v, sym_file):
    # These are the indexes of the points that are on the symmetry axis
    I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 55, 119, 120, 163, 209, 210, 211, 213, 216, 227, 326, 395, 452, 578, 910, 959, 964, 975, 976, 977, 1172, 1175, 1176, 1178, 1194, 1243, 1739, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809,
         1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1870, 1919, 1960, 1961, 1965, 1967, 2003]

    v = v - np.mean(v)
    y = np.mean(v[I, 1])
    v[:, 1] = v[:, 1] - y
    v[I, 1] = 0

    # symIdx = pkl.load(open(sym_path))
    with open(sym_file, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'  # type: ignore
        symIdx = u.load()

    left = v[:, 1] < 0
    right = v[:, 1] > 0
    center = v[:, 1] == 0
    v[left[symIdx]] = np.array([1, -1, 1])*v[left]

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    try:
        assert (len(left_inds) == len(right_inds))
    except:
        import pdb
        pdb.set_trace()

    return v, left_inds, right_inds, center_inds


def load_smal_model():
    with open(SMAL_FILE, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'  # type: ignore
        dd = u.load()
    return dd


def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


def align_smal(v_template):
    v_sym, _, _, _ = align_smal_template_to_symmetry_axis(
        v_template,
        sym_file=SMAL_SYM_FILE
    )
    return v_sym


def batch_skew(vec, batch_size=None, opts=None):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    if batch_size is None:
        batch_size = vec.shape.as_list()[0]
    col_inds = torch.LongTensor([1, 2, 3, 5, 6, 7])
    indices = torch.reshape(torch.reshape(torch.arange(
        0, batch_size) * 9, [-1, 1]) + col_inds, [-1, 1])
    updates = torch.reshape(
        torch.stack(
            [
                -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                vec[:, 0]
            ],
            dim=1), [-1])
    out_shape = [batch_size * 9]
    res = torch.Tensor(np.zeros(out_shape[0])).to(device=vec.device)
    res[np.array(indices.flatten())] = updates
    res = torch.reshape(res, [batch_size, 3, 3])

    return res


def batch_rodrigues(theta, opts=None):
    """
    Theta is Nx3
    """
    batch_size = theta.shape[0]

    angle = (torch.norm(theta + 1e-8, p=2, dim=1)).unsqueeze(-1)
    r = (torch.div(theta, angle)).unsqueeze(-1)

    angle = angle.unsqueeze(-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    outer = torch.matmul(r, r.transpose(1, 2))

    eyes = torch.eye(3).unsqueeze(0).repeat(
        [batch_size, 1, 1]).to(device=theta.device)
    H = batch_skew(r, batch_size=batch_size, opts=opts)
    R = cos * eyes + (1 - cos) * outer + sin * H

    return R


class SMALModel(nn.Module):
    def __init__(
        self,
        beta: np.ndarray,
        device,
        log_betascale: np.ndarray
    ) -> None:
        super(SMALModel, self).__init__()

        self.device = device

        # self.beta: [1, 20]
        self.beta = torch.from_numpy(beta).to(device=device)[None]

        # self.betas_logscale： [1, 6]
        self.betas_logscale = torch.from_numpy(
            log_betascale
        ).to(device=device)[None]

        dd = load_smal_model()

        # self.faces: [7774, 3]
        self.faces = dd['f']

        # v_template
        v_template = dd['v_template']
        v_sym = align_smal(v_template)
        self.v_template = Variable(
            torch.Tensor(v_sym),
            requires_grad=False
        ).to(device=device)  # [3889, 3]
        self.num_verts = self.v_template.shape[0]

        # shapedir
        num_beta = dd['shapedirs'].shape[-1]
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']),
            [-1, num_beta]
        ).T.copy()
        self.shapedirs = Variable(
            torch.Tensor(shapedir),
            requires_grad=False
        ).to(device=device)

        # J_regressor
        self.J_regressor = Variable(
            torch.Tensor(dd['J_regressor'].T.todense()),
            requires_grad=False
        ).to(device=device)

        # posedirs
        num_pose_basis = dd['posedirs'].shape[-1]
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]
        ).T
        self.posedirs = Variable(
            torch.Tensor(posedirs),
            requires_grad=False
        ).to(device=device)

        # LBS weights
        self.weights = Variable(
            torch.Tensor(undo_chumpy(dd['weights'])),
            requires_grad=False
        ).to(device=device)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

    def _regress_joint(self, vertices):
        ''' 
        regress joint from vertices
        '''
        assert len(vertices.shape) == 3
        Jx = torch.matmul(vertices[:, :, 0], self.J_regressor)
        Jy = torch.matmul(vertices[:, :, 1], self.J_regressor)
        Jz = torch.matmul(vertices[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)
        return J

    def __call__(
        self,
        beta: torch.Tensor,
        theta: torch.Tensor,
        betas_logscale: torch.Tensor,
    ):
        '''
        get joints position

        args:
        -----
            theta: [bs, 35, 1, 3]
        '''
        self.beta = beta

        # 1. Add shape blend shapes
        num_used_beta = self.beta.shape[-1]
        v_shaped = self.v_template + torch.reshape(
            torch.matmul(
                self.beta,
                self.shapedirs[:num_used_beta, :]
            ),
            [-1, self.num_verts, 3]
        )

        # 2. Infer shape-dependent joint locations.
        J = self._regress_joint(v_shaped)

        # 3. Add pose blend shapes
        # Rs: [N, 35, 3, 3]
        Rs = torch.reshape(
            batch_rodrigues(
                torch.reshape(theta, [-1, 3])
            ),
            [-1, 35, 3, 3]
        )

        # Ignore global rotation.
        pose_feature = torch.reshape(
            Rs[:, 1:, :, :] - torch.eye(3).to(self.beta.device),
            [-1, 306]
        )

        v_posed = v_shaped + torch.reshape(
            torch.matmul(pose_feature, self.posedirs),
            [-1, self.num_verts, 3]
        )

        # 4. Get the global joint location
        # A: [bs, 35, 4, 4]
        self.J_transformed, A = batch_global_rigid_transformation(
            Rs,
            J,
            self.parents,
            betas_logscale=betas_logscale
        )

        # 5. Do skinning:
        num_batch = theta.shape[0]

        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])

        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])),
            [num_batch, -1, 4, 4]
        )
        v_posed_homo = torch.cat(
            [
                v_posed,
                torch.ones(
                    [num_batch, v_posed.shape[1], 1]
                ).to(device=self.beta.device)
            ],
            2
        )
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        # if trans is None:
        trans = torch.zeros((num_batch, 3)).to(device=self.beta.device)

        verts = verts + trans[:, None, :]

        # Get joints:
        joints = self._regress_joint(verts)

        # joints = torch.cat(
        #     [
        #         joints,
        #         verts[:, None, 1863],  # end_of_nose
        #         verts[:, None,   26],  # chin
        #         verts[:, None, 2124],  # right ear tip
        #         verts[:, None,  150],  # left ear tip
        #         verts[:, None, 3055],  # left eye
        #         verts[:, None, 1097],  # right eye
        #     ],
        #     dim=1
        # )

        return verts, joints, Rs, v_shaped
