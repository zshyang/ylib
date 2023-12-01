'''
author
    zhangsihao yang

logs
    2023-10-07
        file created
'''
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .smal_model import SMALModel

SMAL_FACE_NUM = 7774
SMAL_NO_TAIL_FACE_NUM = 7590
SMAL_VERTEX_NUM = 3889
SMAL_NO_TAIL_VERTEX_NUM = 3800
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, 'smal_no_tail_index.json')
with open(
    json_path, 'r', encoding='utf-8'
) as file:
    SMAL_NO_TAIL_INDEX = json.load(file)


def init_smal_model(
    list_beta: List[float], device
) -> Tuple[nn.Parameter, SMALModel]:
    '''
    initialize the smal model with an initial guess of beta.
    '''
    # this is just for initialize SMALModel
    beta_np = np.array(list_beta, dtype=np.float32)

    # this is for forward function
    beta = torch.tensor(beta_np).to(device).float()
    beta = torch.nn.Parameter(beta, requires_grad=True)

    # for initiazlie SMALModel
    betas_logscale_np = np.zeros((1, 6), dtype=np.float32)

    smal_model = SMALModel(
        beta_np, device, betas_logscale_np,
    )

    return beta,  smal_model


def no_tail_vert_to_smal_vert(
    no_tail_vertices
):
    '''
    take some time to figure out the index. 
    '''

    tail_index = 449

    smal_vertices = np.zeros((3889, 3), dtype=np.float32)
    smal_vertices[:] = no_tail_vertices[tail_index]
    smal_vertices[SMAL_NO_TAIL_INDEX] = no_tail_vertices

    return smal_vertices
