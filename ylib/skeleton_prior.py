'''
author
    zhangsihao yang

logs
    2023-09-17
        file created
'''
import numpy as np

# ================================== SMPL 22 ==================================
# parents | offsets | kinematic_chain | face_joint_index | key_joint_index
# ee_idx
smpl_22_parents = [
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    -1,  0, 0, 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19,
]

t2m_raw_offsets = np.array(
    [
        [0,  0, 0],
        [1,  0, 0],
        [-1, 0, 0],
        [0,  1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0,  1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0,  1, 0],
        [0,  0, 1],
        [0,  0, 1],
        [0,  1, 0],
        [1,  0, 0],
        [-1, 0, 0],
        [0,  0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0]
    ]
)

t2m_kinematic_chain = [
    [0,      2,  5,  8, 11],  # root -> right leg
    [0,      1,  4,  7, 10],  # root -> left leg
    [0,  3,  6,  9, 12, 15],  # root -> head
    [9,     14, 17, 19, 21],  # head -> R_Wrist
    [9,     13, 16, 18, 20],  # head -> L_Wrist
]

# r_hip, l_hip, r_shoulder, l_shoulder
smpl_22_face_joint_indx = [2, 1, 17, 16]

#                   root, LFoot, RFoot, head, L_Wrist, R_Wrist, Spine3
smpl_key_joint_index = [0,   10,    11,   15,      20,      21,      9]

smpl_ee_idx = [11, 10, 15, 21, 20]

# ================================== SMAL 35 ==================================
# parents | kinematic_chain | face_joint_index | key_joint_index
# ee_idx  |
smal_35_parents = [
    -1,  0,  1,  2,  3,
    4,   5,  6,  7,  8,
    9,   6, 11, 12, 13,
    6,  15,  0, 17, 18,
    19,  0, 21, 22, 23,
    0,  25, 26, 27, 28,
    29, 30, 16, 16, 16
]

smal_35_kinematic_chain = [
    [0, 1, 2, 3, 4, 5, 6, 15, 16, 32],  # root -> mouth
    [6, 7, 8, 9, 10],  # spine3 -> LFoot
    [6, 11, 12, 13, 14],  # spine3 -> RFoot
    [0, 17, 18, 19, 20],  # root -> LFootBack
    [0, 21, 22, 23, 24],  # root -> RFootBack
    [0, 25, 26, 27, 28, 29, 30, 31],  # root -> Tail
    [16, 33],  # head -> LEar
    [16, 34],  # head -> REar
]

# RLeg1, LLeg1, RLegBack1, LLegBack1
smal_35_face_joint_indx = [11, 7, 21, 17]

#                    root, LFootBack, RFootBack, head, LFoot, RFoot, Spine3
smal_key_joint_index = [0,        20,        24,   16,    10,     14,     6]

smal_ee_idx = [24, 20, 16, 14, 10]

SMAL_NAME2ID35 = {
    'root':       0,
    'pelvis0':    1,
    'spine':      2,
    'spine0':     3,
    'spine1':     4,
    'spine2':     5,
    'spine3':     6,
    'LLeg1':      7,
    'LLeg2':      8,
    'LLeg3':      9,
    'LFoot':     10,
    'RLeg1':     11,
    'RLeg2':     12,
    'RLeg3':     13,
    'RFoot':     14,
    'Neck':      15,
    'Head':      16,
    'LLegBack1': 17,
    'LLegBack2': 18,
    'LLegBack3': 19,
    'LFootBack': 20,
    'RLegBack1': 21,
    'RLegBack2': 22,
    'RLegBack3': 23,
    'RFootBack': 24,
    'Tail1':     25,
    'Tail2':     26,
    'Tail3':     27,
    'Tail4':     28,
    'Tail5':     29,
    'Tail6':     30,
    'Tail7':     31,
    'Mouth':     32,
    'LEar':      33,
    'REar':      34,
}

smal_symmetry_joint_index = [
    0, 1, 2, 3, 4, 5, 6,
    11, 12, 13, 14,  # right leg
    7, 8, 9, 10,  # left leg
    15, 16,
    21, 22, 23, 24,  # right leg back
    17, 18, 19, 20,  # left leg back
    25, 26, 27, 28, 29, 30, 31,  # tail
    32, 34, 33
]


# =================================== DME 28 ==================================
# deep motion editing with 28 joints
dme_28_kinematic_chain = [
    [0,           1,  2,  3,  4,  5],  # root -> left leg
    [0,           6,  7,  8,  9, 10],  # root -> right leg
    [0,  11, 12, 13, 14, 15, 16, 17],  # root -> head
    [14,         18, 19, 20, 21, 22],  # head -> left arm
    [14,         23, 24, 25, 26, 27],  # head -> right arm
]

dme_28_parents = [
    # 0  1   2   3   4   5
    -1,  0,  1,  2,  3,  4,
    # 6   7   8   9   10
    0,  6,  7,  8,  9,
    # 11  12  13  14  15  16  17
    0, 11, 12, 13, 14, 15, 16,
    # 18  19  20  21  22
    14, 18, 19, 20, 21,
    # 23  24  25  26  27
    14, 23, 24, 25, 26,
]
# root, left foot, right foot, head, left hand, right hand
dme_28_key_joint_index = [0, 5, 10, 17, 22, 27]
# enc effector index
dme_28_ee_idx = [5, 10, 17, 22, 27]

# right hip, left hip, right shoulder, left shoulder
dme_28_face_joint_index = [6, 1, 23, 18]


# =================================== DME 22 ==================================
# deep motion editing with 22 joints
dme_22_kinematic_chain = [
    [0,       1,  2,  3,  4],  # root -> left leg
    [0,       5,  6,  7,  8],  # root -> right leg
    [0,   9, 10, 11, 12, 13],  # root -> head
    [11,     14, 15, 16, 17],  # head -> left arm
    [11,     18, 19, 20, 21],  # head -> right arm
]

# right hip, left hip, right shoulder, left shoulder
dme_22_face_joint_index = [5, 1, 18, 14]

# dme 22 parents
dme_22_parents = [
    -1,  0,  1,  2,  3,
    0,   5,  6,  7,  0,
    9,  10, 11, 12, 11,
    14, 15, 16, 11, 18,
    19, 20,
]
# root, left foot, right foot, head, left hand, right hand,
dme_22_key_joint_index = [0, 4, 8, 13, 17, 21]
# enc effector index
dme_22_ee_idx = [4, 8, 13, 17, 21]
