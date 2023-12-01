'''
dataset for baseline training and evalutaion.

author
    zhangsihao yang

logs
    2023-11-22
        file created
'''
import json
from os.path import join as pjoin

import clip
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

CAPTION = '../../_runtime/baselines/captions/ood'
OFFSET = '../../_data/DeformingThings4D/animals_maa_offsets'


class MotionText(data.Dataset):
    '''
    the dataset for training Motion and Text pair
    '''

    def _process_motion(self, motion):
        '''
        process the motion for the motion generated as 2-d array
        '''
        if len(motion.shape) == 3 and motion.shape[0] == 1:
            motion = motion[0]

        if len(motion.shape) == 2:
            input_dim = 36 * 6
            motion = motion[:, :input_dim]
            motion = motion.reshape(-1, 36, 6)

        if motion.shape[0] < 196:
            motion = np.concatenate(
                [motion, np.zeros([196 - motion.shape[0], 36, 6])],
                axis=0
            )

        return motion

    def __init__(self, list_motion_path, args) -> None:
        super().__init__()

        data_debug = bool(args.data_debug)
        if data_debug and len(list_motion_path) > 300:
            list_motion_path = list_motion_path[:300]

        self.data = []
        for motion_path in tqdm(
            list_motion_path,
            desc='loading motion text dataset',
        ):
            # load the motion
            motion = np.load(motion_path)
            motion = self._process_motion(motion)

            # get the motion id
            motion_id = motion_path.split('/')[-2]

            # load the meta information
            caption_path = pjoin(
                CAPTION, motion_id, 'meta.json'
            )

            with open(
                caption_path, 'r', encoding='utf-8'
            ) as openfile:
                meta = json.load(openfile)

            smal_id = meta['smal_id'].split('/')[-1].split('_')[0]

            # load the offset
            offset_path = pjoin(
                OFFSET, smal_id, 'offset.pkl'
            )
            offset = np.load(offset_path, allow_pickle=True)

            # load the caption
            caption = meta['smal_text']

            self.data.append(
                {
                    'motion': motion,
                    'caption': caption,
                    'offset': offset,
                }
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    @staticmethod
    def _tokenize_text(raw_text):
        '''
        encode text with clip.

        inputs:
        -------
        raw_text
            list (batch_size length) of strings with input text prompts

        return:
        -------
        tensor : [batch_size, 512]
            the clip text feature
        '''
        # Specific hardcoding for humanml dataset
        max_text_len = 20

        default_context_length = 77
        context_length = max_text_len + 2  # start_token + 20 + end_token
        assert context_length < default_context_length

        # [bs, context_length] # if n_tokens > context_length -> will truncate
        texts = clip.tokenize(
            raw_text, context_length=context_length, truncate=True
        )

        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype
        )
        texts = torch.cat([texts, zero_pad], dim=1)
        return texts

    @classmethod
    def collate(cls, batch):
        '''
        collate function for the dataset
        '''
        list_motions = [torch.from_numpy(item['motion']) for item in batch]
        motions = torch.stack(list_motions, dim=0).float()

        list_text = [item['caption'][0] for item in batch]
        clip_token = cls._tokenize_text(list_text)

        list_offsets = [torch.from_numpy(item['offset']) for item in batch]
        offsets = torch.stack(list_offsets, dim=0)

        return {
            'motions': motions,
            'clip_token': clip_token,
            'offsets': offsets,
        }
