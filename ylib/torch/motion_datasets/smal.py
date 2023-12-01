'''
the text dataset for SMAL joint motion dataset

author
    zhangsihao yang

logs
    2023-10-09
        file created
    2023-10-19
        create another class for strong text to motion dataset
'''
import codecs as cs
import os
import random
from os.path import join as pjoin
from typing import Dict, List

import clip
import numpy as np
# import spacy
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

# from data_loaders.humanml.utils.get_opt import get_opt
# from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from .utils import collate_smal


def _is_valid_motion_length(motion, min_len, max_len):
    """Check if the motion's length is within the valid range."""
    return min_len <= len(motion) < max_len


class WeakText2Motion(data.Dataset):
    '''
    the text is weak because the text is only a very short file name.

    '''

    def __init__(self, list_filepath) -> None:
        super().__init__()

        self.max_motion_length = 196
        self.min_motion_len = 20
        self.unit_length = 4

        self.data_dict = {}
        self.name_list = []
        self.length_list = []

        for filepath in tqdm(list_filepath):
            # motion : (num_frames, 36, 6)
            motion = np.load(filepath)

            if not _is_valid_motion_length(
                motion, self.min_motion_len, self.max_motion_length
            ):
                continue

            # load text
            caption = self._get_caption_from_filepath(filepath)

            # load mean
            mean_path = pjoin(
                self._get_meta_dir(filepath),
                self._get_animal_name(filepath),
                'mean.npy',
            )
            mean = np.load(mean_path)

            # load std
            std_path = pjoin(
                self._get_meta_dir(filepath),
                self._get_animal_name(filepath), 'std.npy'
            )
            std = np.load(std_path)

            # load offset
            offset_path = pjoin(
                self._get_meta_dir(filepath),
                self._get_animal_name(filepath), 'offset.pkl'
            )
            offset = np.load(offset_path, allow_pickle=True)

            self.name_list.append(filepath)
            self.length_list.append(len(motion))
            self.data_dict[filepath] = {
                'motion': motion,
                'length': len(motion),
                'text': caption,
                'mean': mean,
                'std': std,
                'offset': offset,
            }

    def _get_caption_from_filepath(self, filepath):
        '''
        get the caption from the file name
        '''
        with_us = filepath.split('/')[-1].split('.')[0]
        without_us = with_us.replace('_', ' ')
        return without_us

    def _get_animal_name(self, filepath):
        '''
        get the animal name from the file path
        '''
        return filepath.split('/')[-1].split('.')[0].split('_')[0]

    def _get_meta_dir(self, filepath):
        '''
        get the meta dir from the file path
        '''
        split_path = filepath.split('/')
        split_path[-2] = 'animals_maa_offsets'
        return '/'.join(split_path[:-1])

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        data = self.data_dict[self.name_list[index]]

        m_length = data['length']
        motion = data['motion']
        mean = data['mean']
        std = data['std']
        offset = data['offset']
        sent_len = len(data['text'].split(' '))

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        num_chunck = m_length // self.unit_length
        if coin2 == 'double':
            m_length = (num_chunck - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = num_chunck * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        # Z Normalization
        # mask std
        masked_std = std.copy()
        masked_std[masked_std == 0] = 1
        motion = (motion - mean) / masked_std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros(
                        (
                            self.max_motion_length - m_length,
                            motion.shape[1],
                            motion.shape[2],
                        )
                    )
                ],
                axis=0
            )

        return data['text'], sent_len, motion, m_length, '_'.join(data['text']), \
            offset, mean, std, self.name_list[index]


def _convert_path(original_path):
    # Split the path to isolate directory and filename without extension
    base_dir, filename = os.path.split(original_path)
    filename_without_ext = os.path.splitext(filename)[0]

    # Replace 'animals_maa_motions' with 'animals'
    new_dir = base_dir.replace('animals_maa_motions', 'animals')

    # Construct the new path
    new_path = os.path.join(new_dir, f"{filename_without_ext}/screenshots.txt")

    return new_path


def _check_empty_file(file_path):
    '''
    check if the file is empty.

    args
        file_path: str, the path of the file.

    returns
        bool, True if the file is empty.
    '''
    with open(file_path, 'r') as f:
        content = f.read()
    if content == '':
        return True
    return False


class Text2Motion(WeakText2Motion):
    '''
    the text is strong because the text is a sentence this time.
    '''

    def __init__(self, list_filepath) -> None:
        super().__init__(list_filepath)

        self.name_list = []

        keys_to_remove = []

        for filepath in tqdm(self.data_dict, desc='Loading text data'):
            # load caption and modify data_dict[filepath]['text']
            text_path = _convert_path(filepath)
            if not os.path.exists(text_path):
                # self.data_dict[filepath] = None
                keys_to_remove.append(filepath)
                continue

            # skip if the file is empty
            if _check_empty_file(text_path):
                # self.data_dict[filepath] = None
                keys_to_remove.append(filepath)
                continue

            text_data = []
            open_file = cs.open(text_path)

            for line in open_file.readlines():
                if line.strip() == '':
                    continue
                line_split = line.strip().split('#')
                caption = line_split[0]
                text_entry = {'caption': caption}
                text_data.append(text_entry)

            open_file.close()

            self.data_dict[filepath]['text'] = text_data

            self.name_list.append(filepath)

        # remove all the dict keys collected
        for key in keys_to_remove:
            self.data_dict.pop(key, None)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        data_pool = self.data_dict[self.name_list[index]]

        m_length = data_pool['length']
        motion = data_pool['motion']
        mean = data_pool['mean']
        std = data_pool['std']
        offset = data_pool['offset']

        text_list = data_pool['text']
        text_data = random.choice(text_list)
        caption = text_data['caption']

        sent_len = len(caption.split(' ')) + 2

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        num_chunck = m_length // self.unit_length
        if coin2 == 'double':
            m_length = (num_chunck - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = num_chunck * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        # Z Normalization
        # mask std
        masked_std = std.copy()
        masked_std[masked_std == 0] = 1
        motion = (motion - mean) / masked_std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros(
                        (
                            self.max_motion_length - m_length,
                            motion.shape[1],
                            motion.shape[2],
                        )
                    )
                ],
                axis=0
            )

        return caption, sent_len, motion, m_length, '_'.join(caption), \
            offset, mean, std, self.name_list[index]

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
        the collate function for the human motion dataset.
        # TODO rename varible names in this function.
        '''
        smpl_meta = {
            'mean': [b[6] for b in batch],
            'std':  [b[7] for b in batch],
            'name': [b[8] for b in batch],
        }

        real_smpl_input, _ = collate_smal(batch)

        # tokenlize the text data
        real_smpl_input['token_text'] = cls._tokenize_text(
            real_smpl_input['smal_text']
        )

        return {
            'real_smpl_input': real_smpl_input,
            'smpl_meta': smpl_meta,
        }

    @classmethod
    def _flatten_motion(cls, motion):
        '''
        flatten the motion from [seqlen, nj, 6] to [seqlen, nj*6]
        '''
        return motion.reshape(motion.shape[0], -1)

    @classmethod
    def mdm_collate(cls, batch):
        '''
        TODO to be finished.
        '''
        adapted_batch = [
            {
                # [seqlen, J] -> [J, 1, seqlen]
                'inp': torch.tensor(
                    cls._flatten_motion(b[2]).T
                ).float().unsqueeze(1),
                'text': b[0],  # b[0]['caption']
                'tokens': b[4],
                'lengths': b[3],
            }
            for b in batch
        ]

        return collate(adapted_batch)


def collate(batch):
    '''
    collate function from mdm.

    return:
    -------
    motion : [batch_size, feature_dim, 1, seq_len]
        the collated motions
    '''
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(
        lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)  # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond


# an adapter to our collate func


def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        # [seqlen, J] -> [J, 1, seqlen]
        'inp': torch.tensor(b[4].T).float().unsqueeze(1),
        'text': b[2],  # b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas
