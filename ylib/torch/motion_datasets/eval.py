'''
dataset for evaluation on caption only input.

author:
    zhangsihao yang

logs
    2023-11-13
        file created
'''
import json
from glob import glob
from os.path import join as pjoin

from torch.utils import data
from tqdm import tqdm

CAPTION = '/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/_runtime/baselines/captions/ood.json'
ID_CAPTION = '/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/_runtime/baselines/captions/id'


class EvaluationDataset(data.Dataset):
    '''
    evaluation dataset for motion generated and caption
    '''

    def _load_caption(self, caption_file_path):
        '''
        load caption from the caption file path.
        '''
        with open(caption_file_path, 'r', encoding='utf-8') as openfile:
            meta = json.load(openfile)
        caption = meta['smal_text']
        return caption

    def __init__(self) -> None:
        '''
        init function with ood caption loaded
        '''
        # load the path of meta files
        with open(CAPTION, 'r', encoding='utf-8') as openfile:
            caption_path = json.load(openfile)

        self.data = []
        for caption_file_path in tqdm(
            caption_path, desc='loading ood captions',
        ):
            caption = self._load_caption(caption_file_path)
            self.data.append(
                {
                    'caption': caption,
                    'motion_id': caption_file_path.split('/')[-2],
                }
            )

    def __len__(self) -> int:
        '''
        get the length of the dataset
        '''
        return len(self.data)

    def __getitem__(self, index: int):
        '''
        get the item from the dataset
        '''
        return self.data[index]

    @classmethod
    def collate(cls, batch):
        '''
        collate function for the dataset
        '''
        list_catption = [item['caption'][0] for item in batch]
        list_motion_id = [item['motion_id'] for item in batch]
        return {
            'list_catption': list_catption,
            'list_motion_id': list_motion_id,
        }


class EvaluationMMDataset(EvaluationDataset):
    '''
    the evaluation dataset for ood mm motion generation
    '''

    def _append(self, caption_file_path, save_path_root):
        '''
        append caption, motion_id, and save_path 20 times to the dataset
        '''
        caption = self._load_caption(caption_file_path)
        motion_id = caption_file_path.split('/')[-2]
        for repeat_time in range(20):  # 20 is hard coded here
            save_path = pjoin(
                save_path_root, motion_id, f'motion{repeat_time:02d}.npy'
            )
            self.data.append(
                {
                    'caption': caption,
                    'motion_id': motion_id,
                    'save_path': save_path,
                }
            )

    def __init__(self, save_path_root: str) -> None:
        '''
        init function load the repeat captions
        '''
        with open(CAPTION, 'r', encoding='utf-8') as openfile:
            caption_path = json.load(openfile)

        self.data = []
        for i, caption_file_path in tqdm(
            enumerate(caption_path), desc='loading ood captions',
        ):
            if i % 10 != 0:
                continue
            self._append(caption_file_path, save_path_root)

    @classmethod
    def collate(cls, batch):
        '''
        collate function for the dataset
        '''
        list_caption = [item['caption'][0] for item in batch]
        list_motion_id = [item['motion_id'] for item in batch]
        list_save_path = [item['save_path'] for item in batch]
        return {
            'list_caption': list_caption,
            'list_motion_id': list_motion_id,
            'list_save_path': list_save_path,
        }


class EvaluationIDDataset(EvaluationDataset):
    '''
    evaluation dataset for in-distribution motion generation with mm in-built
    '''

    def _append(self, caption_file_path, save_path_root):
        '''
        append caption, motion_id, and save_path 20 times to the dataset
        '''
        caption = self._load_caption(caption_file_path)
        motion_id = caption_file_path.split('/')[-2]
        for repeat_time in range(20):  # 20 is hard coded here
            save_path = pjoin(
                save_path_root, motion_id, f'motion{repeat_time:02d}.npy'
            )
            self.data.append(
                {
                    'caption': caption,
                    'motion_id': motion_id,
                    'save_path': save_path,
                }
            )

    def __init__(self, save_path_root):
        '''
        the init function to init self.data
        '''
        caption_path = glob(pjoin(ID_CAPTION, '*/meta.json'))
        caption_path.sort()

        self.data = []

        for caption_file_path in tqdm(
            caption_path, desc='loading ood captions',
        ):
            self._append(caption_file_path, save_path_root)

    @classmethod
    def collate(cls, batch):
        '''
        collate function for the dataset
        '''
        list_caption = [item['caption'][0] for item in batch]
        list_motion_id = [item['motion_id'] for item in batch]
        list_save_path = [item['save_path'] for item in batch]
        return {
            'list_caption': list_caption,
            'list_motion_id': list_motion_id,
            'list_save_path': list_save_path,
        }
