'''
provide dataloaders

author
    zhangsihao yang

logs
    2023-10-20
        add get_dataloader() function
    2023-10-23
        add get_smpl_dataloader() function
    2023-11-22
        add get_five_fold_dataloader() function
'''
import json
import os
from glob import glob

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from ylib.deforming_things_4d.prior import smal_animal_names_v1_train
from ylib.torch.motion_datasets.baselines import MotionText

from .combine import CombineDataset
from .datasets import Text2MotionDataset
from .eval import EvaluationDataset  # ========================================
from .eval import EvaluationIDDataset  # ======================================
from .eval import EvaluationMMDataset  # ======================================
from .smal import Text2Motion

SMPL_MAA = '/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/_data/MDM/maa/'
SMPL_MAA_META = '/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/_data/MDM/maa_meta/'
SMAL_MAA = '/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/_data/DeformingThings4D/animals_maa_motions/'
MDM_OOD_MM = '/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/_runtime/baselines/motion_diffusion_model/ood/mm'
MDM_ID = '/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/_runtime/baselines/motion_diffusion_model/id'


def _process_list_filepath(list_filepath, process: bool):
    '''
    process the list of file path. this function is for quick debug propose 
    such that we do not need to load the entire dataset.
    '''
    if process:
        if len(list_filepath) > 300:
            list_filepath = list_filepath[:300]
    return list_filepath


def _determine_num_workers(args) -> int:
    '''
    determine the number of workers used in dataloader
    '''
    if args.data_debug:
        num_workers = 0
    else:
        num_workers = os.cpu_count()
        assert num_workers is not None
    return num_workers


def get_dataloader(args):
    '''
    get dataloader from args
    '''

    # SMPL
    # get the list of motion data
    list_smpl_maa = glob(os.path.join(SMPL_MAA, '*.npy'))
    list_smpl_maa = _process_list_filepath(
        list_smpl_maa, bool(args.data_debug)
    )
    t2m_dataset = Text2MotionDataset(
        list_filepath=list_smpl_maa
    )

    # SMAL
    list_smal_maa = []
    for animal_name in smal_animal_names_v1_train:
        list_smal_maa.extend(
            glob(
                os.path.join(SMAL_MAA, f'{animal_name}_*.npy')
            )
        )
    list_smal_maa = _process_list_filepath(
        list_smal_maa, bool(args.data_debug)
    )
    smal_t2m_dataset = Text2Motion(list_filepath=list_smal_maa)

    # combine the dataset together
    combine_dataset = CombineDataset(
        smpl_t2m_dataset=t2m_dataset,
        smal_t2m_dataset=smal_t2m_dataset
    )

    loader = DataLoader(
        combine_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True, collate_fn=combine_dataset.collate
    )

    return loader


def get_mdm_eval_dataloader(args):
    '''
    get evaluation dataloader for mdm model.
    '''
    if args.loader_type == 'ood':
        dataset = EvaluationDataset()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=_determine_num_workers(args),
            drop_last=True,
            collate_fn=dataset.collate
        )
    elif args.loader_type == 'ood_mm':
        dataset = EvaluationMMDataset(
            save_path_root=MDM_OOD_MM,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=_determine_num_workers(args),
            drop_last=True,
            collate_fn=dataset.collate
        )
    elif args.loader_type == 'id':
        dataset = EvaluationIDDataset(
            save_path_root=MDM_ID,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=_determine_num_workers(args),
            drop_last=True,
            collate_fn=dataset.collate
        )
    else:
        raise NotImplementedError(
            f'loader type {args.loader_type} not implemented'
        )
    return loader


def get_smpl_dataloader(args):
    '''
    get dataloader for smpl motion and caption dataset.
    '''
    # SMPL
    # get the list of motion data
    list_smpl_maa = glob(os.path.join(SMPL_MAA, '*.npy'))
    list_smpl_maa = _process_list_filepath(
        list_smpl_maa, bool(args.data_debug)
    )
    t2m_dataset = Text2MotionDataset(
        list_filepath=list_smpl_maa
    )

    loader = DataLoader(
        t2m_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True, collate_fn=t2m_dataset.collate
    )

    return loader


def get_smal_dataloader(args):
    '''
    get dataloader for smal motion and caption dataset.
    '''
    # SMPL
    # get the list of motion data

    list_smal_maa = []
    for animal_name in smal_animal_names_v1_train:
        list_smal_maa.extend(
            glob(
                os.path.join(SMAL_MAA, f'{animal_name}_*.npy')
            )
        )

    list_smal_maa = _process_list_filepath(
        list_smal_maa, bool(args.data_debug)
    )
    smal_t2m_dataset = Text2Motion(list_filepath=list_smal_maa)

    loader = DataLoader(
        smal_t2m_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True, collate_fn=smal_t2m_dataset.collate
    )

    return loader


def _get_split(all_keys, fold, cross_vali_num=5, seed=19941106):
    '''
    get the split for cross-validation.
    '''
    # Convert list to NumPy array for processing
    all_keys = np.array(all_keys)

    # Initialize train_keys and test_keys to avoid unbound errors
    train_keys, test_keys = None, None

    # Create KFold splits
    kf = KFold(n_splits=cross_vali_num, shuffle=True, random_state=seed)
    splits = kf.split(all_keys)

    # Iterate through splits and break when the desired fold is reached
    for i, (train_idx, test_idx) in enumerate(splits):
        if i == fold:
            train_keys = all_keys[train_idx].tolist()
            test_keys = all_keys[test_idx].tolist()
            break

    # Handle the case where the fold is not found
    if train_keys is None or test_keys is None:
        raise ValueError("Fold number out of range")

    return train_keys, test_keys


def _list_loader(list_motion_path, args):
    '''
    create data loader from list of motion paths.
    '''
    dataset = MotionText(list_motion_path, args)
    # create data loader
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=_determine_num_workers(args),
        drop_last=True,
        collate_fn=dataset.collate
    )

    return loader


def _get_list_generated_motion_path(baseline_folder):
    '''
    get the list of generated motion paths from a baseline folder.
    '''
    baseline_meta_path = baseline_folder + '.json'
    assert os.path.exists(baseline_meta_path)
    with open(baseline_meta_path, 'r') as openfile:
        list_generated_motion_path = json.load(openfile)
    return list_generated_motion_path


def get_five_fold_dataloader(args):
    '''
    five-fold dataloader for training only on generated motions from baselines.
    '''
    # get the list of generated motions
    train_data_meta_path = args.train_data_path + '.json'
    assert os.path.exists(train_data_meta_path)
    with open(train_data_meta_path, 'r') as openfile:
        list_generated_motion_path = json.load(openfile)
    train_list, val_list = _get_split(list_generated_motion_path, args.fold)

    # get data loaders from list of motion paths
    train_loader = _list_loader(train_list, args)

    val_loader = _list_loader(val_list, args)

    # get other loaders
    assert len(args.list_test_data_path) == 4

    list_baseline_0 = _get_list_generated_motion_path(
        args.list_test_data_path[0]
    )
    list_baseline_1 = _get_list_generated_motion_path(
        args.list_test_data_path[1]
    )
    list_baseline_2 = _get_list_generated_motion_path(
        args.list_test_data_path[2]
    )
    list_baseline_3 = _get_list_generated_motion_path(
        args.list_test_data_path[3]
    )

    baseline_0_loader = _list_loader(list_baseline_0, args)
    baseline_1_loader = _list_loader(list_baseline_1, args)
    baseline_2_loader = _list_loader(list_baseline_2, args)
    baseline_3_loader = _list_loader(list_baseline_3, args)

    return {
        'train': train_loader,
        'val': val_loader,
        'baseline_0': baseline_0_loader,
        'baseline_1': baseline_1_loader,
        'baseline_2': baseline_2_loader,
        'baseline_3': baseline_3_loader,
    }
