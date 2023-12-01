'''
mainly will be the functions for collate data

author
    zhangsihao yang

logs
    2023-10-23
        file created
'''
import torch


def _lengths_to_mask(lengths, max_len):
    '''
    convert lengths to mask
    '''
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def _collate_length(batch, index, max_len):
    '''
    the collate function for a given index
    '''
    lengths = torch.tensor([b[index] for b in batch])
    masks = _lengths_to_mask(lengths, max_len).unsqueeze(-1).unsqueeze(-1)
    return masks


def _collate_index(batch, index):
    '''
    the collate function for a given index
    '''
    list_tensor = [torch.from_numpy(b[index]) for b in batch]
    stack_tensor = torch.stack(list_tensor, dim=0)
    return stack_tensor


def collate_smpl(smpl_batch):
    '''
    the collate function for the smpl motion dataset
    '''
    # smpl_motions : [batch_size, 196, 23, 6]
    smpl_motions = _collate_index(smpl_batch, 2).float()

    # smpl_offsets : [batch_size, 22, 6]
    smpl_offsets = _collate_index(smpl_batch, 5).float()

    # smpl_means : [batch_size, 23, 6]
    smpl_means = _collate_index(smpl_batch, 6).float()

    # smpl_stds : [batch_size, 23, 6]
    smpl_stds = _collate_index(smpl_batch, 7).float()

    smpl_mask = _collate_length(
        smpl_batch, 3, smpl_motions.shape[1]
    ).float()

    smpl_text = [b[0] for b in smpl_batch]
    smpl_token = [b[4] for b in smpl_batch]
    smpl_subjects = [b[9] for b in smpl_batch]

    return {
        'smpl_motions': smpl_motions,
        'smpl_offsets': smpl_offsets,
        'smpl_mask': smpl_mask,
        'smpl_text': smpl_text,
        'smpl_token': smpl_token,
        'smpl_means': smpl_means,
        'smpl_stds': smpl_stds,
        'smpl_subjects': smpl_subjects,
    }, smpl_token


def collate_smal(smal_batch):
    '''
    the collate function for the smal motion dataset
    '''
    # smal_motions : [batch_size, 196, 36, 6]
    smal_motions = _collate_index(smal_batch, 2).float()

    # smal_offsets : [batch_size, 35, 3]
    smal_offsets = _collate_index(smal_batch, 5).float()

    # smal_means : [batch_size, 36, 6]
    smal_means = _collate_index(smal_batch, 6).float()

    # smal_stds : [batch_size, 36, 6]
    smal_stds = _collate_index(smal_batch, 7).float()

    smal_mask = _collate_length(
        smal_batch, 3, smal_motions.shape[1]
    ).float()

    smal_text = [b[0] for b in smal_batch]
    smal_token = [b[4] for b in smal_batch]

    return {
        'smal_motions': smal_motions,
        'smal_offsets': smal_offsets,
        'smal_mask': smal_mask,
        'smal_text': smal_text,
        'smal_token': smal_token,
        'smal_means': smal_means,
        'smal_stds': smal_stds,
    }, smal_token
