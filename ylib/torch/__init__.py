'''
some functions for the common use of pytorch

author
    zhangsihao yang

logs
    2023-10-24
        file created
'''
import torch


def safe_l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    '''
    safely performs L2 normalization on a tensor at the last dimension

    inputs:
    -------
    tensor (torch.Tensor): 
        The input tensor to be normalized.

    return:
    -------
    torch.Tensor: 
        The L2-normalized tensor. Returns the original tensor if its L2 
        norm is zero.
    '''
    l2_norm = torch.norm(tensor, dim=-1, keepdim=True, p=2)

    # Create a mask for vectors with non-zero L2 norm
    mask = (l2_norm != 0.)

    # Handle division by zero by using masked division
    normalized_tensor = torch.where(mask, tensor / l2_norm, tensor)

    return normalized_tensor
