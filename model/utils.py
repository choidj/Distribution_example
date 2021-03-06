import sys

import torch
from torch.distributed import get_world_size, get_rank
from torch.nn.parallel import DistributedDataParallel as torchDDP
import torch.nn.functional as F

from .initialize import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

# 인풋 이미지 feature를 각 gpu의 Convolution layer에서 사용할 수 있도록 맞게 나눈다.
def split_conv_input_tensor_parallel_group(_input, ngpus_per_node,kernel_size):
    """Split the input tensor into ngpus_per_node tensors."""
    local_rank = get_rank() % ngpus_per_node

    sliced_input = torch.chunk(_input,
                               ngpus_per_node,
                               dim=1)
    if local_rank != (ngpus_per_node-1):
        sliced_input = torch.cat([sliced_input[local_rank], sliced_input[local_rank + 1][:, :kernel_size]], dim=1)

    return sliced_input[local_rank]

# 인풋 이미지 feature를 각 gpu의 pooling layer에서 사용할 수 있도록 맞게 나눈다.
def split_pooling_input_tensor_parallel_group(_input, ngpus_per_node, kernel_size):
    """Split the input tensor into ngpus_per_node tensors."""
    local_rank = get_rank() % ngpus_per_node

    sliced_input = torch.chunk(_input,
                               ngpus_per_node,
                               dim=1)

    return sliced_input[local_rank]


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor_along_last_dim(tensor, num_partitions,
                                kernel_size, padding, conv,
                                contiguous_split_chunks=False, ver="width"):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    rank = get_tensor_model_parallel_rank()
    padding_int = 0
    if not isinstance(padding, int):
        padding_int = padding[0]


    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    
    
    if conv and padding_int != 0 and ver == "width":
        tensor_list = list(tensor_list)
        padded_tensor = custom_pad(tensor_list, padding_int, last_dim, num_partitions)
            
        tensor_list[rank] = padded_tensor
        tensor_list = tuple(tensor_list)
    elif ver == "channel":
        last_dim_size = divide(tensor.size()[1], num_partitions)
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if not __debug__:
        print("[Rank {} GPU] Division size : {}, **TO CUSTOM SPLIT** Splited Input Size : ".format(str(rank), str(last_dim_size)), tensor_list[rank].size())
        print("[Rank {} GPU] Padding : {}, **TO CUSTOM SPLIT** Splited Input (0, 0, padding, ) : ".format(str(rank), str(padding_int), str(last_dim_size)), tensor_list[rank][0][0][padding_int])

    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)


    return tensor_list



def custom_pad(tensor, padding, apply_dim, num_partitions):
    rank = get_tensor_model_parallel_rank()
    
    if rank == 0:
        pad_dim = (padding, 0, padding, padding)
        tensor_custom_split = torch.cat([tensor[rank], tensor[rank+1][:, :, :, :padding]], dim=apply_dim)
        return F.pad(tensor_custom_split, pad_dim) 
    elif rank == num_partitions - 1:
        pad_dim = (0, padding, padding, padding)
        tensor_custom_split = torch.cat([tensor[rank-1][:, :, :, -padding:], tensor[rank]], dim=apply_dim)
        return F.pad(tensor_custom_split, pad_dim) 
    else:
        pad_dim = (0, 0, padding, padding)
        tensor_custom_split = torch.cat([tensor[rank-1][:, :, :, -padding:], tensor[rank]], dim=apply_dim)
        tensor_custom_split = torch.cat([tensor_custom_split, tensor[rank+1][:, :, :, :padding]], dim=apply_dim)
        return F.pad(tensor_custom_split, pad_dim)
