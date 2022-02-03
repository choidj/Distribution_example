import sys

import torch
from torch.distributed import get_world_size, get_rank
from torch.nn.parallel import DistributedDataParallel as torchDDP

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
                                kernel_size, conv,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    print(last_dim_size)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    
    if conv:
        tensor_custom_split = [torch.cat([tensor_list[i], tensor[:, :, :, ((i+1)*last_dim_size):((i+1)*last_dim_size)+kernel_size-1]], dim=last_dim) for i in range(num_partitions-1)]
        tensor_list = tensor_custom_split


    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list