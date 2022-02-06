import sys

import torch
from torch.distributed import get_world_size, get_rank
from torch.nn.parallel import DistributedDataParallel as torchDDP

from model.initialize import get_tensor_model_parallel_group
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
                                 group=get_tensor_model_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

    return averaged_losses
