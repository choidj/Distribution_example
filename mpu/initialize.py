# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model and data parallel groups."""

import torch

from .utils import ensure_divisibility

_TENSOR_MODEL_PARALLEL_GROUP = None


def initialize_model_parallel():
    """
    Initialize model data parallel groups.
    
    Arguments:
        self.tensor_model_parallel_size: number of GPUs used for tensor model parallelism. Now we only support 4 GPUs.
        self.pipeline_model_parallel_size: number of GPUs used for pipeline model parallelism. Now we don't support pipeline model parallelism.
    """
    if torch.distributed.get_rank() == 0:
        print('> initializing parallel work')

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = world_size
    rank = torch.distributed.get_rank()

    # Build the tensor model parallel groups.
    # we are only using one tensor model parallel group for now
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized.'
    ranks = range(tensor_model_parallel_size)
    group = torch.distributed.new_group(ranks)

    if rank in ranks:
        _TENSOR_MODEL_PARALLEL_GROUP = group
        

def parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        return False
    return True



def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size



def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())



def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank




def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())



def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None