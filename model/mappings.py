import torch
from .initialize import get_tensor_model_parallel_rank, get_tensor_model_parallel_group
from .utils import split_tensor_along_last_dim, divide

def _split(input_, kernel_size=0, conv=False):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_
    rank = get_tensor_model_parallel_rank()
    
    print("[Rank {} GPU] **TO SPLIT** Input Size : {}".format(str(rank), str(input_.size())))
    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size, kernel_size, conv)

    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[rank].contiguous() # 새로운 주소로 할당함.
    
    print("[Rank {} GPU] **SPLITED** Output Size : {}".format(str(rank), str(output.size())))
    return output


def _gather(input_, kernel_size=0, conv=False):
    """Gather tensors and concatinate along the last dimension."""
    
    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()
    
    print("[Rank {} GPU] **TO GATHER** Input Size : {}".format(str(rank), str(input_.size())))

    result_kernel_size = divide(kernel_size[0] - 1, 2)

    print("[Rank {} GPU] **TO GATHER** previous Input : Index (0, 0, 0, ) -> ".format(str(rank)), input_[0][0][0])

    if conv and rank != (world_size-1):
        input_ = input_[:, :, :, :-result_kernel_size].contiguous()
        print("[Rank {} GPU] **TO GATHER** next Input Splited Size : {}".format(str(rank), str(input_.size())))
    
    print("[Rank {} GPU] **TO GATHER** next Input : Index (0, 0, 0, ) -> ".format(str(rank)), input_[0][0][0])
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_

    for i in range(world_size):
        print("[Rank {} GPU] **TO GATHER** Prepared Input List Size[{}] : {}".format(str(rank), str(i), str(tensor_list[i].size()))) 
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    
    print("[Rank {} GPU] **GATHERED** Output Size : {}".format(str(rank), str(output.size())))
    
    return output


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_, kernel_size=0, conv=False):
    return _ScatterToModelParallelRegion.apply(input_, kernel_size, conv)


def gather_from_tensor_model_parallel_region(input_, kernel_size=0, conv=False):
    return _GatherFromModelParallelRegion.apply(input_, kernel_size, conv)


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output



class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, kernel_size, conv):
        return _split(input_, kernel_size, conv)

    @staticmethod
    def forward(ctx, input_, kernel_size, conv):
        return _split(input_, kernel_size, conv)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, kernel_size, conv):
        return _gather(input_, kernel_size, conv)
    
    @staticmethod
    def forward(ctx, input_, kernel_size, conv):
        return _gather(input_, kernel_size, conv)

    @staticmethod
    def backward(ctx, grad_output, kernel_size, conv):
        return _split(grad_output, kernel_size, conv)


class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


