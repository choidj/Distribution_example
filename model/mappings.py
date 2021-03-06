import torch
from .initialize import get_tensor_model_parallel_rank, get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from .utils import split_tensor_along_last_dim, divide

def _split(input_, kernel_size=0, padding=0, conv=False, ver="width"):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    world_size = torch.distributed.get_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_
    rank = get_tensor_model_parallel_rank()
    if not __debug__:
        if rank == 0 and conv:
            print("[Master GPU] **TO SPLIT** Input Size : {}, Input : ".format(str(input_.size()), input_[0][0][padding[0]]))
        elif rank == 0 and not conv:
            print("[Master GPU] **TO SPLIT** Input Size : {}, Input : ".format(str(input_.size()), input_))

    input_list = split_tensor_along_last_dim(input_, world_size, kernel_size, padding, conv, ver)
    
    if not __debug__:
        if rank == 0 and conv:
            for i in range(world_size):
                print("[Master GPU] **TO SPLIT** [ {} ] Output : ".format(str(i)), input_list[i][0][0][padding[0]])
        elif rank == 0 and not conv:
            for i in range(world_size):
                print("[Master GPU] **TO SPLIT** [ {} ] Output : ".format(str(i)), input_list[i])

    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[rank].contiguous() # 새로운 주소로 할당함.
    if not __debug__:
        print("[Rank {} GPU] **SPLITED** Output Size : {}".format(str(rank), str(output.size())))

    return output


def _conv_split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    world_size = torch.distributed.get_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_
    rank = get_tensor_model_parallel_rank()
    num_partitions = get_tensor_model_parallel_world_size()

    # Get the size and dimension.
    channel_dim = 1
    
    channel_dim_size = divide(input_.size()[channel_dim], num_partitions)
    
    # Split.
    tensor_list = torch.split(input_, channel_dim_size, dim=channel_dim)


    # Note: torch.split does not create contiguous tensors by default.

    input_list = tensor_list


    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[rank].contiguous() # 새로운 주소로 할당함.


    return output



def _linear_split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    rank = get_tensor_model_parallel_rank()
    num_partitions = get_tensor_model_parallel_world_size()

    # Get the size and dimension.
    last_dim = 1
    
    last_dim_size = divide(input_.size()[last_dim], num_partitions)
    
    # Split.
    tensor_list = torch.split(input_,last_dim_size, dim=last_dim)


    # Note: torch.split does not create contiguous tensors by default.

    input_list = tensor_list


    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[rank].contiguous() # 새로운 주소로 할당함.

    return output



def _gather(input_, kernel_size=0, padding=0, conv=False, ver="weight"):
    """Gather tensors and concatinate along the last dimension."""
    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()
    
    if not __debug__:
        if rank == 0 and conv:
            print("[Rank {} GPU] **TO GATHER** Input Size -> ".format(str(rank)), input_.size())
            print("[Rank {} GPU] **TO GATHER** Input (0, 0, 0, ) -> ".format(str(rank)), input_[0][0][padding[0]])
        elif rank == 0 and not conv:
            print("[Rank {} GPU] **TO GATHER** Input Size -> ".format(str(rank)), input_.size())

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_


    
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())


    # Note: torch.cat already creates a contiguous tensor.
    if ver == "weight":
        output = torch.cat(tensor_list, dim=1).contiguous()
    else:
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
    if not __debug__:
        print("[Rank {} GPU] **GATHERED** Output Size : {}".format(str(rank), str(output.size())))
        if rank == 0 and conv:
            print("[Master GPU] **GATHERED** Output (0, 0, 0, ): ", output[0][0][0])
    
    return output


def _conv_gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()
    

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_


    
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())


    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=1).contiguous()
    
    return output


def _linear_gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()
    

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_


    
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())


    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

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


def scatter_to_tensor_model_parallel_region(input_, kernel_size=0, padding=0, conv=False, ver="width"):
    return _ScatterToModelParallelRegion.apply(input_, kernel_size, padding, conv, ver)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_linear_region(input_):
    return _GatherFromModelParallelLinearRegion.apply(input_)


def gather_from_tensor_model_parallel_conv_region(input_):
    return _GatherFromModelParallelConvRegion.apply(input_)


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
    def symbolic(graph, input_, kernel_size, padding, conv, ver):
        return _split(input_, kernel_size, padding, conv, ver)

    @staticmethod
    def forward(ctx, input_, kernel_size, padding, conv, ver):
        return _split(input_, kernel_size, padding, conv, ver)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


class _GatherFromModelParallelLinearRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _linear_gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _linear_gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _linear_split(grad_output)


class _GatherFromModelParallelConvRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _conv_gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _conv_gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_split(grad_output)


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


