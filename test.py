import torch
import torch.multiprocessing as mp
from model.resnet import  WeightParallelConv2d, WidthParallelConv2d
from model.initialize import initialize_model_parallel
import torch.distributed as dist
import random
import numpy as np
from model.random import model_parallel_cuda_manual_seed

def test_conv(rank, ngpus_per_node, serial_conv, input_data):

    if rank == 0:
        torch.set_printoptions(profile="full")
        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True) 
    

    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:6006", world_size=ngpus_per_node, rank=rank)
    initialize_model_parallel()

    _set_random_seed(26)

    #for p in parallel_conv.parameters():
        #print("[ rank {} ] parallel weight : ".format(str(rank)), p)
    torch.cuda.set_device(rank)
    torch.cuda.synchronize()
    input_ = input_data.cuda(rank)

    parallel_conv = WidthParallelConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
    parallel_conv.weight = serial_conv.weight
    parallel_conv_ = parallel_conv.cuda(rank)

    serial_result_time = 0
    parallel_result_time = 0


    if rank == 0:
        start.record()
        serial_conv_ = serial_conv.cuda(rank)
        serial_result = serial_conv_(input_)
        end.record()
        serial_result_time = start.elapsed_time(end)
 
        
        #for p in serial_conv.parameters():
            #print("serial weight :",  p)
        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True) 
        
        start.record()
    torch.cuda.synchronize()
    parallel_result = parallel_conv_(input_)
    torch.cuda.synchronize()
    if rank == 0:
        end.record()

        parallel_result_time = start.elapsed_time(end)

    if rank == 0:
        if not __debug__:
            print("[ Master Rank  : {} ]".format(str(torch.cuda.current_device())))
            print("[ Master Rank ] Parallel Result Shape : ", parallel_result.size())
            print("[ Master Rank ] Parallel : ", parallel_result[0][0][0])
            print("[ Master Rank ] Serial Result Shape : ", serial_result.size())
            print("[ Master Rank ] Serial : ", serial_result[0][0][0])

        # parallel_result and serial_result should be the same
        assert torch.allclose(parallel_result, serial_result), "Parallel and Serial results are not the same"
        print("Parallel and Serial Result are same.")
        print("***Conv layer Test Passed!!***")
        print("Serial elapsed time : {serial_result_time} ms")
        print("Parallel elapsed time : {parallel_result_time} ms")




def main():

    if __debug__:
        print("__SIMPLE DEBUG MODE__")
    else:
        print("__HARD DEBUG MODE__")
    # ??? ????????? GPU?????? ?????????.
    ngpus_per_node = torch.cuda.device_count()
    input_data = torch.randn(64, 3, 224, 224)
    # multiprocessing_distributed ????????? true??????, world_size??? ??? GPU????????? ????????? ??????, ?????? ????????? ?????????.
    serial_conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)


    mp.spawn(test_conv, nprocs=ngpus_per_node, args=(ngpus_per_node, serial_conv,  input_data,))

def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))

if __name__ == '__main__':
    main()