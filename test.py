import torch
import torch.multiprocessing as mp
from model.resnet import ParallelConv2d
from model.initialize import initialize_model_parallel
import torch.distributed as dist



def test_conv(rank, ngpus_per_node, serial_conv, parallel_conv, input_data):
    if rank == 0:
        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True) 
    
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:6006", world_size=ngpus_per_node, rank=rank)
    initialize_model_parallel()
    #for p in parallel_conv.parameters():
        #print("[ rank {} ] parallel weight : ".format(str(rank)), p)
    torch.cuda.set_device(rank)
    input_ = input_data.cuda(rank)
    parallel_conv_ = parallel_conv.cuda(rank)
    if rank == 0:
        start.record()
        serial_conv_ = serial_conv.cuda(rank)
        serial_result = serial_conv_(input_)
        end.record()
        print(start.elapsed_time(end))

        #for p in serial_conv.parameters():
            #print("serial weight :",  p)
   
        start.record()
    parallel_result = parallel_conv_(input_)
    if rank == 0:
        end.record()

        print(start.elapsed_time(end))

    if rank == 0:
        print("[Device : {} ]".format(str(torch.cuda.current_device())))
        print("Parallel Result Shape : ", parallel_result.size())
        print("Serial Result Shape : ", serial_result.size())
        print("Parallel : ", parallel_result)
        print("Serial : ", serial_result)
        # parallel_result and serial_result should be the same
        assert torch.allclose(parallel_result, serial_result), "Parallel and Serial results are not the same"
        print("Conv layer Test Passed!!")




def main():
    # 한 노드의 GPU수를 가져옴.
    ngpus_per_node = torch.cuda.device_count()
    input_data = torch.randn(1, 3, 224, 224)
    # multiprocessing_distributed 변수가 true라면, world_size를 총 GPU개수로 설정한 후에, 메인 워커를 실행함.
    serial_conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
    parallel_conv = ParallelConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
    parallel_conv.weight = serial_conv.weight

    print("Are they same? : ", torch.allclose(serial_conv.weight, parallel_conv.weight))

    mp.spawn(test_conv, nprocs=ngpus_per_node, args=(ngpus_per_node, serial_conv, parallel_conv, input_data,))



if __name__ == '__main__':
    main()
