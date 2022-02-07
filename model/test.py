import torch
import torch.multiprocessing as mp
from .resnet import ParallelConv2d

def test_conv(rank, ngpus_per_node):
    parallel_conv = ParallelConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    if rank == 0:
        serial_conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, 
                                bias=False)
    
    test_data = torch.randn(1, 3, 224, 224)
    
    parallel_result = parallel_conv(test_data)

    if rank == 0:
        serial_result = serial_conv(test_data)
        print("Parallel Result Shape : ", parallel_result.size())
        print("Serial Result Shape : ", serial_result.size())
        # parallel_result and serial_result should be the same
        assert torch.allclose(parallel_result, serial_result), "Parallel and Serial results are not the same"
        print("Conv layer Test Passed!!")




def main():
    # 한 노드의 GPU수를 가져옴.
    ngpus_per_node = torch.cuda.device_count()

    # multiprocessing_distributed 변수가 true라면, world_size를 총 GPU개수로 설정한 후에, 메인 워커를 실행함.

    mp.spawn(test_conv, nprocs=ngpus_per_node, args=(ngpus_per_node, ))



if __name__ == '__main__':
    main()