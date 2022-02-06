import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

from .mappings import gather_from_tensor_model_parallel_region, copy_to_tensor_model_parallel_region, scatter_to_tensor_model_parallel_region
from .initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from .utils import divide
import torch.nn.init as init
from torch import Tensor
from torch import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional, Callable

num_classes = 1000
 
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return ParallelConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return ParallelConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ParallelBottleNeck(Bottleneck):
    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs):
        super(ParallelBottleNeck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer, **kwargs)
        self.inplanes = inplanes
        self.planes = planes
        
        self.conv1 = conv1x1(self.inplanes, self.planes, stride)
        self.conv2 = conv3x3(self.planes, self.planes, groups, dilation)
        self.conv3 = conv1x1(self.planes, self.expansion * self.planes)


class ParallelConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1,
                 padding=0, dilation=1, groups=1, last_cnn=False, bias=True, **kwargs):
        super(ParallelConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, device=torch.cuda.current_device(), **kwargs)
        self.ngpus_per_node = torch.cuda.device_count()
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):

        splited_input = scatter_to_tensor_model_parallel_region(input, self.kernel_size, True)
        parallel_weight = copy_to_tensor_model_parallel_region(weight)
        if self.padding_mode != 'zeros':
            splited_output = F.conv2d(F.pad(splited_input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            parallel_weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            splited_output = F.conv2d(splited_input, parallel_weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        output = gather_from_tensor_model_parallel_region(splited_output, self.kernel_size, True)

        return output




class ColumnParallelLinear(torch.nn.Linear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, **kwargs):
        super(ColumnParallelLinear, self).__init__(in_features, out_features, bias,
                 device, dtype, **kwargs)

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features

        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(out_features, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.


        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition, self.in_features,
            device=torch.cuda.current_device()))

        self.bias = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            device=torch.cuda.current_device()))

            # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()



    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Set up backprop all-reduce.
        input_parallel = copy_to_tensor_model_parallel_region(input_)

        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, bias)

        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class ParallelResnet(ResNet):
    def __init__(self, dev0, dev1, *args, **kwargs):
        super(ParallelResnet, self).__init__(
            Bottleneck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)
        # dev0 and dev1 each point to a GPU device (usually gpu:0 and gpu:1)
        self.dev0 = dev0
        self.dev1 = dev1
 
        # splits the model in two consecutive sequences : seq0 and seq1 
        self.seq0 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2
        ).to(self.dev0)  # sends the first sequence of the model to the first GPU
 
        self.seq1 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to(self.dev1)  # sends the second sequence of the model to the second GPU
 
        self.fc.to(self.dev1)  # last layer is on the third, fourth GPU
 
    def forward(self, x):
        x= self.seq0(x)     # apply first sequence of the model on input x
        x= x.to(self.dev1)  # send the intermediary result to the second GPU
        x = self.seq1(x)    # apply second sequence of the model to x
        return self.fc(x.view(x.size(0), -1))  


class PipelinedResnet(ResNet):
    def __init__(self, dev0, dev1, split_size=8, *args, **kwargs):
        super(PipelinedResnet, self).__init__(
            Bottleneck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)
        # dev0 and dev1 each point to a GPU device (usually gpu:0 and gpu:1)
        self.dev0 = dev0
        self.dev1 = dev1
        self.split_size = split_size
 
        # splits the model in two consecutive sequences : seq0 and seq1 
        self.seq0 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2
        ).to(self.dev0)  # sends the first sequence of the model to the first GPU
 
        self.seq1 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to(self.dev1)  # sends the second sequence of the model to the second GPU
 
        self.fc.to(self.dev1)  # last layer is on the second GPU
 
    def forward(self, x):
        # split setup for x, containing a batch of (image, label) as a tensor
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        # initialisation: 
        # - first mini batch goes through seq0 (on dev0)
        # - the output is sent to dev1
        s_prev = self.seq0(s_next).to(self.dev1)
        ret = []
 
        for s_next in splits:
            # A. s_prev runs on dev1
            s_prev = self.seq1(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
 
            # B. s_next runs on dev0, which can run concurrently with A
            s_prev = self.seq0(s_next).to(self.dev1)
 
        s_prev = self.seq1(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
 
        return torch.cat(ret)

# resnet 101 임.
class OwnParallelResnet(ResNet):
    def __init__(self, num_classes, *args, **kwargs):
        super(OwnParallelResnet, self).__init__(
            ParallelBottleNeck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)
        
 
        self.conv1 = ParallelConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ColumnParallelLinear(512 * ParallelBottleNeck.expansion, num_classes)

        # 여기서 결과를 all_gather로 합쳐서, columnparallel 고.


    