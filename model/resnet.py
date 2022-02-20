import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

from torch.nn.common_types import _size_2_t
from typing import Optional, Union

from .mappings import gather_from_tensor_model_parallel_region, copy_to_tensor_model_parallel_region, scatter_to_tensor_model_parallel_region, reduce_from_tensor_model_parallel_region
from .initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from .utils import divide
import torch.nn.init as init
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Type, Any, Callable, Union, List, Optional

num_classes = 1000
 



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


class WidthParallelConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(WidthParallelConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, device=torch.cuda.current_device(), **kwargs)
        self.ngpus_per_node = torch.cuda.device_count()
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        parallel_weight = copy_to_tensor_model_parallel_region(weight)
        if self.padding_mode != 'zeros':
            splited_input = scatter_to_tensor_model_parallel_region(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), self.kernel_size, padding=self.padding, conv=True)
            splited_output = F.conv2d(splited_input,
                            parallel_weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            splited_input = scatter_to_tensor_model_parallel_region(input, self.kernel_size, self.padding, True)
            splited_output = F.conv2d(splited_input, parallel_weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        output = gather_from_tensor_model_parallel_region(splited_output, self.kernel_size, self.padding, conv=True)

        return output


class WeightParallelConv2d(nn._ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
        ) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            kernel_size_ = _pair(kernel_size)
            stride_ = _pair(stride)
            padding_ = padding if isinstance(padding, str) else _pair(padding)
            dilation_ = _pair(dilation)
            super(WeightParallelConv2d, self).__init__(
                in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
                        False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
            if self.transposed:
                    self.weight = nn.Parameter(torch.empty(
                        (in_channels, out_channels // groups, *kernel_size), device=torch.cuda.current_device(), **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty(
                    (out_channels, in_channels // groups, *kernel_size), device=torch.cuda.current_device(), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_channels, device=torch.cuda.current_device(), **factory_kwargs))
            else:
                self.register_parameter('bias', None)

                self.reset_parameters()
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            output =  F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)

            output = gather_from_tensor_model_parallel_region(output, self.kernel_size, self.padding, conv=True)
            return output

        output = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        output = gather_from_tensor_model_parallel_region(output, self.kernel_size, self.padding, conv=True)
        return output
        
    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class ChannelParallelConv2d(nn._ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
        ) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            kernel_size_ = _pair(kernel_size)
            stride_ = _pair(stride)
            padding_ = padding if isinstance(padding, str) else _pair(padding)
            dilation_ = _pair(dilation)
            super(ChannelParallelConv2d, self).__init__(
                in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
                        False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
            if self.transposed:
                    self.weight = nn.Parameter(torch.empty(
                        (in_channels // groups, out_channels, *kernel_size), device=torch.cuda.current_device(), **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty(
                    (out_channels // groups, in_channels, *kernel_size), device=torch.cuda.current_device(), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_channels, device=torch.cuda.current_device(), **factory_kwargs))
            else:
                self.register_parameter('bias', None)

                self.reset_parameters()
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            output =  F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)

            output = reduce_from_tensor_model_parallel_region(output)
            return output

        output = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        output = reduce_from_tensor_model_parallel_region(output)
        return output
        
    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)



def conv3x3(in_planes: int, out_planes: int, block: Type[Union[WidthParallelConv2d, WeightParallelConv2d, ChannelParallelConv2d]],stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return block(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return block(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



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




# resnet 101 임.
class OwnParallelResnet(ResNet):
    def __init__(self, num_classes, *args, **kwargs):
        super(OwnParallelResnet, self).__init__(
            ParallelBottleNeck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)
        
 
        self.conv1 = WidthParallelConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = ColumnParallelLinear(512 * ParallelBottleNeck.expansion, num_classes)

        # 여기서 결과를 all_gather로 합쳐서, columnparallel 고.

    def _forward_impl(self, x: Tensor) -> Tensor:
        rank = get_tensor_model_parallel_rank()
        # See note [TorchScript super()]
        x = self.conv1(x)
        if not __debug__:
            print("[Rank {} GPU] **self.conv1** Output size : ".format(str(rank)), x.size())
        x = self.bn1(x)
        if not __debug__:
            print("[Rank {} GPU] **self.bn1** Output size : ".format(str(rank)), x.size())
        x = self.relu(x)
        if not __debug__:
            print("[Rank {} GPU] **self.relu** Output size : ".format(str(rank)), x.size())
        x = self.maxpool(x)
        if not __debug__:
            print("[Rank {} GPU] **self.maxpool** Output size : ".format(str(rank)), x.size())
        x = self.layer1(x)
        if not __debug__:
            print("[Rank {} GPU] **self.layer1** Output size : ".format(str(rank)), x.size())
        x = self.layer2(x)
        if not __debug__:
            print("[Rank {} GPU] **self.layer2** Output size : ".format(str(rank)), x.size())
        x = self.layer3(x)
        if not __debug__:
            print("[Rank {} GPU] **self.layer3** Output size : ".format(str(rank)), x.size())
        x = self.layer4(x)
        if not __debug__:
            print("[Rank {} GPU] **self.layer4** Output size : ".format(str(rank)), x.size())
        x = self.avgpool(x)
        if not __debug__:
            print("[Rank {} GPU] **self.avgpool** Output size : ".format(str(rank)), x.size())
        x = torch.flatten(x, 1)
        if not __debug__:
            print("[Rank {} GPU] **self.flatten** Output size : ".format(str(rank)), x.size())
        x = self.fc(x)
        if not __debug__:
            print("[Rank {} GPU] **self.fc** Output size : ".format(str(rank)), x.size())

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


    
