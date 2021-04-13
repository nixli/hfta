import functools
import torch.nn

from .conv import Conv1d, Conv2d, ConvTranspose2d
from .linear import Linear
from .pool import MaxPool2d, AdaptiveAvgPool2d
from .dropout2d import Dropout2d
from .batchnorm import BatchNorm1d, BatchNorm2d
from .utils import testcase_automator
from .embedding import Embedding
from .layernorm import LayerNorm
from .multiheadattention import MultiheadAttention
from .transformerencoderlayer import TransformerEncoderLayer

_OPS_MAP = {
    torch.nn.Conv1d: Conv1d,
    torch.nn.Conv2d: Conv2d,
    torch.nn.ConvTranspose2d: ConvTranspose2d,
    torch.nn.Linear: Linear,
    torch.nn.MaxPool2d: MaxPool2d,
    torch.nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d,
    torch.nn.Dropout2d: Dropout2d,
    torch.nn.BatchNorm1d: BatchNorm1d,
    torch.nn.BatchNorm2d: BatchNorm2d,
    torch.nn.LayerNorm: LayerNorm,
    torch.nn.Embedding: Embedding,
}

_HFTA_TORCH_IDENTICAL_OPS = {
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.Tanh,
    torch.nn.LeakyReLU,
    torch.nn.Dropout,
    torch.nn.TransformerEncoder,
}



_OPS_ARGS_PACKED = {
    torch.nn.Conv1d: lambda x: (x.in_channels, x.out_channels, x.kernel_size, x.stride,
                                x.padding, x.dilation,
                                x.groups, x.bias, x.padding_mode),

    torch.nn.Conv2d: lambda x: (x.in_channels, x.out_channels, x.kernel_size, x.stride,
                                x.padding, x.dilation,
                                x.groups, x.bias, x.padding_mode),

    torch.nn.ConvTranspose2d: lambda x: (x.in_channels, x.out_channels, x.kernel_size, x.stride,
                                        x.padding, x.output_padding,  x.groups,
                                        x.bias, x.dilation, x.padding_mode),

    torch.nn.Linear: lambda x: (x.in_features, x.out_features, x.bias is not None),

    torch.nn.MaxPool2d: lambda x: (x.kernel_size, x.stride, x.padding,
                                x.dilation, x.return_indices, x.ceil_mode,) ,

    torch.nn.AdaptiveAvgPool2d: lambda x: (x.output_size, ),

    torch.nn.Dropout2d: lambda x: (x.p, x.inplace),

    torch.nn.BatchNorm1d: lambda x: (x.num_features,
                                     x.eps, x.momentum, x.affine,
                                    x.track_running_stats),
    torch.nn.BatchNorm2d: lambda x: (x.num_features,
                                     x.eps, x.momentum, x.affine,
                                    x.track_running_stats),
    torch.nn.LayerNorm: lambda x: None,
    torch.nn.Embedding: lambda x: None,
}


def get_hfta_op_for(torch_op_class, B=1):
  if B > 0:
    if torch_op_class in _HFTA_TORCH_IDENTICAL_OPS:
      return torch_op_class
    else:
      return functools.partial(_OPS_MAP[torch_op_class], B=B)
  else:
    return torch_op_class


def convert_ops(B, *torch_op_classes):
  return (get_hfta_op_for(op_class, B=B) for op_class in torch_op_classes)


def expand_to_hfta_op(torch_op, B=1):

  torch_op_class = type(torch_op)

  if B > 0:
    if torch_op_class in _HFTA_TORCH_IDENTICAL_OPS:
      return torch_op
    else:
      args = _OPS_ARGS_PACKED[torch_op_class](torch_op)
      return functools.partial(_OPS_MAP[torch_op_class], B=B)(*args)
  else:
    return torch_op
