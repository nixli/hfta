import numpy as np
import torch
import torch.nn as nn

from hfta.ops import get_hfta_op_for, testcase_automator


def testcase(B=3, N=32, L=8, in_features=20, out_features=50, bias=True):
  with torch.no_grad():
    x_array = [torch.rand(N, L, in_features) for _ in range(B)]
    x_fused = torch.cat([x.unsqueeze(0) for x in x_array], dim=0)
    args = (in_features, out_features)
    kwargs = {'bias': bias}
    linear_array = [nn.Linear(*args, **kwargs) for _ in range(B)]
    linear_fused = get_hfta_op_for(nn.Linear, B=B)(*args, **kwargs)
    # Init weights and biases.
    for b in range(B):
      linear_fused.snatch_parameters(linear_array[b], b)
    y_array = [linear_array[b](x_array[b]) for b in range(B)]
    y_fused_actual = linear_fused(x_fused)
    y_fused_expect = torch.cat([y.unsqueeze(0) for y in y_array], dim=0)
    try:
      np.testing.assert_allclose(
          y_fused_actual.numpy(),
          y_fused_expect.numpy(),
          rtol=1e-4,
      )
    except AssertionError as e:
      print(e)


if __name__ == '__main__':
  testcase_automator(
      testcase,
      {
          'B': [1, 2, 5, 10],
          'N': [1, 8, 64],
          'L': [1, 16, 32],
          'in_features': [10, 256, 1],
          'out_features': [100, 10, 1],
          'bias': [False],
      },
  )
