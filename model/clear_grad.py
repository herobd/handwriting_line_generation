# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import torch
  
class ClearGrad(torch.nn.Module):
  def __init__(self, module):
      super(ClearGrad, self).__init__()
      def d(_, gradInput, gradOutput):
          for m in module.parameters():
              if m.grad is None:
                  continue
              m.grad.data.zero_()
          return gradInput
      self.register_backward_hook(d)

  def forward(self, x):
      return x
