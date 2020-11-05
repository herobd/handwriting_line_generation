#Curtis's code to clear the weight gradients in a module at a specific point in the computation graph (where the data flows through this module)
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
