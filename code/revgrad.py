import torch
from torch.nn import Module
from torch.autograd import Function


class GradientReversal(Function):

    # TODO: Optimize alpha?
    @staticmethod
    def forward(ctx, input_, alpha=torch.tensor(1.0)):
        ctx.save_for_backward(input_, alpha)
        output = input_.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, alpha = ctx.saved_tensors
        grad_input = None

        # Only compute if input needs gradient, otherwise return None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.neg() * alpha
        return grad_input, None


class RevGrad(Module):
    def __init__(self, *args, **kwargs):
        """
        A gradient reversal layer.
        """

        super().__init__(*args, **kwargs)

    def forward(self, input_):
        return GradientReversal.apply(input_)
