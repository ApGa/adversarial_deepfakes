"""
File: HHReLU.py -- Half-Huber ReLU Function 
Authors: Apurva Gandhi and Shomik Jain
Date: 2/02/2020
"""

import torch
import torch.nn as nn


class HHReLU(nn.Module):
    """Like ReLU, but with a square bowl s.t. derivative is smooth around zero.
    """
    DELTA = 0.5  # Natural choice for imitating ReLU (?)
    def __init__(self):
        super().__init__()
        self.register_buffer('d', torch.tensor(self.DELTA, dtype=torch.float))

    def forward(self, x):
        return HHReLU._forward2.apply(x, self.d)

    class _forward2(torch.autograd.Function):
        """More efficient due to usage of inplace operators, which aren't
        allowed through autograd.
        """
        @staticmethod
        def forward(ctx, x, d):
            ctx.save_for_backward(x, d)

            x = torch.nn.functional.relu(x)
            q = (x.detach() > d).float()

            # Cover linear region
            x.add_(-0.5 * d, q)

            # Hmm... another temporary for multiplying x to get bowl region
            nq = q.neg().add_(1. / (2 * d))
            nq.mul_(x)
            nq.add_(q)
            x.mul_(nq)
            return x
        @staticmethod
        def backward(ctx, grad_output):
            x, d = ctx.saved_tensors

            g = x.clamp(0, d)
            g.div_(d).mul_(grad_output)
            return g, None