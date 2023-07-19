import torch
import torch.nn as nn
from torch.autograd import Variable


class DynamicRegularizationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, training=True, R=0, d=0, A=1):
        if training:
            R_save = torch.cuda.FloatTensor([R])
            d_save = torch.cuda.FloatTensor([d])
            A_save = torch.cuda.FloatTensor([A])
            ctx.save_for_backward(R_save, d_save, A_save)

            r = torch.cuda.FloatTensor(x.size(0)).uniform_(-R, R)
            Theta = A + d * r
            Theta = Theta.view(Theta.size(0), 1, 1, 1).expand_as(x)
            return Theta * x
        else:
            return A * x

    @staticmethod
    def backward(ctx, grad_output):
        R_save, d_save, A_save = ctx.saved_tensors
        R = R_save.item()
        d = d_save.item()
        A = A_save.item()

        r = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(-R, R)
        u = A + d * r
        u = u.view(u.size(0), 1, 1, 1).expand_as(grad_output)
        u = Variable(u)
        return u * grad_output, None, None, None, None


class DynamicRegularization(nn.Module):
    def __init__(self, R=0, d=0, A=5):
        super(DynamicRegularization, self).__init__()
        self.R = R
        self.d = d
        self.A = A

    def forward(self, x):
        return DynamicRegularizationFunction.apply(x, self.training, self.R, self.d, self.A)
