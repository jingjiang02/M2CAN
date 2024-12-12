import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageFeatureExtractor(nn.Module):
    def __init__(self, projection=False):
        super(ImageFeatureExtractor, self).__init__()
        self.projection = projection

        self.linear = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True)
        )

        if self.projection:
            self.g = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64)
            )

    def forward(self, input):
        output = self.linear(input)
        if self.projection:
            return F.normalize(output, dim=-1), F.normalize(self.g(output), dim=-1)
        else:
            return F.normalize(output, dim=-1)


class TextFeatureExtractor(nn.Module):
    def __init__(self, projection=False):
        super(TextFeatureExtractor, self).__init__()
        self.projection = projection

        self.hidden = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True)
        )

        if self.projection:
            self.g = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64)
            )

    def forward(self, input):
        output = self.hidden(input)
        if self.projection:
            return F.normalize(output, dim=-1), F.normalize(self.g(output), dim=-1)
        else:
            return F.normalize(output, dim=-1)


class DomainDiscriminator(nn.Module):
    def __init__(self, dim=None):
        super(DomainDiscriminator, self).__init__()
        if dim is not None:
            self.dim = dim
        else:
            self.dim = 64
        self.linear = nn.Linear(self.dim, 1)

    def forward(self, input):
        output = self.linear(input)
        return torch.sigmoid(output)


class TaskClassifier(nn.Module):
    def __init__(self, dim=None, out_dim=3):
        super(TaskClassifier, self).__init__()
        if dim is None:
            self.dim = 64
        else:
            self.dim = dim
        self.linear = nn.Linear(self.dim, out_dim)

    def forward(self, input):
        output = self.linear(input)
        return output


class MLB(nn.Module):
    def __init__(self):
        super(MLB, self).__init__()
        self.dim = 256
        self.U = nn.Linear(self.dim, self.dim // 2)
        self.V = nn.Linear(self.dim, self.dim // 2)
        self.P = nn.Linear(self.dim // 2, self.dim // 4)
        self.h1 = nn.Linear(self.dim, self.dim // 4)
        self.h2 = nn.Linear(self.dim, self.dim // 4)

    def forward(self, input1, input2):
        output1 = self.U(input1)
        output2 = self.V(input2)
        output = F.sigmoid(output1 * output2)
        output = self.P(output) + self.h1(input1) + self.h2(input2)
        return output


# Gradient Reversal Layer
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(self, inputs):
        return inputs

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input
