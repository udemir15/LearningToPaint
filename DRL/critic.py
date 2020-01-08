import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm


def conv3x3(in_planes, out_planes, stride):
    return weightNorm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True))


class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x


def cfg(depth):
    cf_dict = {
        '18': (BasicBlock, [2, 2, 2, 2]),
        '34': (BasicBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101': (Bottleneck, [3, 4, 23, 3]),
        '152': (Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.cnn1 = conv3x3(in_, planes, stride)
        self.cnn2 = conv3x3(planes, planes, 1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ != self.expansion * planes:
            self.shortcut = nn.Sequential(
                weightNorm(nn.Conv2d(in_, self.expansion * planes, kernel_size=1, stride=stride, bias=True)),
            )
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()

    def forward(self, x):
        out = self.relu_1(self.cnn1(x))
        out = self.cnn2(out)
        out += self.shortcut(x)
        out = self.relu_2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.cnn1 = weightNorm(nn.Conv2d(in_, planes, kernel_size=1, bias=True))
        self.cnn2 = weightNorm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True))
        self.cnn3 = weightNorm(nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True))
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()
        self.relu_3 = TReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ != self.expansion * planes:
            self.shortcut = nn.Sequential(
                weightNorm(nn.Conv2d(in_, self.expansion * planes, kernel_size=1, stride=stride, bias=True)),
            )

    def forward(self, x):
        out = self.relu_1(self.cnn1(x))
        out = self.relu_2(self.cnn2(out))
        out = self.cnn3(out)
        out += self.shortcut(x)
        out = self.relu_3(out)

        return out


class ResNet_wobn(nn.Module):
    def __init__(self, num_inputs, depth, num_outputs):
        super(ResNet_wobn, self).__init__()
        self.in_planes = 64

        block, num_blocks = cfg(depth)

        self.cnn1 = conv3x3(num_inputs, 64, 2)
        self.l1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.l2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.l3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.l4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_outputs)
        self.relu_1 = TReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu_1(self.cnn1(x))
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x