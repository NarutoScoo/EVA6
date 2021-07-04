'''ResNet in PyTorch. (https://github.com/kuangliu/pytorch-cifar)

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    
    
Modified to select different normalization methods
(either Batch Norm, Group Norm, Layer Norm or Instance Norm)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, normalization='batch_norm'):
        super(BasicBlock, self).__init__()
        
        # Check if the valid normalization strategy was passed
        # Number of groups used is set to 2 for group norm
        assert normalization in ['group_norm', 'layer_norm', 'batch_norm', 'instance_norm'], \
            '`normalization` should be one of `group_norm`, `layer_norm`, ' + \
            '`batch_norm` or `instance_norm`'
        
        # Initialize normalization
        # Similate Layer Norm and Instance Norm using Group Norm 
        # with groups 1 and no. of channels respectively    
        self.norm = normalization
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=4, num_channels=planes) if self.norm=='group_norm' \
                    else (nn.GroupNorm (num_groups=1, num_channels=planes) if self.norm=='layer_norm' \
                    else nn.GroupNorm (num_groups=planes, num_channels=planes)))
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=4, num_channels=planes) if self.norm=='group_norm' \
                    else (nn.GroupNorm (num_groups=1, num_channels=planes) if self.norm=='layer_norm' \
                    else nn.GroupNorm (num_groups=planes, num_channels=planes)))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=4, num_channels=self.expansion*planes) if self.norm=='group_norm' \
                    else (nn.GroupNorm (num_groups=1, num_channels=self.expansion*planes) if self.norm=='layer_norm' \
                    else nn.GroupNorm (num_groups=self.expansion*planes, num_channels=self.expansion*planes)))
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, normalization='batch_norm'):
        super(ResNet, self).__init__()
        
        # Check if the valid normalization strategy was passed
        # Number of groups used is set to 2 for group norm
        assert normalization in ['group_norm', 'layer_norm', 'batch_norm', 'instance_norm'], \
            '`normalization` should be one of `group_norm`, `layer_norm`, ' + \
            '`batch_norm` or `instance_norm`'
        
        # Initialize normalization
        # Similate Layer Norm and Instance Norm using Group Norm 
        # with groups 1 and no. of channels respectively        
        self.norm = normalization
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=4, num_channels=64) if self.norm=='group_norm' \
                    else (nn.GroupNorm (num_groups=1, num_channels=64) if self.norm=='layer_norm' \
                    else nn.GroupNorm (num_groups=64, num_channels=64)))
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(normalization='batch_norm'):
    # Check if the valid normalization strategy was passed
    # Number of groups used is set to 2 for group norm
    assert normalization in ['group_norm', 'layer_norm', 'batch_norm', 'instance_norm'], \
        '`normalization` should be one of `group_norm`, `layer_norm`, ' + \
        '`batch_norm` or `instance_norm`'
    
    return ResNet(BasicBlock, [2, 2, 2, 2], normalization=normalization)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], normalization=normalization)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())