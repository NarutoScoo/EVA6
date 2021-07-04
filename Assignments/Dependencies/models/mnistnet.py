# This file `mnistnet.py` contains the custom network architecture
# defined for MNIST using different normalization methods (Batch, Group, Layer and Instance Norm)

################# Imports #################
import torch # PyTorch, everything from PyTorch can be referred with this
import torch.nn as nn # Import the neural network model seperately
import torch.nn.functional as F # Contains functions that are required such as ReLu

from collections import OrderedDict # Utils

################# Network Architecture #################

class MNISTNet(nn.Module):
    ''' Define a class initializing the layers constituting the required 
    CNN Architecture and the code for forward pass
    Note: The class extends to the nn.Module, which is a base class for 
    Neural Network modules in PyTorch https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    '''

    def __init__(self, drop_perc=0.05, normalization='batch_norm'):
        ''' Define the layers that constitute the network and
        initialize the base class 
        
        Parameters:
            drop_prec - Dropout percentage to use
            normalization - The normalization strategy to select
                                (Values: [`batch_norm`, `group_norm`, 
                                    `layer_norm`, `instance_norm`];
                                Default: `batch_norm`)
        '''

        # Start by initializing the base class
        super().__init__()

        # Store the dropout percentage
        self.drop_perc = drop_perc
    
        # Check if the valid normalization strategy was passed
        # Number of groups used is set to 2 for group norm
        assert normalization in ['group_norm', 'layer_norm', 'batch_norm', 'instance_norm'], \
            '`normalization` should be one of `group_norm`, `layer_norm`, ' + \
            '`batch_norm` or `instance_norm`'
        self.norm = normalization
        
        # Define the layers that make up the network
        # i.e. the Network Architecture
        # nn.Conv2d - Used to perform 2-dimensional convolution using the defined size of the kernel
        # nn.MaxPool2d - 2d MaxPooling Layer of the defined size

        # MNIST contains images of size 28x28
        # Since the images are padded, the resultant images after convolution
        # would have the same size
        
        # Input Block
        self.input_block = nn.Sequential (
            OrderedDict ([
                ('input_block_conv1', nn.Conv2d (1, 16, 3, padding=0, bias=False)), # Out: 26
                (f'input_block_{self.norm}', 
                    nn.BatchNorm2d (num_features=16) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=4, num_channels=16) if self.norm=='group_norm' \
                    else (nn.LayerNorm (normalized_shape=[16, 26, 26]) if self.norm=='layer_norm' \
                    else nn.InstanceNorm2d (num_features=16)))),
                ('input_block_relu', nn.ReLU ()),
                ('input_block_dropout', nn.Dropout (self.drop_perc)), 
                ('input_block_one', nn.Conv2d (16, 8, 1, padding=0, bias=False)) # Out: 26
            ])
        )

        # First Convolution Block
        self.block1 = nn.Sequential (
            OrderedDict ([
                ('block1_conv1', nn.Conv2d (8, 8, 3, padding=0, bias=False)), # Out: 24
                (f'block1_{self.norm}1',
                    nn.BatchNorm2d (num_features=8) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=2, num_channels=8) if self.norm=='group_norm' \
                    else (nn.LayerNorm (normalized_shape=[8, 24, 24]) if self.norm=='layer_norm' \
                    else nn.InstanceNorm2d (num_features=8)))),
                ('block1_relu1', nn.ReLU ()),
                ('block1_dropout1', nn.Dropout (self.drop_perc)), 

                ('block1_conv2', nn.Conv2d (8, 16, 3, padding=1, bias=False)), # Out: 24
                (f'block1_{self.norm}2',
                    nn.BatchNorm2d (num_features=16) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=2, num_channels=16) if self.norm=='group_norm' \
                    else (nn.LayerNorm (normalized_shape=[16, 24, 24]) if self.norm=='layer_norm' \
                    else nn.InstanceNorm2d (num_features=16)))),
                ('block1_relu2', nn.ReLU ()),
                ('block1_dropout2', nn.Dropout (self.drop_perc)),     
            ])    
        )

        # Pooling Layer
        self.pool_block1 = nn.Sequential (
            OrderedDict ([
                ('pool_b1_mp', nn.MaxPool2d (2, 2)), # Out: 12
                ('pool_b1_one', nn.Conv2d (16, 8, 1, padding=0, bias=False)) # Out: 12
            ])
        )

        # Second Convolution Block
        self.block2 = nn.Sequential (
            OrderedDict ([
                ('block2_conv1', nn.Conv2d (8, 8, 3, padding=0, bias=False)), # Out: 10
                (f'block2_{self.norm}1',
                    nn.BatchNorm2d (num_features=8) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=2, num_channels=8) if self.norm=='group_norm' \
                    else (nn.LayerNorm (normalized_shape=[8, 10, 10]) if self.norm=='layer_norm' \
                    else nn.InstanceNorm2d (num_features=8)))),
                ('block2_relu1', nn.ReLU ()),
                ('block2_dropout1', nn.Dropout (self.drop_perc)), 

                ('block2_conv2', nn.Conv2d (8, 14, 3, padding=1, bias=False)), # Out: 10 
                (f'block2_{self.norm}2',
                    nn.BatchNorm2d (num_features=14) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=2, num_channels=14) if self.norm=='group_norm' \
                    else (nn.LayerNorm (normalized_shape=[14, 10, 10]) if self.norm=='layer_norm' \
                    else nn.InstanceNorm2d (num_features=14)))),
                ('block2_relu2', nn.ReLU ()),
                ('block2_dropout2', nn.Dropout (self.drop_perc)),     
                ('block2_one', nn.Conv2d (14, 8, 1, padding=0, bias=False)) # Out: 10
            ])    
        )

        # Last Convolution Block
        self.block3 = nn.Sequential (
            OrderedDict ([
                ('block3_conv1', nn.Conv2d (8, 16, 3, padding=0, bias=False)), # Out: 8
                (f'block3_{self.norm}1',
                    nn.BatchNorm2d (num_features=16) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=2, num_channels=16) if self.norm=='group_norm' \
                    else (nn.LayerNorm (normalized_shape=[16, 8, 8]) if self.norm=='layer_norm' \
                    else nn.InstanceNorm2d (num_features=16)))),
                ('block3_relu1', nn.ReLU ()),
                ('block3_dropout1', nn.Dropout (self.drop_perc)), 

                ('block3_conv2', nn.Conv2d (16, 18, 3, padding=0, bias=False)), # Out: 6
                (f'block3_{self.norm}2',
                    nn.BatchNorm2d (num_features=18) if self.norm=='batch_norm' \
                    else (nn.GroupNorm (num_groups=2, num_channels=18) if self.norm=='group_norm' \
                    else (nn.LayerNorm (normalized_shape=[18, 6, 6]) if self.norm=='layer_norm' \
                    else nn.InstanceNorm2d (num_features=18)))),
                ('block3_relu2', nn.ReLU ()),
                ('block3_dropout2', nn.Dropout (self.drop_perc)),    
            ])
        )

        # Gap & Final layer with the predictions
        self.prediction = nn.Sequential (
            OrderedDict ([
                ('gap', nn.AvgPool2d (kernel_size=6)),
                ('pred_layer', nn.Conv2d (18, 10, 1, padding=0, bias=False))
            ])
        )

        
    def forward(self, x):
        ''' Define the forward pass
        Each convolution layer is activated using ReLU to add non-linearity
        '''
        # Start with the input block to convert the grayscale (no. of channels = 1)
        # to something higher
        x = self.input_block (x)

        # Convolution layer followed by ReLU Activation, followed by Batch Normalization
        # followed by Dropout and then finally a pooling layer
        # Block 1
        x = self.block1 (x)

        # Pooling layer
        x = self.pool_block1 (x)

        # Block 2
        x = self.block2 (x)

        # Block 3
        x = self.block3 (x)

        # The final layer shouldn't be passed through ReLU, but should be retained
        # Using Global Average Pooling & and 1x1 to reduce the final output size
        x = self.prediction (x)

        # Reshape to fit the output and return
        x = x.view (-1, 10)
        return F.log_softmax (x)