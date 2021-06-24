# This file `model.py` contains the network architecture
# and the functions for training and testing

# Imports
import torch # PyTorch, everything from PyTorch can be referred with this
import torch.nn as nn # Import the neural network model seperately
import torch.nn.functional as F # Contains functions that are required such as ReLu

from collections import OrderedDict # Utils

# TQDM is just awesome... provides a progress status bar as the training 
# (or any operation) proceeds
from tqdm import tqdm


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


def train (model, device, train_loader, optimizer, scheduler=None,
           regularization=None, l1_lambda=0, l2_lambda=0):
    ''' Define the training steps 
    
    Parameters:
        model          - Created model object with the required architecture
        device         - Inidicates where to run 
                            (Values: [`cpu`, `cuda`])
        train_loader   - Data loader for training set
        scheduler      - Learning Rate Scheduler if used
                          (Default: None)
        regularization - Additional constraint added to the loss fn
                            (Values: [`None`, `l1`, `l2`, `l1_l2`];
                             Default: `None`)
        l1_lambda      - Weight given to L1 regularization loss
                            (Values: [0, 1]; Default: `0`)
        l2_lambda      - Weight given to L2 regularization loss
                            (Values: [0, 1]; Default: `0`)
    '''

    # Set the model to training mode
    model.train()
    
    # Initialize the counters
    correct = 0
    processed = 0

    # Initialize the progress bar
    pbar = tqdm(train_loader)

    # Initialize the list of accuracies and losses
    training_accuracy = []
    training_losses = []

    # Start iterating through the training data
    for batch_idx, (data, target) in enumerate(pbar):
        # Start by converting the data to the required type 
        # (PyTorch Cuda Tensor while using GPU)
        data, target = data.to(device), target.to(device)

        # Discard/reset the gradients from the last iteration
        optimizer.zero_grad()

        # Get the predictions for the given data
        output = model(data)

        # Compute the negative loss likelihood of the predictions vs the actuals
        # and propogate the loss backwards (back propogation)
        loss = F.nll_loss(output, target)
        
        # Check if regularization needs to be applied
        if regularization and 'l1' in regularization:
            # Check the lambda for the loss
            assert l1_lambda != 0, \
                'Lambda value for L1 (`l1_lambda`) is 0; Should be > 0 to apply L1 Regularization'
            
            # Apply L1 Regularization to the loss
            # Reference: https://stackoverflow.com/a/66543549
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            
            # Updat the loss
            loss += l1_lambda * l1_norm
            
        if regularization and 'l2' in regularization:
            # Check the lambda for the loss
            assert l2_lambda != 0, \
                'Lambda value for L2 (`l2_lambda`) is 0; Should be > 0 to apply L2 Regularization'
            
            # Apply L2 Regularization to the loss
            # Reference: https://stackoverflow.com/a/66543549
            l2_norm = sum (p.pow (2.0).sum() for p in model.parameters())
            
            # Update the loss
            loss += l2_lambda * l2_norm
        
        # Backpropogate the loss
        loss.backward()

        # Use the optimizer to take a step in the -ve of the gradient direction
        # by the amount equal to the gradient magnitude times the learning rate 
        optimizer.step()
        
        # Get the index of the prediction
        # i.e. the output is one-hot encoded, so get the argument with the max
        # log probability
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        # Get a count of the correct preditcions
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Update the no of records processed
        processed += len (data)

        # Update the progress bar
        pbar.set_description(
            desc=f'Loss={loss.item():.4f}; '+\
                 f'Accuracy={correct*100/processed:.2f}; '+\
                 f'Learning Rate={optimizer.param_groups[0]["lr"]:.4f}')
        
        # Store the training accuracy & loss
        training_accuracy.append (correct*100/processed)
        training_losses.append (loss)

    # Update the learning rate after each training cycle
    if scheduler:
        scheduler.step ()

    # Return the accuracy and losses after each epoch
    #return training_accuracy [-1], training_losses [-1]
    # Return the accuracy and losses on the entire training set
    # after training an epoch
    return test (model, device, train_loader, test_r_train='train')
    


def test(model, device, test_loader, test_r_train='test'):
    ''' Validate the trained model on a hold-out set 
    
    Parameters:
        model       - Trainined model used to predict
        device      - Indicates the device to run on 
                          (Values: [`cpu`, `cuda`])
        test_loader - Data loader for test/validation data
    '''

    # Set the model to evalution mode
    model.eval()
    
    # Initialize the losses
    # and the no of correct predictions to 0
    test_loss = 0
    correct = 0
    
    # Store the misclassified images
    misclassified_images = None

    # Disable the gradient computations
    # While evaulating only forward pass is used and the backward pass
    # along with the gradient (likewise the gradient update) isn't required
    with torch.no_grad():
        # Iterate over the test/validation set
        for data, target in test_loader:
            # Converting the data to the required type 
            # (PyTorch Cuda Tensor while using GPU)
            data, target = data.to(device), target.to(device)

            # Get the predictions
            output = model(data)

            # Compute the loss against the target
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            # Get the index of the prediction
            # i.e. the output is one-hot encoded, so get the argument with the max
            # log probability
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # Get a count of the correct preditcions
            correct += pred.eq (target.view_as(pred)).sum().item()
            
            # Get a list of misclassified images
            if isinstance (misclassified_images, torch.Tensor):
                misclassified_images = torch.cat (
                    (misclassified_images, data [~pred.eq (target.view_as(pred))]), 
                    dim=0
                )
            else:
                misclassified_images = data [~pred.eq (target.view_as(pred))]
                
            
    # Compute the final loss on the test/validation data
    test_loss /= len(test_loader.dataset)

    # Display the results
    print('\n{} set: Loss={:.4f}; Accuracy={}/{} ({:.2f}%)\n\n'.format(
        test_r_train.title (), test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Return the test accuracy and loss
    # Update LR based on this
    return correct*100/len(test_loader.dataset), test_loss, misclassified_images