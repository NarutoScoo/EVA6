# This file `main.py` contains the code to train and test the network
# It also has functions to choose the optimizer and the schedulers to use

################# Imports #################

import torch # PyTorch, everything from PyTorch can be referred with this
import torch.optim as optim # Optimizers required to converge using Backpropogation

import torch.nn.functional as F # Contains functions such as ReLu (Activation function)

import numpy as np
# TQDM is just awesome... provides a progress status bar as the training 
# (or any operation) proceeds
from tqdm import tqdm


################# Functions #################

def get_optimizer (model, optimizer='sgd', learning_rate=0.05, momentum=0.9):
    ''' Define an optimizer to use to perform gradient descent
        Various optimizers are suitable for different usecase, which help
        reach the global optimal (i.e. a model with least errors/loss) quickly
    
    Parameters:
        model           - Model parameters for which the optimizer would be applied
        optimizer       - Optimzer to choose from
                            (Values: ['sgd', 'adam']; Default: 'sgd')
        learning_rate   - Initial Learning Rate to use
                            (Default: 0.05)
        momentum        - Amount of momentum 
                            (i.e. amout of previous weight to consider)
                            (Default: 0.9)
                            
    Return:
        optimizer       - The configured optimizer
    '''

    # Check for the optimizers
    assert optimizer in ['sgd', 'adam'], \
        f'{optimizer.title ()} is not defined/implemented yet'
    
    
    if optimizer == 'sgd':
        # In this case, Stochastic Gradient Descent (SGD) is used with momentum of 0.9
        # and the learning rate (alpha) set to 0.01 

        # Have the learning rate initially so that it gets to the region of the global minima
        # faster and then reduce it so the it converges slowly to the required minima
        # This should avoid falling into local minima
        optimizer = optim.SGD (model.parameters(), lr=0.04, momentum=0.9)
        
    if optimizer == 'adam':
        pass
        
    # Return the configured optimizer
    return optimizer
        
    
def get_lr_scheduler (optimizer, lr_scheduler='ReduceLROnPlateau', **kwargs):
    ''' Define a learning rate scheduler 
        Refered from: https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
            and https://pytorch.org/docs/master/optim.html
            
    Parameters:
        optimizer     - The optimizer used
        lr_scheduler  - LR Scheduler to configure
                            (Values: ['ReduceLROnPlateau', 'MultiStepLR'];
                             Default: 'ReduceLROnPlateau')
        **kwargs      - Additional parameters required for the optimizers
    
    Return:
        lr_scheduler - Defined Learning Rate Scheduler
    '''
    
    # Check the scheduler
    assert lr_scheduler in ['ReduceLROnPlateau', 'MultiStepLR'], \
        'Should be one of "ReduceLROnPlateau", "MultiStepLR"'

    
    if lr_scheduler == 'ReduceLROnPlateau':
        # https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        # Decrease the learning rate after a particular step
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau (
            optimizer, mode='min', 
            factor=0.8, patience=1, 
            min_lr=0.0001, threshold=0.001, 
            verbose=True)

    elif lr_scheduler == 'MultiStepLR':
        # Reduce the Learning Rate at the specified iteration
        # Choose this is based on the previous runs
        multistep_scheduler = optim.lr_scheduler.MultiStepLR (
            optimizer, milestones=[25], 
            gamma=0.05, verbose=True)


def train (model, device, train_loader, optimizer, criteria, 
        scheduler=None, regularization=None, l1_lambda=0, l2_lambda=0):
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
        criteria       - Criteria on which the loss is calculated
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
        loss = criteria (output, target)
        
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
    return test (model, device, train_loader, criteria, test_r_train='train')
    


def test(model, device, test_loader, criteria, test_r_train='test'):
    ''' Validate the trained model on a hold-out set 
    
    Parameters:
        model       - Trainined model used to predict
        device      - Indicates the device to run on 
                          (Values: [`cpu`, `cuda`])
        test_loader - Data loader for test/validation data
        criteria    - The loss criteria
    '''

    # Set the model to evalution mode
    model.eval()
    
    # Initialize the losses
    # and the no of correct predictions to 0
    test_loss = 0
    correct = 0
    
    # Store the misclassified images and its targets
    misclassified_images = None
    misclassified_targets = None

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
            test_loss += criteria (output, target, reduction='sum').item()  # sum up batch loss

            # Get the index of the prediction
            # i.e. the output is one-hot encoded, so get the argument with the max
            # log probability
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # Get a count of the correct preditcions
            correct += pred.eq (target.view_as(pred)).sum().item()
            
            # Get a list of misclassified images
            if isinstance (misclassified_images, torch.Tensor):
                misclassified_images = torch.cat (
                    (misclassified_images, data [~pred.eq (target.view_as(pred)).squeeze ()]), 
                    dim=0
                )
                misclassified_targets = torch.cat (
                    (misclassified_targets, target [~pred.eq (target.view_as(pred)).squeeze ()]), 
                    dim=0
                )
            else:
                misclassified_images = data [~pred.eq (target.view_as(pred)).squeeze ()]
                misclassified_targets = target [~pred.eq (target.view_as(pred)).squeeze ()]
                
            
    # Compute the final loss on the test/validation data
    test_loss /= len(test_loader.dataset)

    # Display the results
    print('\n{} set: Loss={:.4f}; Accuracy={}/{} ({:.2f}%)\n\n'.format(
        test_r_train.title (), test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Return the test accuracy and loss
    # Update LR based on this
    return correct*100/len(test_loader.dataset), test_loss, misclassified_images


def main (model, model_logs, optimizer, criteria, train_loader, test_loader,
        epochs=20, normalization='batch_norm', lr_schedulers=['ReduceLROnPlateau'],
        regularization=None, l1_lambda=0.0001, l2_lambda=0.001):

    ''' The main functions which calls the training and test loops
        also logs the model results

    Parameters:
        model           - Model defining the network architecture
        model_logs      - Store the results of each epoch for the model
        optimizer       - Optimizer to used for gradient descent
        criteria        - Loss criteria on which the loss is computed
                            (Values: ['cross_entropy', 'nll'])
        train_loader    - DataLoader defined on the training set
        test_loader     - DataLoader defined on the test/validation set
        epochs          - No. of epochs of training to be performed
                                (Default: 20)
        normalization   - Normalization method
                                (Values: ['batch_norm', 'layer_norm', 
                                    'group_norm', 'instance_norm];
                                 Default: 'batch_norm')
        regularization  - Regularization method to use
                                (Values: [None, 'l1', 'l2', 'l1_l2'];
                                 Default: None)
        l1_lambda       - The weight given to L1 Regularization Loss
                                (Default: 0.0001)
        l2_lambda       - Weight given to L2 Regularization Loss
                                (Default: 0.001)
        lr_schedulers   - A list of Learning Rate Schedulers to use
                                (Default: ['ReduceLROnPlateau'])

    Return:
        model_logs      - Update logs with the results from the current run
    '''

    # Store the training accuracy and losses
    training_accuracy = []
    training_losses = []

    # Store the test accuracy and losses
    test_accuracy = []
    test_losses = []

    # Set it to use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Check & define the loss criteria
    assert criteria in ['cross_entropy', 'nll'], \
        'The criteria should be either "cross_entropy" or "nll"'
    criteria = F.nll_loss if criteria == 'nll' else F.cross_entropy

    # Get the schedulers
    schedulers = {}
    for lr_scheduler in lr_schedulers:
        schedulers [lr_scheduler] = get_lr_scheduler (optimizer, lr_scheduler)

    try:
        for epoch in range(0, epochs):
            print (f'Iteration {epoch+1}')

            # Initiate training phase
            accuracy, loss, _ = train (
                model, device, 
                train_loader, optimizer, criteria,
                scheduler=schedulers.get ('MultiStepLR', None),
                regularization=regularization, # Regularization
                l1_lambda=l1_lambda, l2_lambda=l2_lambda # L1 and L2 lambdas
            )

            # Add it to the list
            # Accuracy and losses on the entire training set
            # at the end of all the batches
            training_accuracy.append (accuracy)
            training_losses.append (loss)

            # Validate the results on the test/validation set
            accuracy, loss, misclassified_images = test (
                model, 
                device, 
                test_loader,
                criteria
            )

            # Add it to the list
            # The accuracy and loss are computed on the entire
            # test data, so will return a single value
            test_accuracy.append (accuracy)
            test_losses.append (loss)

            # Reduce the learning rate if the loss has plateaued
            if schedulers.get ('ReduceLROnPlateau', None):
                schedulers ['ReduceLROnPlateau'].step (loss)

        print (f'Best Model had a Training Accuracy of {np.max (training_accuracy):.2f}', 
                f'& a Test Accuracy of {np.max (test_accuracy):.2f}')

        # Update the model log
        model_logs [normalization] ['misclassified_images'] = misclassified_images
        model_logs [normalization] ['training_accuracy'] = training_accuracy
        model_logs [normalization] ['training_losses'] = training_losses
        model_logs [normalization] ['test_accuracy'] = test_accuracy
        model_logs [normalization] ['test_losses'] = test_losses
    
    except KeyboardInterrupt:
        # Return the model logs atleast for the epochs done so far
        pass

    # Return the logged results
    return model_logs
    

if __name__ == '__main__':
    # Call the main function
    main ()