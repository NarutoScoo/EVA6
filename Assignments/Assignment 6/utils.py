# This file `utils.py` contains utility functions/classes

# Imports
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

# Define a class a that loads and takes the images from MNIST to use as input 
# along with the actual label of the handwritten digit which constitutes the required output
class MNISTDataset (Dataset):
    ''' Extending the Dataset class to provide customized inputs and modified 
    outputs for training '''

    def __init__(self, mode='training', additional_transforms=None):
        ''' Get/Download the data (MNIST) and setup up other configurations
        required to get the data
        
        Parameters:
            mode                  - Indicates the mode of operation 
                                        (Values: [`training`, `validation`]; Default: `training`)
            additional_transforms - A list of additional transformations to be added while training
                                          (Default: `None)
                                          Note: Normalization of the dataset is applied always
        '''

        # Start by initializing the base class
        super().__init__()

        # Save the mode
        # Check if the mode is valid
        assert mode in ['training', 'validation'], \
            '"mode" should be either "training" or "validation"'
        self.mode = mode

        # Seed the random generator to results that are reproduceable
        # Remove during production
        torch.manual_seed (1)

        # Define the transformations
        #   First convert into a tensor and then normalization
        #   Values for normalization where got from 
        #   https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457

        # Define a list of transforms to chain together            
        # Both the train and validation transforms are the same, but might decide 
        # to update with some data augmentation methods for train later
        # so keeping it seperate
        self.transforms = [
            transforms.ToTensor (), 
            transforms.Normalize((0.1307,), (0.3081,))]

        # Get MNIST data
        if self.mode == 'training':        

            # Chain the transforms
            if additional_transforms:
                self.transforms = self.transforms+additional_transforms 
        
            # Download the training set and store it at the root directory
            self.data = datasets.MNIST (
                root='./data/train/',
                train=True,
                download=True,
                transform=transforms.Compose (self.transforms))
            
        elif self.mode == 'validation':

            # Get the validation set
            self.data = datasets.MNIST (
                root='./data/val/',
                train=False,
                download=True,
                transform=transforms.Compose (self.transforms))
        

    def __len__(self):
        ''' Get the total size of the dataset '''
        return len (self.data)


    def __getitem__(self, index):
        ''' Used to get the index-th item from the dataset 
        
        Parameters:
            index - Element value to retrieve from the data
        '''

        # Check if the within the bounds of the data
        assert index < len (self.data), 'The index value is out-of-bound'

        # Index into the data to get the image and the corresponding label
        img, label = self.data [index]

        # Return the training set and the corresponding label
        return img, label
    
    
    
def load_train_test (mode='both', additional_transforms=None, batch_size=128, **kwargs):
    ''' Load the training or the test set 
    
    Parameters:
        mode                  - Load the dataset for training and/or test 
                                    (Values: [`both`, `train` or `test`]; Default: `both`)
        additional_transforms - Additional transformation to be applied while training
                                    (Default: `None`)
        batch_size            - No. of samples per batch (Default: `128`)
    '''
    
    # Check the passed values
    assert mode in ['both', 'train', 'test'], \
        '`mode` should be either `both`, `train` or `test`'
    
    # Define a dataloader 
    # Set the batch_size and shuffle to select
    # random images for every batch and epoch
    
    # Set the loaders to None and assign based on the selected mode
    train_loader, test_loader = None, None
    
    # Training mode
    if mode in ['both', 'train']:
        train_loader = DataLoader (
            MNISTDataset (
                mode='training',
                additional_transforms=additional_transforms),
            batch_size=batch_size,
            shuffle=True,
            **kwargs)

    # Validation mode
    if mode in ['both', 'test']:
        test_loader = DataLoader (
            MNISTDataset (mode='validation'),
            batch_size=batch_size,
            shuffle=True,
            **kwargs)
        
    # Return the initialized loaders
    return train_loader, test_loader