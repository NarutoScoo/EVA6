# This file `utils.py` contains utility functions/classes
# Extending this to load CIFAR 10

################# Imports #################

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


################# Custom DataLoader #################

class CustomDataset (Dataset):
    ''' Extending the Dataset class to provide customized inputs and modified 
    outputs for training '''

    def __init__(self, mode='training', dataset='MNIST', additional_transforms=None, mean_std=([0.1307,], [0.3081,])):
        ''' Get/Download the data (MNIST/CIFAR-10) and setup up other configurations
        required to get the data
        
        Parameters:
            mode                  - Indicates the mode of operation 
                                        (Values: [`training`, `validation`]; Default: `training`)
            dataset               - Select the dataset to load
                                        (Values: [`MNIST`, `CIFAR10`]; Default: `MNIST`)
            additional_transforms - A list of additional transformations to be added while training
                                        (Default: `None`)
                                        Note: Normalization of the dataset is applied always
            mean_std              - Mean and Standard Deviation to use to normalize the data
                                        (Default: `([0.1307,], [0.3081,])])`) # Mean and Std for MNIST
        '''

        # Start by initializing the base class
        super().__init__()

        # Save the mode
        # Check if the mode is valid
        assert mode in ['training', 'validation'], \
            '"mode" should be either "training" or "validation"'
        self.mode = mode

        # Check the dataset to load is supported
        assert dataset in ['MNIST', 'CIFAR10'], \
            '`dataset` only supports from `MNIST` or `CIFAR-10`'

        # Choose the dataset source
        if dataset == 'MNIST':
            dataset_fn = datasets.MNIST
        elif dataset == 'CIFAR10':
            dataset_fn = datasets.CIFAR10

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
            transforms.Normalize(mean_std [0], mean_std [1]),   
        ]

        # Split the additional transformation into torchvision transforms 
        # and albumentation transforms as they differ in the call
        self.albumentations = None
        if additional_transforms:
            self.albumentations = additional_transforms.get ('albumentation', None)
            additional_transforms = additional_transforms.get ('torchvision', None)

        # Get the data
        if self.mode == 'training':        

            # Chain the transforms
            if additional_transforms:
                self.transforms = additional_transforms+self.transforms
        
            # Download the training set and store it at the root directory
            self.data = dataset_fn (
                root='./data/train/',
                train=True,
                download=True,
                transform=transforms.Compose (self.transforms))
            
        elif self.mode == 'validation':

            # Get the validation set
            self.data = dataset_fn (
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

        # Apply additional augmentations
        if self.albumentations is not None:
            img = self.albumentations (image=img.numpy ()) ['image']

        # Return the training set and the corresponding label
        return img, label

    
################# Custom Data Augmentation #################
    
class RandomErosion ():
    ''' Erosion helps to reduce the thickness of the digit 
    Refer to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
    on creating custom transformations
    '''

    def __init__ (self, kernel_size=3):
        ''' Initialize the transformation parameters '''

        # Check if the kernel_size is of type of int or tuple
        assert isinstance (kernel_size, (int, tuple)), \
            'Kernel Size (`kernel_size`) should be either be an integer or a tuple'

        # Convert to a tuple in case of int
        kernel_size = (kernel_size, kernel_size) if isinstance (kernel_size, int) else kernel_size

        # Define the kernel of the required size
        self.kernel = np.ones (kernel_size, dtype=np.uint8)

    
    def __call__ (self, sample):
        ''' This method is called when the samples are passed to the transformation object '''
        
        # Get the image and its corresponding label from the sample tuple
        # Check if the sample passed is just an image
        # or tuple containing the labels as well
        if isinstance (sample, tuple):
            image, label = sample
        else:
            image = sample

        # Do this at random
        if not np.random.randint (0, 2):
            # Return the image as is
            return image#, label 

        # Convert the image to a numpy array
        if not isinstance (image, np.ndarray):
            # Resize and convert to numpy array
            image = image.view (28, 28).numpy ()

        # Perform erosion using the defined kernel
        transformed_image = cv2.erode (image, self.kernel)

        # Return the transformed image with the label
        return torch.from_numpy (transformed_image).resize (1, 28, 28)#, label
    
    
class RandomDilation ():
    ''' Dilation helps to increase the thickness of the digit 
    Refer to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
    on creating custom transformations
    '''

    def __init__ (self, kernel_size=2):
        ''' Initialize the transformation parameters '''

        # Check if the kernel_size is of type of int or tuple
        assert isinstance (kernel_size, (int, tuple)), \
            'Kernel Size (`kernel_size`) should be either be an integer or a tuple'

        # Convert to a tuple in case of int
        kernel_size = (kernel_size, kernel_size) if isinstance (kernel_size, int) else kernel_size

        # Define the kernel of the required size
        self.kernel = np.ones (kernel_size, dtype=np.uint8)


    def __call__ (self, sample):
        ''' This method is called when the samples are passed to the transformation object '''

        # Get the image and its corresponding label from the sample tuple
        # Check if the sample passed is just an image
        # or tuple containing the labels as well
        if isinstance (sample, tuple):
            image, label = sample
        else:
            image = sample

        # Do this at random
        if not np.random.randint (0, 2):
            # Return the image as is
            return image#, label 

        # Convert the image to a numpy array
        if not isinstance (image, np.ndarray):
            # Resize and convert to numpy array
            image = image.view (28, 28).numpy ()

        # Perform erosion using the defined kernel
        transformed_image = cv2.dilate (image, self.kernel)

        # Return the transformed image with the label
        return torch.from_numpy (transformed_image).resize (1, 28, 28)#, label
    
    
################# Functions #################

def load_samples (dataset='MNIST'):
    ''' Load a sample (training set) to understand the data 

    Parameters:
        dataset     - The dataset to load
                        (Values: ['MNIST', 'CIFAR10'];
                         Default: 'MNIST')
    
    Return:
        samples     - Sample of the data downloaded
    '''

    # Check the dataset to load is supported
    assert dataset in ['MNIST', 'CIFAR10'], \
        '`dataset` only supports from `MNIST` or `CIFAR-10`'

    # Choose the dataset source
    if dataset == 'MNIST':
        dataset_fn = datasets.MNIST
    elif dataset == 'CIFAR10':
        dataset_fn = datasets.CIFAR10

    # Fetch the data
    samples = dataset_fn (
        root='./data/sample/',
        train=True,
        download=True,
        transform=transforms.Compose ([transforms.ToTensor ()])
    )

    # Convert to a tensor if it isn't
    if not isinstance (samples.data, torch.Tensor):
        samples.data = torch.from_numpy (samples.data)

    # Convert it to range [0, 1] if it hasn't been done
    if isinstance (samples.data, torch.ByteTensor):
        samples.data = samples.data / 255

    # Return the sample
    return samples


def describe_data (samples):
    ''' Used to describe the statistics of the dataset

    Parameters:
        data     - Dataset on which the statistics needs to be shown
    '''

    print ('\n\n'+'-'*50)
    print ('No. of samples:', len (samples))
    print ('Size of the sample:', samples.data.shape)
    print ('Minimum value:', samples.data.min ())
    print ('Maximum value:', samples.data.max ())
    print ('Mean of the sample:', torch.mean (samples.data, axis=[0,1,2]))
    print ('Standard Deviation of the sample:', torch.std (samples.data, axis=[0,1,2]))
    print ('No. of classes:', len (samples.classes))
    print ('Class Names:', samples.classes)
    print ('-'*50, end='\n\n\n')


def load_train_test (mode='both', 
                     dataset='MNIST',
                     additional_transforms=None, 
                     batch_size=128,
                     mean_std=([0.1307,], [0.3081,]), # Mean and std for MNIST
                     **kwargs):
    ''' Load the training or the test set 
    
    Parameters:
        mode                  - Load the dataset for training and/or test 
                                    (Values: [`both`, `train` or `test`]; Default: `both`)
        dataset               - Select the dataset to load
                                    (Values: [`MNIST`, `CIFAR10`]; Default: `MNIST`)
        additional_transforms - Additional transformation to be applied while training
                                    (Default: `None`)
        batch_size            - No. of samples per batch (Default: `128`)
        mean_std              - Mean and standard deviation to use for normalization
                                    (Default: ([0.1307,], [0.3081,]))
        
    Return:
        train_loader          - Data loader defined on the training set ('None' based on the `mode`)
        test_loader           - Data loader defined on the test set ('None' based on the `mode`)
    ''' 
    
    # Check the passed values
    assert mode in ['both', 'train', 'test'], \
        '`mode` should be either `both`, `train` or `test`'
    
    # Check the dataset to load is supported
    assert dataset in ['MNIST', 'CIFAR10'], \
        '`dataset` only supports from `MNIST` or `CIFAR-10`'
        
    # Define a dataloader 
    # Set the batch_size and shuffle to select
    # random images for every batch and epoch
    
    # Set the loaders to None and assign based on the selected mode
    train_loader, test_loader = None, None
    
    # Training mode
    if mode in ['both', 'train']:
        train_loader = DataLoader (
            CustomDataset (
                mode='training',
                dataset=dataset,
                additional_transforms=additional_transforms,
                mean_std=mean_std
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs)

    # Validation mode
    if mode in ['both', 'test']:
        test_loader = DataLoader (
            CustomDataset (
                mode='validation',
                dataset=dataset,
                mean_std=mean_std
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs)
        
    # Return the initialized loaders
    return train_loader, test_loader