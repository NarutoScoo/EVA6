# This file `visualize.py` has functions to plot the results

################# Imports #################

import cv2
import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use ('dark_background') # Dark mode rocks!!!

# GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


################# Functions #################

def display_activation_map (model, target_layer, input_tensor, 
        misclassified_images, nrows_ncols, figsize, target_category=None):
    ''' Display the Class Activation Map for the selected layer

    Parameters:
        model                   - Trained model
        target_layer            - The layer for which to get the activation map
        input_tensor            - Normalized input images give to the model
        misclassified_images    - The images for which the model failed
        nrows_ncols             - No. of rows and cols
        figsize                 - Size of the figure
        target_category         - The target class of the misclassified image(s)
    '''

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM (model=model, 
        target_layer=target_layer, 
        use_cuda=torch.cuda.is_available()
    )

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    # Iterate through the misclassified image and get the activation map
    list_of_images = []
    for img, cam in zip (misclassified_images, grayscale_cam):
        # Append the img
        list_of_images.append (img)

        # Visualize the activation map
        list_of_images.append (show_cam_on_image(img, cam))

    # Display the images in a grid
    display_results (
        list_of_images,
        fig_title='Misclassified Images with Class Activation Map',
        nrows_ncols=nrows_ncols,
        figsize=figsize
    )
        


def display_misclassified (misclassified_images, fig_title, figsize, nrows_ncols):
    ''' Display misclassified images 

    Parameters:
        misclassified_images    - Misclassified images
        fig_title               - Title of the image grid
        figsize                 - Size of the figure
        nrows_ncols             - No. of rows and cols
    '''

    # Get the no of rows and cols
    nrows, ncols = nrows_ncols [0], nrows_ncols [1]

    # Get a list of random indicies
    random_idx = torch.randint (0, len (misclassified_images), size=(1, nrows*ncols))

    # Get the sample of images to display
    # Unnormalize the images before displaying
    # Also required for GradCAM visualization
    sample_misclassified = []
    for sample in misclassified_images [random_idx].squeeze ().permute (
            0, 2, 3, 1).cpu ().data.numpy ():
        sample_misclassified.append (cv2.normalize (sample, None, norm_type=cv2.NORM_MINMAX))
    sample_misclassified = np.array (sample_misclassified)

    # Dislay the images in a grid
    display_results (
        sample_misclassified,
        figsize=figsize,
        nrows_ncols=nrows_ncols, image_r_plot='image',
        fig_title=f'{fig_title} - Misclassified Images'
    )

    return sample_misclassified, random_idx


def display_samples (data, fig_title, nrows_ncols, 
        figsize=None, image_r_plot='image', label_map=None, titles=None):
    ''' Display samples from the dataset 

    Parameters:
        samples    - Dataset containing the images
    '''

    # Get random image indices to display
    rand_idx = torch.randint (0, len (data), 
        size=(1, nrows_ncols[0]*nrows_ncols[1]))
        
    # Sample the data and the titles
    data = data [rand_idx]
    titles = titles [rand_idx] [0]

    # Check if label map is given
    # if so convert map the titles using it
    if label_map:
        titles = list (map (lambda x: label_map[x].title (), titles))

    # Display the images
    display_results (
        data=data.reshape (data.shape [1:]).numpy (),
        fig_title=fig_title,
        nrows_ncols=nrows_ncols,
        figsize=figsize,
        image_r_plot=image_r_plot,
        titles=titles
    )


def plot_accuracy_loss (accuracy_loss, mode='training'):
    ''' Plot the accuracy and the loss
    
    Parameters:
        accuracy_loss - Tuple containing the accuracy and the loss
        mode          - Mode of operation (i.e. train or test)
                            (Values=[`training`, `test`]; Default: `training`)
    '''
    
    # Define the size of the figure
    plt.figure (figsize=(15, 5))

    # Plot the Accuracy
    plt.subplot (121)
    plt.plot (accuracy_loss [0])
    plt.title (f'{mode.title ()} Accuracy')
    plt.ylabel ('Accuracy'); plt.xlabel ('Epochs');

    # Losses
    plt.subplot (122)
    plt.plot (accuracy_loss [1])
    plt.title (f'{mode.title ()} Loss')
    plt.ylabel ('Loss'); plt.xlabel ('Epochs');
    
    
def display_results (data, fig_title, nrows_ncols, figsize=None, 
        titles=None, image_r_plot='image'):
    ''' Display images/plots in a grid 
    
    Parameters:
        data         - A list of images/plots to display
        fig_title    - Title of the figure
        nrows_ncols  - Number of rows and columns
        figsize      - Size of the figure (Default: `None`)
        titles       - Corresponding list of titles (Default: `None`)
                            x and y labels can be passes as tuples 
                            (Values: [[subplot_title1, xlabel1, ylabel1], [],..]
                                or [subplot_title1, subplot_title2, ...])
        image_r_plot - Indicates whether images are to be displayed or plots
                            (Values: [`image`, `plot`, `hist`]; Default: `image`)
        
    '''
    
    # Get the rows and cols
    n_rows, n_cols = nrows_ncols [0], nrows_ncols [1]

    # Get the size of the figure
    if not figsize:
        figsize = (10*n_rows, 3*n_cols)

    # Create a figure
    fig = plt.figure (figsize=figsize)
    fig.suptitle (fig_title, fontsize=20)
    
    # Use the subplot title if provided
    if titles:
        assert len (data)==len (titles), 'Image missing a title'
        iterator = zip (data, titles)
    else:
        iterator = zip (data, [None]*len (data))

    # Iterate through the images
    for i, (datum, title) in enumerate (iterator, start=1):
        # Create an axis to display
        ax = fig.add_subplot (n_rows, n_cols, i)

        # Display the image or the plot the data
        if image_r_plot == 'image':
            ax.imshow (datum, cmap='gray')

        elif image_r_plot == 'plot':
            ax.plot (datum)

        elif image_r_plot == 'hist':
            sns.distplot (datum, ax=ax)
        
        # Set the title if given
        if title:
            assert isinstance (title, str) or len (title)==3,\
                'Title should have the values: [[subplot_title1, xlabel1, ylabel1], [],..] '+\
                'or [subplot_title1, subplot_title2, ...]'
            
            ax.set_title (title) if isinstance (title, str) else ax.set_title (title [0])
         
        # Set the axis labels
        if isinstance (title, (list, tuple)) and len (title)==3:
            ax.set_xlabel (title [1])
            ax.set_ylabel (title [2])
        else:
            ax.axis ('off')
            
            
def plot_multiple (data_points, fig_title, figsize=None):
    ''' Plot multiple data points in the same plot 
    
    Parameters:
        data_points  - A list of tuples containing the required data points to plot
                            along with the corresponding label
        fig_title    - Title of the figure
        figsize      - Size of the figure
    '''
    # Check for figsize
    if not figsize:
        figsize = (15, 5)

    # Set the figure details
    fig = plt.figure (figsize=figsize)
    fig.suptitle (fig_title, fontsize=20)
    
    # Iteratively plot the data elements
    for datum, label in data_points:        
        plt.plot (datum, label=label)
        
    # Display the legend
    plt.legend ()
    
    # Display the plots
    plt.show ()