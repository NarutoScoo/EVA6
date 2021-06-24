# This file `visualize.py` has functions to plot the results

# Imports
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use ('dark_background') # Dark mode rocks!!!


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
                            (Values: [`image`, `plot`]; Default: `image`)
        
    '''
    
    # Get the rows and cols
    n_rows, n_cols = nrows_ncols [0], nrows_ncols [1]

    # Get the size of the figure
    if not figsize:
        figsize = (10*n_rows, 3*n_cols)

    # Create a figure
    fig = plt.figure (figsize=figsize)
    fig.suptitle (fig_title, fontsize=20)
    
    grid = ImageGrid (
        fig, 111, 
        nrows_ncols=(n_rows, n_cols),
        axes_pad=0.5
    )
    
    # Use the subplot title if provided
    if titles:
        assert len (data)==len (titles), 'Image missing a title'
        iterator = zip (grid, data, titles)
    else:
        iterator = zip (grid, data, [None]*len (data))

    # Iterate through the images
    for ax, datum, title in iterator:
        # Display the image or the plot the data
        ax.imshow (datum, cmap='gray') if image_r_plot == 'image' else ax.plot (datum)
        
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