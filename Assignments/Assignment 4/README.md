# Assignment 4
---

## Objective

<ins>**Part A**</ins>

Train a Fully Connected Neural Network (on a Spreadsheet) showing the gradient computations and the losses as the learning rate varies over epochs.

<ins>**Part B**</ins>

The task is to build a Convolution Neural Network to predict the digits in MNIST, with the following constraints

- Should achieve an accuracy of 99.4% or higher on the validation/test set
- The network should have less than 20k parameters
- Should stablize the model within 20 epochs
- And the network should defined should use BatchNormalization, Dropout and Global Average Pooling (GAP) followed by Fully Connected Layers.

---

## Part A - Training a FC NN on a Spreadsheet

A fully connected network with 2 inputs, 2 hidden layers and 2 outputs is simulated using a spreadsheet (Libre Calc was used).

There are 3 sheets in the workbook
- _Network Architecture and Global Definitions:_ Defines the network structure and shows the calculation for forward pass and the gradient computation required for backward pass.

<p align='center'>

|![fc_1](../../Images/markdown_images/network_fc_1.png)|![fc_2](../../Images/markdown_images/network_fc_1.png)|
|:---:|:---:|
|_Network Architecture - Part 1_|_Network Architecture - Part 2_|

</p>    
    
    
    
<p align='center'>

|![fc_3](../../Images/markdown_images/network_fc_3.png)|![fc_4](../../Images/markdown_images/network_fc_4.png)|
|:---:|:---:|
|_Network Architecture - Part 3_|_Network Architecture - Part 4_|

</p>

- _Training:_ The formulae derived in Sheet 1 is used on a sample data and run for 2000 epochs. This simulates the training process with 2000 epochs.

<p align='center'>

|![train_1](../../Images/markdown_images/train_fc_1.png)|![train_2](../../Images/markdown_images/train_fc_2.png)|
|:---:|:---:|
|_Training - Part 1_|_Training - Part 2_|

</p>

<p align='center'>

|![train_3](../../Images/markdown_images/train_fc_3.png)|![train_4](../../Images/markdown_images/train_fc_4.png)|
|:---:|:---:|
|_Training - Part 3_|_Training - Part 4_|

</p>

- _Plots:_ This sheet contains the plots for the _Total Loss_ and the _Both the Predictions_.

<p align='center'>

|![lr_01](../../Images/markdown_images/lr_0.1.png)|![lr_02](../../Images/markdown_images/lr_0.2.png)|
|:---:|:---:|
|_Training with Learning Rate $\eta=0.1$_|_Training with Learning Rate $\eta=0.2$_|

</p>

<p align='center'>

|![lr_08](../../Images/markdown_images/lr_0.8.png)|![lr_2](../../Images/markdown_images/lr_2.0.png)|
|:---:|:---:|
|_Training with Learning Rate $\eta=0.8$_|_Training with Learning Rate $\eta=2.0$_|

</p>

**Note:** _Please use the drop-down field in Sheet 1 to change the Learning Rate_

---

## Part B - Training a CNN to get 99.4% on MNIST

### Fetch and Explore the data 

First the data was loaded; PyTorch's custom dataset loader was used to get the image and the corresponding labels. The images where normalized with the Mean ($\mu=0.1307$) and Standard Deviation ($\sigma=0.3081$).

The images in MNIST are grayscal with the size of _28x28_. Since the size of the image is small, we need not build a network where the _Receptive Field (RF)_ of each block goes to _11x11_. We could instead use a smaller RF per block to extract the required edges and gradients and pass these to successive blocks to extract the patterns/textures, parts of digits and finally to predict the digit.

<p align='center'>

|![digit_image](../../Images/markdown_images/digit_image.png)|
|:---:|
|_Image of a digit, with the pixels marked along the horizontal and vertical axis_|

</p>

To verify the claim, 1000 images were taken at random from MNIST and the distance/number of pixels required to find the edges/gradients was computed. This would help reduce the number of layers required and thus the parameters required to train (and the time taken as well).

<p align='center'>

|![digit_distribution](../../Images/markdown_images/digit_distribution.png)|
|:---:|
|_Distribution of the pixel width/height across a gradient_|

</p>

**We see from that the peak of the curve being at around 5-7 substantiates our claim that a Receptive Field of _5x5_ or _7x7_ is ideal for MNIST.**

---

### Build the network

The network can be built considering the observations described above. The final architecture start with a block of Convolution Layers with Batch Normalization applied after each. Since at each iteration over a batch different set of images can be passed; _Batch Normalization_ helps to standardize (by normalizing the images in the batch), so that the weights learned are consistant across the batch, without resulting in sporadic updates. In addition, a dropout of _0.008_ was used  along with 1x1 convolution to summarize the channels after each block.


<p align='center'>

|![mnist_network_99_4](../../Images/markdown_images/mnist_network_99_4.png)|
|:---:|
|_Network Architecture_|

</p>

The block was restricted to a RF of _5x5_ as it was sufficient to predict the digits and the biases where removed to reduce the number of parameters further.

After defining the architecture, **Stocastic Gradient Descent (SGD)** was chosen as the optimizer to use to update the weights based on gradient descent values, with the initial **Learning Rate of  $\eta=0.05$** instead of 0.01. During previous iterations, it was noted that the model began to overfit after **Epochs 6 and 11-13**. To prevent this, **MultiStepLR** scheduler was used to reduce the learning rate at different epochs periods (_6, 13 and 17_). In addition, **ReduceLROnPlateau** was also used as a fail safe if validation loss increased or stopped descreasing (plateaued); there by chaining the schedulers (_References: [1](https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling) | [2](https://pytorch.org/docs/master/optim.html) | [3](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)_).

<p align='center'>

|![multistep_lr](../../Images/markdown_images/mnist_99_4_multistep_lr.png)|
|:---:|
|_Drop in the Learning Rate while using MultiStepLR_|

</p>


The network was trainined for 20 epochs, with the validaiton accuracy going over the targeted 99.4 from epoch 17 onwards.

<p align='center'>

|![mnist_99_4_training](../../Images/markdown_images/mnist_99_4_training.png)|
|:---:|
|_Last few epochs showing the validation/test accuracy_|

</p>

---

### Future Scope

- The model has overfitted, perhaps adding/increasing regularization parameter might help.
- Number of parameters used is just below 20K, perhaps a different architecture where the first block has more parameters (going with 64 channels) are reducing the number of blocks to 2 instead of 3 could be tried

---