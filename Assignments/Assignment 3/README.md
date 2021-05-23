# Assignment 3
---

## Objective

Use the images from MNIST along with a random number (integer) generated between 0 to 9 as input and return as outputs the predicted digit (predictions on the image) along with the sum of the values, i.e. Predicted digit from MNIST + Generated Random Number.

---

## Process

To train a model to achieve the object, the data needs to be fetched/downloaded and prepped in the correct format and with the required batch size to be passed for training and validation. 

---

### Data loading and processing
We start by building a class that loads and takes the images from MNIST and a random number (between 0-9) generator to use as input. The actual labels of the handwritten digit and the sum are the required outputs, which the model would ultimately return.

The _Dataset_ abstract class defined under PyTorch's utility function is extended to define our custom dataset loader and the function under PyTorch's _torchvision_ function, _datasets_ is used to download the data and the _transforms_ function is used to apply the required augmentation methods on the dataset. In the current approach the images are normalized with a $\mu=0.1307$ and $\sigma=0.3081$, as suggeested from in this [post](https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457).

To generate the random integer (in the range of 0 to 9), PyTorch's in-built _randint_ is used and then this is one-hot encoded duing the _functional_'s _one_hot_ encode function to represent input integer. Likewise, the actual label of the hardwritten digit, got from MNIST and the sum of the two, which it the other output required is also encoded using the same function. Since our objective limits the random integers generated to the range of 0 to 9, and because MNIST only contains values from 0 to 9, i.e. 10 classes for both, the range of the sum of the two is from 0 (when both are 0) to a maximum of 18 (with both 9). So the _number classes for the resultant sum would be 19_ (0 to 18 inclusive).

These 4 objects, i.e. the 28x28 image, the one-hot encoded random integer, the ground truth (actual) label of the image (one-hot encoded) and the resultant sum on adding the two number (one-hot encoded) forms one sample of the custom dataset that is used.

|![sample_images](../../Images/markdown_images/sample_mnist.png)|
|:---:|
|*Sample images from MNIST with ground truth*|

During training we prefer to have batches of samples rather than single sample and iterating using a simple for loop over the data is not optimal. So we used `DataLoader` method defined in PyTorch's utilities to get the samples of the required batch size and to shuffle the images every batch ([reference](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)).

---

### Build the network architecture
Keeping the objective in mind, utlimately, given an image (of size 28x28) and an integer (in the range of 0-9), the solution needs to return the predicted digit in the image and the sum of the digit and the input integer.

There are two ways in which we could do this

#### <ins>Approach 1 - Train the model in two stages as described below</ins>

- _Stage 1:_ Train/build a classifier (using Convolution layers) to predict the digits given the image

|![stage 1_arch](../../Images/markdown_images/mnist_arch.png)|
|:---:|
|*Stage 1: Network Architecture used to predict the digit trained on MNIST*|


The architecture has a series of convolution and max-pooling blocks to extract the required edges, patterns and parts of objects to finally predict the digit.
    

|![stage_1_traning](../../Images/markdown_images/mnist_training.png)|
|:---:|
|*Stage 1: Training on MNIST images to predict the digit*|


The model was trained for _10 epochs_ with a _batch size of 128_. _SGD with a momentum of 0.9_ was choosen as the optimizer for gradient descent, with the momentum preventing spurious batches from affecting the final model. The batch size of 128 worked better than 64 and 32, and a higher batch size if possible for as the images are of 28x28, thus giving a good representation of all the classes within a batch.

    
- _Stage 2:_ Train another classifier (using Fully Connected layers) with the one-hot encoded random number and the one-hot encoded digit predictions as inputs to get the resultant sum.

|![stage 2_arch](../../Images/markdown_images/sum_arch.png)|
|:---:|
|*Stage 2: Network Architecture used to predict the sum of two integers*|


The network architecture used to calculate the sum of the two numbers is a bit of an over-kill, but it ensures that the results are consistant. This is also the reason why the final output (and the accuracy) is dictated by the predictions of Stage 1.


|![stage 2_training](../../Images/markdown_images/sum_training.png)|
|:---:|
|*Stage 2: Training to calculate the sum of two numbers*|


Since the architecture itself was extreme, it doesn't require much training as seen in the image snippet. The training is done in 2 epochs, but ran for 5 just to be sure.

- These trained models are then used/combined into a consolidated network, which uses the trained models to predict the digit and the sum given the two inputs (as mentioned above)

|![results](../../Images/markdown_images/consolidated_results.png)|
|:---:|
|*Results combining the two models with the actuals (A) and the predictions (P)*|

<br>

#### <ins>Approach 2 - Train a single model to predict the digit and compute the sum</ins>

In this approach, instead of breaking the problem into two seperate models, _a single Convolution Network_ with blocks and Convolution and MaxPooling layers are used to get extract the required edges, gradients, patterns, texts and the sub portions of the object to predict the digit (as in Stage 1 of Approach 1). But instead of building a seperate model to predict the sum of the random integer with the predicted handwritten digit, _the same model is branch out and extended and passed through a set of Fully Connected layers_ (similar to the one in Stage 2 of Approach 1) and the results of the both predictions (i.e. the handwritten digit and the sum) are returned as the output of the forward pass.

The losses by the two sub-networks are combined as suggested in the [post](https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch/53995165). In short, the loss from one branch (i.e. predicting digit) and the loss from the other branch (i.e. to compute the sum) are combined (given equal weight, even this could be parameterized to see which works best instead of 1:1 ratio) to get the final loss on which the gradient changes are computed.

|![loss_img](../../Images/markdown_images/loss_branch.png)|
|:---:|
|*Illustration of branching from the main network*|

**Note:** The image below isn't the network used here, it just to illustrate at any stage of the network it can be forked to go along two different paths for different set of predictions

The network was trained for 15 epochs, with the parameters for learning rate, batch size etc., and even the convolution network being the same as first approach (except the modifications to incorporate the branch at the end).

|![stage 2_training](../../Images/markdown_images/merged_training.png)|
|:---:|
|*Stage 2: Training to calculate the sum of two numbers*|

In Approach 1, the final results of the sum was dependent primarly on the model trained on MNIST (i.e. Stage 1) and not so much on the model for calculating the sum, since the complexity wasn't much.
But this approach of combining both the stages into a single model, by branching it out at the end of convolution gives much better results/accuracy (99.43 was the highest as compared to 99.3 from Approach 1).

|![results](../../Images/markdown_images/merged_results.png)|
|:---:|
|*Results combining the two models with the actuals (A) and the predictions (P)*|

---

_EVA by The School of AI; Thank you Rohan for the intriguing assignments. Looking forward for more :v: :metal:_

---