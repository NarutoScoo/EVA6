# Assignment 5
---

## Objective

Build up to an architecture in multiple stages with the following requirements
- Dataset: <img src='https://render.githubusercontent.com/render/math?math=\text{MNIST}'>
- Model Parameters: <img src='https://render.githubusercontent.com/render/math?math=\le 8000'>
- Dropout: <img src='https://render.githubusercontent.com/render/math?math=0.05'>
- Rotation: <img src='https://render.githubusercontent.com/render/math?math=\pm 6.9^\circ'> 
- Test Accuracy: <img src='https://render.githubusercontent.com/render/math?math=\ge 99.4\text{%25}'> _(should be consistent for atleast 4 epochs)_

The work is shown in 3 stages as mentioned below

- _First Stage:_ Refer to [Assignment 5 - Stage 1. Prepare the data and define a barebone architecture.ipynb](./Assignment%205%20-%20Stage%201.%20Prepare%20the%20data%20and%20define%20a%20barebone%20architecture.ipynb) for the code and the section [Stage 1](#first_stage) for more details

- _Second Stage:_ Refer to [Assignment 5 - Stage 2. Modify the architecture to meet the architectural requiremenets.ipynb](./Assignment%205%20-%20Stage%202.%20Modify%20the%20architecture%20to%20meet%20the%20architectural%20requiremenets.ipynb) for the code and the section [Stage 2](#second_stage) for more details 

- _Third Stage:_ Refer to [Assignment 5 - Stage 3. Tune the parameters to achieve the desired results.ipynb](./Assignment%205%20-%20Stage%203.%20Tune%20the%20parameters%20to%20achieve%20the%20desired%20results.ipynb) for the code and the seciton [Stage 3](#third_stage) for more details

---
## Stage 1: Prepare the data and define a barebone architecture
<a name="first_stage"></a>

In this stage, the data was loaded and explored to compute find the _mean_ and the _standard deviation_, which is required to normalize the image set. The mean (<img src='https://render.githubusercontent.com/render/math?math=\small\mu=0.1307'>) and the standard deviation (<img src='https://render.githubusercontent.com/render/math?math=\small\sigma=0.3081'>) was used to normalize the training and test sets.

Looking further at the sample images, we see that thickness of given digit vary across images of the same digit.

<div align='center'>
<a name='thickness_original'></a>

|![thickness_original](../../Images/markdown_images/thickness_original.png)|
|:---:|
|_Varying of thickness of the digit_|
</div>

An augmentation method such as _Morphological Dilation and Erosion_ could be used to help the model learn these variation better while training. _Dilation_  increase the width/thickness of the digit and the _Erosion_ reduces the thickness of the digit (_Reference:[OpenCV Docs](https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html)_). 

<div align='center'>
<a name='thickness_morphological'></a>

|![thickness_dilation](../../Images/markdown_images/thickness_dilation.png)|![thickness_erosion](../../Images/markdown_images/thickness_erosion.png)|
|:---:|:---:|
|_Morphological Dilation_|_Morphological Erosion_|

<sub><em>Note:These operations are applied randomly with the probablity of 0.5</em></sub>
</div>

Additionally, further explorations of the images were done to find the optimal size of the Receptive Field (RF) to aim for, which would potentially reduce the number of training paramters required. It was seen that the size of the images in MNIST is _28x28_ grayscale images and since the size is small a smaller RF would be sufficient to detect the required edges/gradients, patterns/textures and so on.

<div align='center'>
<a name='digit_image'></a>


|![digit_image](../../Images/markdown_images/digit_image.png)|
|:---:|
|_Image of a digit, with the pixels marked along the horizontal and vertical axis_|

</div>

Based on the region highlighted in the [image](#digit_image) and the [plot](#plot) showing the distribution of the distance between the highest intensity pixel and lowest intensity pixel along both horizontal and vertical directions shows an RF of _5x5_ or _7x7_ would be sufficient for each convolution block (1000 images were randomly sampled from MNIST).

<div align='center'>
<a name='plot'></a>

|![digit_distribution](../../Images/markdown_images/digit_distribution.png)|
|:---:|
|_Distribution of the pixel width/height across a gradient_|

</div>

Once the dataloader was defined, a barebone network was written to check the flow of the operation performed so far and to act as a skeleton for further stages where the model parameters, the augmentation stategies, learning rate, etc., will be tuned.

<div align='center'>
<a name='barebone_network'></a>


|![barebone_network](../../Images/markdown_images/barebone_network.png)|
|:---:|
|_Barebone Network_|

<sub><em>Note: The number of parameters is quite high, this will be reduce to the meet the objective in the next stage</em><sub>
</div>
 


---
## Stage 2: Modify the architecture to meet the architectural requiremenets
<a name="second_stage"></a>
    
**Target**
- Refine the model architecture to meet the specifications
    - Parameters should be <img src='https://render.githubusercontent.com/render/math?math=\le 8000'>
    - Use _Dropout_ of <img src='https://render.githubusercontent.com/render/math?math=0.05'>
- Prevent fluctuations in the test accuracy
    - Using _BatchNormalization_ might help stabalize
- Reduce overfitting (i.e. training and test/validation accuracy should be vary proportionally)

**Results**
- Best Training Accuracy: _99.02_ (Iteration 14)
- Best Test Accuracy: _99.30_ (Iteration 12)
- Model Parameters: _7,940_

**Analysis**
- The number of model parameters required has been reduced substantially, which would reduce the training time.
- It no longer overfits (rather its a bit underfit)
- Perhaps using a higher learning rate with a LR scheduler should help the model learn better with additional augmentation methods incorporated as well (refer to Stage 3)


The network build had 7940 parameters as shown below.

<div align='center'>
<a name='stage_2_net'></a>


|![stage_2_net](../../Images/markdown_images/stage_2_network.png)|
|:---:|
|_Network with lesser than 8000 parameters_|

</div> 
    
    
<div align='center'>
<a name='stage_2_plot'></a>


|![stage_2_plots](../../Images/markdown_images/stage_2_plots.png)|
|:---:|
|_Accuracy and Loss on Training and Test_|

</div> 
    
It wasn't able to reach the desired 99.4 or higher accuracy rate on the test set, which would be the objective of Stage 3.
    

---
## Stage 3: Tune the parameters to achieve the desired results
<a name="third_stage"></a>
    
The network is similar to the [Stage 2](#stage_2_net).

**Target**
- Use the refined model (without any changes)
- Add Augmentation methods so it generalizes better
    - Rotation of $\pm6.9^\circ$ was used
- Use LR Scheduler to regulate the learning rate during training
- Get the model consistantly over 99.4% accuracy on test set

**Results**
- Best Training Accuracy: _99.17_ (Iteration 14)
- Best Test Accuracy: _99.48_ (Iteration 14)
    - Consistantly over 99.4 from Iteration 10
- Model Parameters: _7,940_

**Analysis**
- Learning Rate Schedulers were used to start the learning faster (with higher initial LR) to quickly find the region of global minima and then reduce the Learning Rate, so that it converges to the minima around that region, thus avoiding falling into the trap of local minima.
- Training accuracy is still not as high as possible; there's still room for improving the results further
    
    
<div align='center'>
<a name='stage_3_plot'></a>


|![stage_3_plots](../../Images/markdown_images/stage_3_plots.png)|
|:---:|
|_Accuracy and Loss on Training and Test_|

    
</div>
    

The model was able to consistantly get an accuracy score of over 99.4 for the last few epochs. Training further can perhaps help improve it further as the highest training accuracy is only 99.17.
    
    
<div align='center'>
<a name='stage_3_resuts'></a>


|![stage_3_results](../../Images/markdown_images/stage_3_results.png)|
|:---:|
|_Accuracy and Loss on Training and Test_|

    
</div>

---
## Future Scope

- Explore additional augmentation methods
- There's lots of room to train more, need to explore on how that can be leveraged without increasing the total parameters of the model
    - Perhaps having an even higher initial learning rate might help?
    
---