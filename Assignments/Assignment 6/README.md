# Assignment 6
---

## Objective

### Part A

**<ins>Target</ins>**

Try out different normalizations methods and analyze which performs better
- _[Batch Normalizaiton](#bn_exp)_
- _[Group Normalization](#gn_exp)_
- _[Layer Normalization](#ln_exp)_
- _[Instance Normalization](#in_exp)_

Also use _L1_ and _L2_ talong with the normalization methods and tabulate the findings


**<ins>Inference</ins>**

- Based on the plots on [accuracies](#acc_with_in) and [loss](#loss_with_in) comparing Batch, Group, Layer and Instance Normalizaiton, we see instance normalization underperfoming as compared to the other normalization methods
- Comparing the [accuracies](#acc_wo_in) and [loss](#loss_wo_in) of Batch, Group and Layer Normalization (i.e. without Instance Norm), the best accuracy (and correspondingly the least loss) is seen while using _Group Normalization_, followed by Batch Norm and then Layer Norm
    - Although they perform equally well (with a margin of 1-1.5% variation in accuracy), the effect could be due to the difference in regularization methods used
- Training setup used the same [model architecture](./model.py) with the following changes
    - For BN, both L1 and L2 Regularization was used along with a dropout of 0.01
    - For GN, L1 Regularization was used along with a dropout of 0.01
    - For LN, L2 Regularization was use along with a dropout of 0.01
    - For IN, both L1 and L2 Regularization was used along with a dropout of 0.01
- The misclassified images for each of the models are shown in their corresponding sections; In these images, the digits are missing portions or its hard to read
    - [Misclassified Images - Batch Norm](#bn_miss)
    - [Misclassified Images - Group Norm](#gn_miss)
    - [Misclassified Images - Layer Norm](#ln_miss)
    - [Misclassified Images - Instance Norm](#in_miss)
   
<sub><em> Note: Instance Norm isn't part of the requirements, but explored here for better understanding </em></sub>

### Part B

**<ins>Target</ins>**

Show how each of the normalization i.e. Batch Norm, Group Norm and Layer Norm are computed on _4_ sample images of size _2x2_.
The calculation are shown at the [Appendix](#appendix).

---
<a name='bn_exp'></a>
## Experiment 1: Using Batch Normalization with L1 and L2 Regularization
The model architecture defined in [model.py](./model.py), is the same for all the experiments, with the variation being the normalization method used and the the regularization loss applied during the loss calculation.

<div align='center'>
<a name='bn_net'></a>


|![bn_net](../../Images/markdown_images/bn_net.png)|
|:---:|
|_Network Architecture using Batch Normalization_|

</div>

L1 Norm (<img src='https://render.githubusercontent.com/render/math?math=\large{L_1 \text{Loss} = \Sigma ||w||}'>) and L2 Norm (<img src='https://render.githubusercontent.com/render/math?math=\large{L_2 \text{Loss} = \Sigma ||w||^2}'>) are computed manually and added to the training loss (that acts as a regularization parameter), which is then backpropogated. This forces the network to training against the added constraints thus making it better at generalizing.

The final loss used for backpropogation in the sum of the corresponding losses

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large{\text{Loss}_\text{total} = \text{Loss}_\text{training} %2B \lambda_1*L_1 %2B \lambda_2*L_2}'>
</div>
    
The training and test performance of the model is as shown and a sample of the misclassified images are presented as well.
    
    
<div align='center'>
<a name='bn_train'></a>

|![bn_train](../../Images/markdown_images/bn_train.png)|
|:---:|
|_Accuracy and Loss on Training Set_|
</div>

<div align='center'>
<a name='bn_test'></a>

|![bn_test](../../Images/markdown_images/bn_test.png)|
|:---:|
|_Accuracy and Loss on Test Set_|
</div>

**Note:** _[Consolidated results](#consolidate) are at the end_

---
<a name='gn_exp'></a>
## Experiment 2: Using Group Normalization with L1 Regularization

The network architecture (defined in [model.py](#model.py)) is used here, which is simlar in all the experiments. 

<div align='center'>
<a name='gn_net'></a>


|![gn_net](../../Images/markdown_images/gn_net.png)|
|:---:|
|_Network Architecture using Group Normalization_|

</div>

For this network, only _L1 Regularization_ was used and the final computed loss function is as shown

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large{\text{Loss}_\text{total} = \text{Loss}_\text{training} %2B \lambda_1 * L_1}'>
</div>

The training and test performance of the model is as shown and a sample of the misclassified images are presented as well.
       
<div align='center'>
<a name='gn_train'></a>

|![gn_train](../../Images/markdown_images/gn_train.png)|
|:---:|
|_Accuracy and Loss on Training Set_|
</div>

<div align='center'>
<a name='gn_test'></a>

|![gn_test](../../Images/markdown_images/gn_test.png)|
|:---:|
|_Accuracy and Loss on Test Set_|
</div>

We see from the results that group norms predictions are consistant across epochs (i.e. it doesn't fluctuate a lot), which is a good for model stability.

**Note:** _[Consolidated results](#consolidate) are at the end_

---
<a name='ln_exp'></a>
## Experiment 3: Using Layer Normalization with L2 Regularization

Similar to the models using Batch Norm and Group Norm, this network architecture (defined in [model.py](#model.py)) is the same except for the use of Layer Normalization. Normalization was done across `(Channel, Height, Width)`, which adds significant number of parameters (increase from 7940 to ~65K). 

<div align='center'>
<a name='ln_net'></a>

|![ln_net](../../Images/markdown_images/ln_net.png)|
|:---:|
|_Network Architecture using Layer Normalization_|
</div>

Additional _L2 Regularization_ was used to help generalize better.

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large{\text{Loss}_\text{total} = \text{Loss}_\text{training} %2B \lambda_2 * L_2}'>
</div>

The training and test performance of the model is as shown and a sample of the misclassified images are presented as well.
       
<div align='center'>
<a name='ln_train'></a>

|![ln_train](../../Images/markdown_images/ln_train.png)|
|:---:|
|_Accuracy and Loss on Training Set_|
</div>

<div align='center'>
<a name='ln_test'></a>

|![ln_test](../../Images/markdown_images/ln_test.png)|
|:---:|
|_Accuracy and Loss on Test Set_|
</div>

**Note:** _[Consolidated results](#consolidate) are at the end_

---
<a name='in_exp'></a>
## Additional Experiment: Using Instance Normalization with L1 and L2 Regularization

In addition to the above experiments using _Batch Normalization_, _Group Normalization_ and _Layer Normalization_, **Instance Normalization** was also tried out to see how well it stacks up against other methods.

<div align='center'>
<a name='in_net'></a>


|![in_net](../../Images/markdown_images/in_net.png)|
|:---:|
|_Network Architecture using Instance Normalization_|
</div>

Both _L1 Regularization_ and _L2 Regularization_ was applied during training and the final results are as shown.

<div align='center'>
<a name='in_train'></a>

|![in_train](../../Images/markdown_images/in_train.png)|
|:---:|
|_Accuracy and Loss on Training Set_|
</div>

<div align='center'>
<a name='in_test'></a>

|![in_test](../../Images/markdown_images/in_test.png)|
|:---:|
|_Accuracy and Loss on Test Set_|
</div>

---

<a name='consolidate'></a>
## Consilated Results and Inference

Comparing all the models together and based on the plots on [accuracies](#acc_with_in) and [loss](#loss_with_in), we se that _Instance Normalization_ doesn't perform as well as other normalization methods.

<div align='center'>
<a name='acc_with_in'></a>

|![train_acc_with_in](../../Images/markdown_images/train_acc_with_in.png)|![train_loss_with_in](../../Images/markdown_images/test_acc_with_in.png)|
|:---:|:---:|
|_Training Accuracies (showing all 4 models)_|_Test Accuracies (showing all 4 models)_|
</div>


<div align='center'>
<a name='loss_with_in'></a>

|![train_acc_with_in](../../Images/markdown_images/train_loss_with_in.png)|![train_loss_with_in](../../Images/markdown_images/test_loss_with_in.png)|
|:---:|:---:|
|_Training Loss (showing all 4 models)_|_Test Loss (showing all 4 models)_|
</div>

Now to compare the performance of the models required for the assignment, i.e. Batch Norm, Group Norm and Layer Norm, the best accuracy (and correspondingly the least loss) is seen while using _Group Normalization_, followed by _Batch Norm_ and then _Layer Norm_

<div align='center'>
<a name='acc_wo_in'></a>

|![train_acc_wo_in](../../Images/markdown_images/train_acc_wo_in.png)|![train_loss_wo_in](../../Images/markdown_images/test_acc_wo_in.png)|
|:---:|:---:|
|_Training Accuracies (Batch Norm, Group Norm and Layer Norm)_|_Test Accuracies (Batch Norm, Group Norm and Layer Norm)_|
</div>


<div align='center'>
<a name='loss_wo_in'></a>

|![train_acc_wo_in](../../Images/markdown_images/train_loss_wo_in.png)|![train_loss_wo_in](../../Images/markdown_images/test_loss_wo_in.png)|
|:---:|:---:|
|_Training Loss (Batch Norm, Group Norm and Layer Norm)_|_Test Loss (Batch Norm, Group Norm and Layer Norm)_|
</div>


Although they perform equally well (with a margin of 1-1.5% variation in accuracy), the effect could be due to the difference in regularization methods used



The misclassified images for each of the models are shown in their corresponding sections

<div align='center'>
<a name='bn_miss'></a>
<a name='gn_miss'></a>

|![bn_miss_img](../../Images/markdown_images/bn_miss_img.png)|![gn_miss_img](../../Images/markdown_images/gn_miss_img.png)|
|:---:|:---:|
|_Images misclassified by Network using Batch Normalization_|_Images misclassified by Network using Group Normalization_|
</div>


<div align='center'>
<a name='ln_miss'></a>
<a name='in_miss'></a>

|![ln_miss_img](../../Images/markdown_images/ln_miss_img.png)|![in_miss_img](../../Images/markdown_images/in_miss_img.png)|
|:---:|:---:|
|_Images misclassified by Network using Layer Normalization_|_Images misclassified by Network using Instance Normalization_|
</div>

The images misclassified by the models are the digits that are missing portions or its hard to read (as its blurry or too dilated).

---
<a name='appendix'></a>
## Appendix

Consider the batch of _4_ images of size _2x2_ as shown in the sample.

<div align='center'>
<a name='sample_2x2'></a>

|![sample_2x2](../../Images/markdown_images/sample_2x2.png)|
|:---:|
|_Batch of 4 sample images_|
</div>

**Batch Normalization**

The mean (<img src='https://render.githubusercontent.com/render/math?math=\small\mu'>) and standard deviation (<img src='https://render.githubusercontent.com/render/math?math=\small\sigma'>) required for _Batch Normalization_ is computed by calculating the mean and standard deviation across all the pixels in the blocks colored the same (as seen in the [figure](#sample_2x2)).

The calculation for one of the channels (_Blue Channel_) is shown and the same can be applied for other channels as well.

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\Large\mu=\frac{\text{Image}_1\text{Channel}_{\text{Blue}}%2B\text{Image}_2\text{Channel}_{\text{Blue}}%2B\text{Image}_3\text{Channel}_{\text{Blue}}%2B\text{Image}_4\text{Channel}_{\text{Blue}}}{\text{Batch Size}*\text{Image Width}*\text{Image Height}}'>
    
</div>

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large=\frac{(-3 %2B -2 %2B 1 %2B -1) %2B (-1 %2B 2 %2B -1 %2B 2) %2B (-3 %2B 1 %2B 0 %2B 3) %2B (3 %2B -3 %2B -1 %2B -3)}{4*2*2}'>
    
    <img src='https://render.githubusercontent.com/render/math?math=\large=\frac{(-5) %2B (2) %2B (1) %2B (-4)}{16}'>
</div>    
    
<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large\mu=-0.375'>
</div>

Similarly Standard Deviation can be shown be calculated to be 

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large\sigma=\sqrt{4.359}'>
</div>

The calculated results for all the channels are shown.
<div align='center'>
<a name='batch_norm_calc'></a>

|![batch_norm_calc](../../Images/markdown_images/batch_norm_calc.png)|
|:---:|
|_Computed Mean and Variance per channel for Batch Normalization_|
</div>

**Group Normalization**

The mean (<img src='https://render.githubusercontent.com/render/math?math=\small\mu'>) and standard deviation (<img src='https://render.githubusercontent.com/render/math?math=\small\sigma'>) required for _Group Normalization_ is computed by calculating the mean and standard deviation across all the pixels in the blocks with the dashed rectangle (as seen in the [figure](#sample_2x2)).

The calculation for one of the groups (_Blue Group_) is shown and the same can be applied for other groups as well.

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\Large\mu=\frac{\text{Image}_1\text{Channel}_{\text{Blue}}%2B\text{Image}_1\text{Channel}_{\text{Green}}}{\text{No. of Channels in Group}*\text{Image Width}*\text{Image Height}}'>
    
</div>

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large=\frac{(-3 %2B -2 %2B 1 %2B -1) %2B (2 %2B -2 %2B 2 %2B 1)}{2*2*2}'>
    
    <img src='https://render.githubusercontent.com/render/math?math=\large=\frac{(-5) %2B (3)}{16}'>
</div>    
    
<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large\mu=0.25'>
</div>

Similarly Standard Deviation can be shown be calculated to be 

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large\sigma=2.938'>
</div>

The calculated results for all the channels are shown.
<div align='center'>
<a name='group_norm_calc'></a>

|![group_norm_calc](../../Images/markdown_images/group_norm_calc.png)|
|:---:|
|_Computed Mean and Variance per group for Group Normalization_|
</div>


**Layer Normalization**

The mean (<img src='https://render.githubusercontent.com/render/math?math=\small\mu'>) and standard deviation (<img src='https://render.githubusercontent.com/render/math?math=\small\sigma'>) required for _Layer Normalization_ is computed by calculating the mean and standard deviation across all the pixels marked with the solid red box (as seen in the [figure](#sample_2x2)).

The calculation for one of the layers is shown and the same can be applied for other layers as well.

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\Large\mu=\frac{\text{Image}_1\text{Channel}_{\text{Blue}}%2B\text{Image}_1\text{Channel}_{\text{Green}}%2B\text{Image}_1\text{Channel}_{\text{Yellow}}%2B\text{Image}_1\text{Channel}_{\text{Brown}}}{\text{No. of Channels}*\text{Image Width}*\text{Image Height}}'>
    
</div>

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large=\frac{(-3 %2B -2 %2B 1 %2B -1) %2B (2 %2B -2 %2B 2 %2B 1) %2B (3 %2B -3 %2B 2 %2B 0) %2B (2 %2B 0 %2B 3 %2B -3)}{4*2*2}'>
    
    <img src='https://render.githubusercontent.com/render/math?math=\large=\frac{(-5) %2B (3) %2B (2) %2B (2)}{16}'>
</div>    
    
<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large\mu=0.125'>
</div>

Similarly Standard Deviation can be shown be calculated to be 

<div align='center'>
    <img src='https://render.githubusercontent.com/render/math?math=\large\sigma=\sqrt{4.484}'>
</div>

The calculated results for all the channels are shown.
<div align='center'>
<a name='layer_norm_calc'></a>

|![layer_norm_calc](../../Images/markdown_images/layer_norm_calc.png)|
|:---:|
|_Computed Mean and Variance per layer (image) for Layer Normalization_|
</div>

The values are used to _Normalize_ the batch of images and _Scale_ and _Shift_ using the learned <img src='https://render.githubusercontent.com/render/math?math=\large\gamma'> and <img src='https://render.githubusercontent.com/render/math?math=\large\beta'> parameters.

---
## References

1. Wu, Y. and He, K., 2018. Group normalization. In Proceedings of the European conference on computer vision (ECCV) (pp. 3-19
2. Future, C. T. B. (2020, August 9). Group Normalization. Committed towards Better Future. https://amaarora.github.io/2020/08/09/groupnorm.html







