# Assignment 8
---

## Objective
Train a model to predict the class in _CIFAR-10_, with the following requirements
- Apply the following augmentations to the data
    - **RandomCrop**
    - **Cutout (<img src="https://render.githubusercontent.com/render/math?math=16x16">)**
    - **Rotation (<img src="https://render.githubusercontent.com/render/math?math=\pm%205^\circ">)**
- No. of Epochs - **40**
- Use **Layer Normalization** instead of Batch Norm
- Use LR Scheduler (**ReduceLROnPlateau**) while training


Once trained check the misclassified images and use **GradCAM** to visualize the layer activations for these misclassified images

---

## Procedure

Define the network architectures to use in a seperate files, so that it can imported into any project/assignment. _ResNet_ architecture defined [here](https://github.com/kuangliu/pytorch-cifar) was cloned and edited. ResNet 18 and 34 is retained and further modifications were done to add _Layer Normalization_, _Group Normalization_ and _Instance Normalization_. The modified version of the file can be found in [resnet.py](../Dependencies/models/resnet.py).

Similar all the training and the test code along with the definition for optimizers and Learning Rate Schedulers are moved into [main.py](../Dependencies/main.py). 

All the data related functions, such as defining a _Custom DataLoader_, _Custom Augmentations (Erosion and Dilation)_; which was used in the previous assignment, sample data loader and function to describe the dataset loaded are defined in [utils.py](../Dependencies/utils.py).

<div align='center'>
<a name='cifar10summary'></a>

|![cifar10summary](../../Images/markdown_images/cifar10summary.png)|
|:---:|
|_Descriptive Statistics of the dataset_|
</div>

We understand from the above its a perfectly balanced dataset, with 5000 samples for each of the 10 classes.

<div align='center'>
<a name='cifar10classdist'></a>

|![cifar10classdist](../../Images/markdown_images/cifar10classdist.png)|
|:---:|
|_Number of images per class_|
</div>



<div align='center'>
<a name='cifar10samples'></a>

|![cifar10samples](../../Images/markdown_images/cifar10samples.png)|
|:---:|
|_Samples from CIFAR-10_|
</div>

The plotting functions and the image grid to be displayed the misclassified and activations maps are defined in [visualize.py](../Dependencies/visualize.py)

---

## Results

The network architecture used for this experiment was _ResNet18_, with _Layer Normalization_ along with different augmentation strategies (i.e. _RandomCrop_, _Cutout (<img src="https://render.githubusercontent.com/render/math?math=16x16">)_, _Rotation (<img src="https://render.githubusercontent.com/render/math?math=\pm%205^\circ">)_). 

<div align='center'>
<a name='resnet18'></a>

|![resnet18](../../Images/markdown_images/resnet18.png)|
|:---:|
|_Network Architecture - ResNet18_|
</div>


The model trained for _40_ epochs with _ReduceLROnPlateau_ learning rate scheduler used to drop the learning rate as the model converges.

<div align='center'>
<a name='resnet18_40epochs'></a>

|![resnet18_40epochs](../../Images/markdown_images/resnet18_40epochs.png)|
|:---:|
|_Network Architecture - ResNet18_|
</div>


It doesn't seem to be overfitting, but it still has some scope to be made better (consider the training accuracy is slightly lower than the test).

<div align='center'>
<a name='train_metric'></a>

|![train_metric](../../Images/markdown_images/train_metric_assi8.png)|
|:---:|
|_Accuracy and Loss on Training Set_|
</div>

<div align='center'>
<a name='test_metric'></a>

|![test_metric](../../Images/markdown_images/test_metric_assi8.png)|
|:---:|
|_Accuracy and Loss on Test Set_|
</div>

The misclassified images along with the class activation map for _Layer 3_ is shown below. We see from the output of the activation map that it at times completely misses the object in question (like the image of the cat and the frog on the last row)

<div align='center'>
<a name='miss_assi8'></a>

|![miss_assi8](../../Images/markdown_images/miss_assi8.png)|
|:---:|
|_Misclassified Images_|
</div>

<div align='center'>
<a name='miss_gradcam'></a>

|![miss_gradcam](../../Images/markdown_images/miss_gradcam.png)|
|:---:|
|_Misclassified Images with Activation Map_|
</div>

---