# Dependencies
---

This directory contains files that can be imported directly into further assignment.

- Define the network architectures to use in a seperate files, so that it can imported into any project/assignment. _ResNet_ architecture defined [here](https://github.com/kuangliu/pytorch-cifar) was cloned and edited. ResNet 18 and 34 is retained and further modifications were done to add _Layer Normalization_, _Group Normalization_ and _Instance Normalization_. The modified version of the file can be found in [resnet.py](../Dependencies/models/resnet.py).

- Similar all the training and the test code along with the definition for optimizers and Learning Rate Schedulers are moved into [main.py](../Dependencies/main.py). 

- All the data related functions, such as defining a _Custom DataLoader_, _Custom Augmentations (Erosion and Dilation)_; which was used in the previous assignment, sample data loader and function to describe the dataset loaded are defined in [utils.py](../Dependencies/utils.py).

- The plotting functions and the image grid to be displayed the misclassified and activations maps are defined in [visualize.py](../Dependencies/visualize.py)

---

**Use the following code to load the files directly into the workspace (Colaboratory)**

Uses `wget` to download the required files into the respective directory.

```python
# Get custom dependencies from GitHub
# Create a folder to contain it
!mkdir -p ./dependencies/models/

# Download the required files
# `resnet.py` defines the ResNet Network architecture to use
#        Reference: https://github.com/kuangliu/pytorch-cifar
# `main.py` contains the train and test code to train the model
# `utils.py` contains the code to load and transform the data
# `visualize.py` contains code to visualize the results (accuracy and loss plots,
#        misclassified images etc.)

# Base URL defining the repo name and the path at which the file are located
base_url = 'https://raw.githubusercontent.com/NarutoScoo/EVA6/main/Assignments/Dependencies'

# Location at which the files will be downloaded
dependencies = './dependencies/'

# List of files to get the from the path defined (`base_url`)
# Format: <filename> or <dirname>/<filename>
list_of_files = ['models/resnet.py', 'main.py', 'utils.py', 'visualize.py']

# Iterate through and download the required files
for filename in list_of_files:
    !wget {base_url}/{filename} -O {dependencies}/{filename}

# Add dependencies to the system path
# to import the files
sys.path.append (dependencies)

# Import the modules
from dependencies.models import resnet
from dependencies import utils, visualize
```
