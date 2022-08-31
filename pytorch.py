#######################################################################################
# pytorch tutorial 60 mins

##########
# TENSORS
# Tensors are a specialized data structure that are very similar to arrays and matrices.
# In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

import torch
import numpy as np

# Tensor Initialization
# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
t = torch.empty(3, 4)


# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)

# Tensor Attributes
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Tensor Operations

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    my_device = torch.device("cuda")
else:
    my_device = torch.device("cpu")


tensor = torch.ones(4, 4)
tensor = tensor.to("cuda")
tensor = torch.ones(4, 4, device="cuda")
tensor[:, 1] = 0
print(tensor)



# Out:
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# matrix multiplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In-place operations Operations that have a _ suffix are in-place. For example: x.copy_(y), x.t_(), will change x.
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# copy a tensor
a = t.clone()   # also copies autograd if enabled
a = t.detach().clone()  # ignore autograd information

# Bridge with NumPy
# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()   # Note: This does not change the underlying memory allcoated
print(f"n: {n}")
numpy_array = np.ones(3, 2)     # Note: This does not change the underlying memory allcoated
t = torch.from_numpy(numpy_array)

# A change in the tensor reflects in the NumPy array.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Squeeze and unsqueeze
a = torch.empty(2, 2)
b = a.unsqueeze(0)   # add new 0th dimension :Add batch size of 1 to the new batch dimension
c = b.squeeze(0)    # Remove 0th dimension of 1

print(a.shape, b.shape)

# reshape
a = torch.empty(6, 20, 20)
b = a.reshape(6 * 20 * 20)  # Note: When it can, reshape() always returns a view

#######################################################################################
# torch.autograd
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# Background
# Neural networks (NNs) are a collection of nested functions that are executed on some input data. These functions are defined by parameters (consisting of weights and biases), which in PyTorch are stored in tensors.
# Training a NN happens in two steps:
# Forward Propagation: In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.
# Backward Propagation: In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using gradient descent. For a more detailed walkthrough of backprop, check out this video from 3Blue1Brown.

# For this example, we load a pretrained resnet18 model from torchvision. We create a random data tensor to represent a single image with 3 channels, and height & width of 64, and its corresponding label initialized to some random values.
# channel(digital image): Color digital images are made of pixels, and pixels are made of combinations of primary colors represented by a series of code. A channel in this context is the grayscale image of the same size as a color image, made of just one of these primary colors. For instance, an image from a standard digital camera will have a red, green and blue channel. A grayscale image has just one channel.
import torch
import torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# forward pass
prediction = model(data)

# We use the model’s prediction and the corresponding label to calculate the error (loss). The next step is to backpropagate this error through the network. Backward propagation is kicked off when we call .backward() on the error tensor. Autograd then calculates and stores the gradients for each model parameter in the parameter’s .grad attribute.
loss = (prediction - labels).sum()
# backward pass
loss.backward()

# Generally speaking, torch.autograd is an engine for computing vector-Jacobian product. That is, given any vector v⃗, compute the product J^T⋅v⃗.

# Next, we load an optimizer, in this case SGD with a learning rate of 0.01 and momentum of 0.9. We register all the parameters of the model in the optimizer.
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Finally, we call .step() to initiate gradient descent. The optimizer adjusts each parameter by its gradient stored in .grad.
# gradient descent
optim.step()


# Exclusion from the DAG
# torch.autograd tracks operations on all tensors which have their requires_grad flag set to True. For tensors that don’t require gradients, setting this attribute to False excludes it from the gradient computation DAG.
# The output tensor of an operation will require gradients even if only a single input tensor has requires_grad=True.

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

# frozen parameters: In a NN, parameters that don’t compute gradients are usually called frozen parameters.

# In finetuning, we freeze most of the model and typically only modify the classifier layers to make predictions on new labels. Let’s walk through a small example to demonstrate this. As before, we load a pretrained resnet18 model, and freeze all the parameters.

from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

# Let’s say we want to finetune the model on a new dataset with 10 labels. In resnet, the classifier is the last linear layer model.fc. We can simply replace it with a new linear layer (unfrozen by default) that acts as our classifier.

model.fc = nn.Linear(512, 10)

# Now all parameters in the model, except the parameters of model.fc, are frozen. The only parameters that compute gradients are the weights and bias of model.fc.

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
loss.backward()     # back propagation
optimizer.step()    # update the weights
optimizer.zero_grad()   # reset the gradients

# Notice although we register all the parameters in the optimizer, the only parameters that are computing gradients (and hence updated in gradient descent) are the weights and bias of the classifier.

# The same exclusionary functionality is available as a context manager in torch.no_grad()


#######################################################################################
# pytorch layer types
# Linear layer
lin = torch.nn.Linear(3, 2)     # 3 inputs, 2 outputs
# Convolutional layer
conv1 = torch.nn.Conv2d(1, 6, 5)
conv2 = torch.nn.Conv2d(6, 16, 3)
# Recurrent layer
lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
# Transformers
# data manipulation layers
torch.nn.MaxPool2d()
torch.nn.MinPool2d()
# Normalization
torch.nn.BatchNorm1d()
# Dropout if overfitting
# Dropout layer: Randomly set some of elements to be 0
dropout = torch.nn.Dropout(p=0.3)
dropout(t)

########################################################################################
# Optimzer
## Common parameters:
###    1. lr: learning rate, size of steps
###    2. momentum: Adjust size of steps given the size of gradient
###    3. weight_decay: encourage weight regularization, avoid overfitting

## hyperparameters: The best value for lr/momentum/weight_decay. Difficult to know priori, often found via grid search or similar methods
########################################################################################
# TensorBoard
from torch.utils.tensoroard import SummaryWriter
########################################################################################
torch.utils.data.DataLoader
########################################################################################
## 1. Data
## 2. Model
## 3. Loss Function
## 4. Optimizer

# inference
model.eval()    # or
model.train(False)

########################################################################################
# torchscript
torch.jit.script()  # may not cover 100% of operators
torch.jit.trace()   # works most of the time; less deterministic(not preserve control flow)


# Epoch, one complete pass of all the training dataset
