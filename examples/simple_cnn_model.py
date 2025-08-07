#https://colab.research.google.com/drive/1HIh8-ZNogP8h2lKZ3LTyo0Zh2uJ6gOQO?usp=sharing
import torch
import torch.nn as nn
import torch.nn.functional as F
import neuronix.module as m
from neuronix.autograd_graph import AutogradGraph
from neuronix.custom_tensor import CustomTensor
from neuronix.module import *
from neuronix.losses import *
from neuronix.optimizers import *
from neuronix.config import device,dtype
import time

"""Calculations for CNN Architecture
The calculations assume an input image shape of 224x224 with 3 color channels (RGB).

Layer 1: Convolutional Block 1
Input: 224x224x3

Conv2D Layer: (3, 3) filter, num_filters=32. The Keras Conv2D layer with default parameters uses a stride of 1, a padding of 0 (valid), and a dilation rate of 1.

Output size:

W_out = (224 - 3 + 2*0)/1 + 1 = 222

H_out = (224 - 3 + 2*0)/1 + 1 = 222

Output shape: 222x222x32

MaxPool2D Layer: pool_size=(2, 2), strides=(2, 2).

Output size:

W_out = (222 - 2 + 2*0)/2 + 1 = 111

H_out = (222 - 2 + 2*0)/2 + 1 = 111

Output shape: 111x111x32

Layer 2: Convolutional Block 2
Input: 111x111x32

Conv2D Layer: (3, 3) filter, num_filters=64. Stride 1, padding 0, dilation rate 1.

Output size:

W_out = (111 - 3 + 2*0)/1 + 1 = 109

H_out = (111 - 3 + 2*0)/1 + 1 = 109

Output shape: 109x109x64

MaxPool2D Layer: pool_size=(2, 2), strides=(2, 2).

Output size:

W_out = (109 - 2 + 2*0)/2 + 1 = 54.5 -> floor(54.5) + 1 = 54

H_out = (109 - 2 + 2*0)/2 + 1 = 54.5 -> floor(54.5) + 1 = 54

Output shape: 54x54x64

Layer 3: Convolutional Block 3
Input: 54x54x64

Conv2D Layer: (3, 3) filter, num_filters=128. Stride 1, padding 0, dilation rate 1.

Output size:

W_out = (54 - 3 + 2*0)/1 + 1 = 52

H_out = (54 - 3 + 2*0)/1 + 1 = 52

Output shape: 52x52x128

MaxPool2D Layer: pool_size=(2, 2), strides=(2, 2).

Output size:

W_out = (52 - 2 + 2*0)/2 + 1 = 26

H_out = (52 - 2 + 2*0)/2 + 1 = 26

Output shape: 26x26x128

Layer 4: Convolutional Block 4
Input: 26x26x128

Conv2D Layer: (5, 5) filter, num_filters=256. Stride 1, padding 0, dilation rate 1.

Output size:

W_out = (26 - 5 + 2*0)/1 + 1 = 22

H_out = (26 - 5 + 2*0)/1 + 1 = 22

Output shape: 22x22x256

MaxPool2D Layer: pool_size=(2, 2), strides=(2, 2).

Output size:

W_out = (22 - 2 + 2*0)/2 + 1 = 11

H_out = (22 - 2 + 2*0)/2 + 1 = 11

Output shape: 11x11x256

Layer 5: Convolutional Block 5
Input: 11x11x256

Conv2D Layer: (7, 7) filter, num_filters=512. Stride 1, padding 0, dilation rate 1.

Output size:

W_out = (11 - 7 + 2*0)/1 + 1 = 5

H_out = (11 - 7 + 2*0)/1 + 1 = 5

Output shape: 5x5x512

MaxPool2D Layer: pool_size=(2, 2), strides=(2, 2).

Output size:

W_out = (5 - 2 + 2*0)/2 + 1 = 2.5 -> floor(2.5) + 1 = 2

H_out = (5 - 2 + 2*0)/2 + 1 = 2.5 -> floor(2.5) + 1 = 2

Output shape: 2x2x512

Final Dense Layer
Flatten Layer: The output of the last max-pooling layer is a 2x2x512 tensor. The flatten layer reshapes this into a one-dimensional vector.

Vector size: 2 * 2 * 512 = 2048

Dense Layer: The input to the first dense layer will be a vector of size 2048.
"""

# Creating the model with pytorch 
print("Heads up the comparison between torch result and neuronix is with an rtol of 1e-4")
class ConvBNActMaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_conv, stride_conv, padding_conv,
                 kernel_size_pool, stride_pool, padding_pool):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_conv,
            stride=stride_conv,
            padding=padding_conv,
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.1)
        self.activation = nn.GELU(approximate='tanh')  # PyTorch ≥ 1.13
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size_pool,
            stride=stride_pool,
            padding=padding_pool,
            dilation=1
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x


class CNN_Model_py(nn.Module):
    def __init__(self):
        super().__init__()
        # Input shape: (3, 224, 224)
        self.layer1 = ConvBNActMaxPool(3, 32, 3, 1, 0, 2, 2, 0)     # -> (32, 111, 111)
        self.layer2 = ConvBNActMaxPool(32, 64, 3, 1, 0, 2, 2, 0)    # -> (64, 54, 54)
        self.layer3 = ConvBNActMaxPool(64, 128, 3, 1, 0, 2, 2, 0)   # -> (128, 26, 26)
        self.layer4 = ConvBNActMaxPool(128, 256, 5, 1, 0, 2, 2, 0)  # -> (256, 11, 11)
        self.layer5 = ConvBNActMaxPool(256, 512, 7, 1, 0, 2, 2, 0)  # -> (512, 2, 2)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(512 * 2 * 2, 512)
        self.relu = nn.ReLU()
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)       # -> (B, 2048)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Creating the same model with neuronix
class conv_batchnorm_activation_maxpool(m.Module):
  def __init__(self, in_channels, out_channels, kernel_size_conv, stride_conv, padding_conv,kernel_size_pool,stride_pool,padding_pool, graph):
    super().__init__()
    self.conv = m.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size_conv,
        stride = stride_conv,
        padding = padding_conv,
        graph = graph,
        activation="gelu_approx"
    )
    self.batchnorm = m.BatchNorm_Nd(
        num_features=out_channels,
        eps=1e-5,
        momentum=0.1,
        graph=graph)
    self.activation = m.GeLu(
        approximate='tanh',
        graph=graph)
    self.maxpool = m.MaxPool2d(
        kernel_size=kernel_size_pool,#(2,2),
        stride=stride_pool,#(2,2)
        padding=padding_pool,#(0,0)
        dilation=1,
        graph=graph)
  def forward(self, x):
    x = self.conv(x)
    x = self.batchnorm(x)
    x = self.activation(x)
    x = self.maxpool(x)
    return x
class CNN_Model(m.Module):
  # configaration kernel size  [(3, 3), (3, 3), (3, 3), (5, 5), (7, 7)]
  # output_channels [32,64,128,256,512]
  # dense layer 512

  # conv 224-3 +1 = 222
  # max 222 -2/2 +1 = 111

  # conv 111-3 +1 = 109
  # max 109 -2/2 +1 = 54

  # conv 54 -3 +1 = 52
  # max 52 -2/2 +1 = 26

  # conv 26 -5 +1 = 22
  # max 22-2/2 +1 =11

  # conv 11 -7 +1 = 5
  # max = 5 -2/2 +1 = 2

  # hence for linear layer the reshaped tensor is 512*2*2 = 2048

  def __init__(self,graph):
    super().__init__()
    #in_channels, out_channels, kernel_size_conv, stride_conv, padding_conv,kernel_size_pool,stride_pool,padding_pool, graph
    self.layer1 = conv_batchnorm_activation_maxpool(3,32, 3,1,0, 2,2,0, graph)
    self.layer2 = conv_batchnorm_activation_maxpool(32,64, 3,1,0, 2,2,0, graph)
    self.layer3 = conv_batchnorm_activation_maxpool(64,128, 3,1,0, 2,2,0, graph)
    self.layer4 = conv_batchnorm_activation_maxpool(128,256, 5,1,0, 2,2,0, graph)
    self.layer5 = conv_batchnorm_activation_maxpool(256,512, 7,1,0, 2,2,0, graph)
    self.linear1 = m.Linear(
        in_features=512*2*2,
        out_features=512,
        graph=graph,
        activation="relu"
    )
    self.ac1 = m.ReLu(graph=graph)
    self.output = m.Linear(
        in_features=512,
        out_features=10,
        graph=graph,
        activation="relu"
    )
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.reshape((x.shape[0],-1))
    x = self.linear1(x)
    x = self.ac1(x)
    x = self.output(x)
    return x

# Verifying Forward Pass with Torch
cnn_model_pytorch = CNN_Model_py()
cnn_model = CNN_Model(graph=None)
with AutogradGraph() as graph, torch.inference_mode():
  # copying all the weights of pytorch to our model
  cnn_model_pytorch.layer1.conv.weight.data.copy_(
    cnn_model._modules['layer1'].conv.weight.tensor
  )
  cnn_model_pytorch.layer1.conv.bias.data.copy_(
      cnn_model._modules['layer1'].conv.bias.tensor
  )
  cnn_model_pytorch.layer1.batchnorm.weight.data.copy_(
      cnn_model._modules['layer1'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer1.batchnorm.bias.data.copy_(
      cnn_model._modules['layer1'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer1.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer1'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer1.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer1'].batchnorm.running_var
  )

  #### Layer 2 ####
  cnn_model_pytorch.layer2.conv.weight.data.copy_(
      cnn_model._modules['layer2'].conv.weight.tensor
  )
  cnn_model_pytorch.layer2.conv.bias.data.copy_(
      cnn_model._modules['layer2'].conv.bias.tensor
  )
  cnn_model_pytorch.layer2.batchnorm.weight.data.copy_(
      cnn_model._modules['layer2'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer2.batchnorm.bias.data.copy_(
      cnn_model._modules['layer2'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer2.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer2'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer2.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer2'].batchnorm.running_var
  )

  #### Layer 3 ####
  cnn_model_pytorch.layer3.conv.weight.data.copy_(
      cnn_model._modules['layer3'].conv.weight.tensor
  )
  cnn_model_pytorch.layer3.conv.bias.data.copy_(
      cnn_model._modules['layer3'].conv.bias.tensor
  )
  cnn_model_pytorch.layer3.batchnorm.weight.data.copy_(
      cnn_model._modules['layer3'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer3.batchnorm.bias.data.copy_(
      cnn_model._modules['layer3'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer3.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer3'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer3.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer3'].batchnorm.running_var
  )

  #### Layer 4 ####
  cnn_model_pytorch.layer4.conv.weight.data.copy_(
      cnn_model._modules['layer4'].conv.weight.tensor
  )
  cnn_model_pytorch.layer4.conv.bias.data.copy_(
      cnn_model._modules['layer4'].conv.bias.tensor
  )
  cnn_model_pytorch.layer4.batchnorm.weight.data.copy_(
      cnn_model._modules['layer4'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer4.batchnorm.bias.data.copy_(
      cnn_model._modules['layer4'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer4.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer4'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer4.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer4'].batchnorm.running_var
  )

  #### Layer 5 ####
  cnn_model_pytorch.layer5.conv.weight.data.copy_(
      cnn_model._modules['layer5'].conv.weight.tensor
  )
  cnn_model_pytorch.layer5.conv.bias.data.copy_(
      cnn_model._modules['layer5'].conv.bias.tensor
  )
  cnn_model_pytorch.layer5.batchnorm.weight.data.copy_(
      cnn_model._modules['layer5'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer5.batchnorm.bias.data.copy_(
      cnn_model._modules['layer5'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer5.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer5'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer5.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer5'].batchnorm.running_var
  )
  #### Linear 1 ####
  cnn_model_pytorch.linear1.weight.data.copy_(
      cnn_model._modules['linear1'].weight.tensor
  )
  cnn_model_pytorch.linear1.bias.data.copy_(
      cnn_model._modules['linear1'].bias.tensor
  )
  #### Output ####
  cnn_model_pytorch.output.weight.data.copy_(
      cnn_model._modules['output'].weight.tensor
  )
  cnn_model_pytorch.output.bias.data.copy_(
      cnn_model._modules['output'].bias.tensor
  )



  cnn_model.eval()
  cnn_model_pytorch.eval()
  sample_input_torch = torch.randn(3,3,224,224)
  sample_input = CustomTensor(sample_input_torch.clone(), _custom_requires_grad=False, graph=None)
  output = cnn_model(sample_input)
  output_torch = cnn_model_pytorch(sample_input_torch)
  print(torch.allclose(output.tensor,output_torch,rtol=1e-4))

del cnn_model,cnn_model_pytorch,sample_input,sample_input_torch

# Verifying Backward Pass with Torch
cnn_model_pytorch = CNN_Model_py().to(device = device ,dtype = dtype)
cnn_model = CNN_Model(graph=None).to(device = device ,dtype = dtype)
# with AutogradGraph() as graph:
#    dummy_tensor = CustomTensor(torch.rand(3,3,224,224),_custom_requires_grad = False,graph = None)
#    cnn_model.attach_graph(graph=graph)
#    o=cnn_model(dummy_tensor)
#    l=o.sum()
#    l.backward()
# del l,o,dummy_tensor
# cnn_model.detach_graph()
# cnn_model.zero_grad()
with AutogradGraph() as graph:
  
  
  # copying all the weights of pytorch to our model
  # assume cnn_model_pytorch (a trained torch CNN_Model_py) and
  # cnn_model (your custom CNN_Model under AutogradGraph) are already instantiated
  cnn_model.attach_graph(graph=graph)
  cnn_model_pytorch.layer1.conv.weight.data.copy_(
    cnn_model._modules['layer1'].conv.weight.tensor
  )
  cnn_model_pytorch.layer1.conv.bias.data.copy_(
      cnn_model._modules['layer1'].conv.bias.tensor
  )
  cnn_model_pytorch.layer1.batchnorm.weight.data.copy_(
      cnn_model._modules['layer1'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer1.batchnorm.bias.data.copy_(
      cnn_model._modules['layer1'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer1.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer1'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer1.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer1'].batchnorm.running_var
  )

  #### Layer 2 ####
  cnn_model_pytorch.layer2.conv.weight.data.copy_(
      cnn_model._modules['layer2'].conv.weight.tensor
  )
  cnn_model_pytorch.layer2.conv.bias.data.copy_(
      cnn_model._modules['layer2'].conv.bias.tensor
  )
  cnn_model_pytorch.layer2.batchnorm.weight.data.copy_(
      cnn_model._modules['layer2'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer2.batchnorm.bias.data.copy_(
      cnn_model._modules['layer2'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer2.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer2'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer2.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer2'].batchnorm.running_var
  )

  #### Layer 3 ####
  cnn_model_pytorch.layer3.conv.weight.data.copy_(
      cnn_model._modules['layer3'].conv.weight.tensor
  )
  cnn_model_pytorch.layer3.conv.bias.data.copy_(
      cnn_model._modules['layer3'].conv.bias.tensor
  )
  cnn_model_pytorch.layer3.batchnorm.weight.data.copy_(
      cnn_model._modules['layer3'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer3.batchnorm.bias.data.copy_(
      cnn_model._modules['layer3'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer3.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer3'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer3.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer3'].batchnorm.running_var
  )

  #### Layer 4 ####
  cnn_model_pytorch.layer4.conv.weight.data.copy_(
      cnn_model._modules['layer4'].conv.weight.tensor
  )
  cnn_model_pytorch.layer4.conv.bias.data.copy_(
      cnn_model._modules['layer4'].conv.bias.tensor
  )
  cnn_model_pytorch.layer4.batchnorm.weight.data.copy_(
      cnn_model._modules['layer4'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer4.batchnorm.bias.data.copy_(
      cnn_model._modules['layer4'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer4.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer4'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer4.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer4'].batchnorm.running_var
  )

  #### Layer 5 ####
  cnn_model_pytorch.layer5.conv.weight.data.copy_(
      cnn_model._modules['layer5'].conv.weight.tensor
  )
  cnn_model_pytorch.layer5.conv.bias.data.copy_(
      cnn_model._modules['layer5'].conv.bias.tensor
  )
  cnn_model_pytorch.layer5.batchnorm.weight.data.copy_(
      cnn_model._modules['layer5'].batchnorm.weight.tensor
  )
  cnn_model_pytorch.layer5.batchnorm.bias.data.copy_(
      cnn_model._modules['layer5'].batchnorm.bias.tensor
  )
  cnn_model_pytorch.layer5.batchnorm.running_mean.data.copy_(
      cnn_model._modules['layer5'].batchnorm.running_mean
  )
  cnn_model_pytorch.layer5.batchnorm.running_var.data.copy_(
      cnn_model._modules['layer5'].batchnorm.running_var
  )
  #### Linear 1 ####
  cnn_model_pytorch.linear1.weight.data.copy_(
      cnn_model._modules['linear1'].weight.tensor
  )
  cnn_model_pytorch.linear1.bias.data.copy_(
      cnn_model._modules['linear1'].bias.tensor
  )
  #### Output ####
  cnn_model_pytorch.output.weight.data.copy_(
      cnn_model._modules['output'].weight.tensor
  )
  cnn_model_pytorch.output.bias.data.copy_(
      cnn_model._modules['output'].bias.tensor
  )


  cnn_model.verify_all_graph_references_are_weak()
  cnn_model.train()
  cnn_model_pytorch.train()

  sample_input_torch = torch.randn(3,3,224,224)
  sample_input = CustomTensor(sample_input_torch.clone(),_custom_requires_grad=True,graph=graph)
  output = cnn_model(sample_input)
  output_torch = cnn_model_pytorch(sample_input_torch)
  loss = output.sum()
  loss_torch = output_torch.sum()
  st = time.time()
  loss.backward()
  et = time.time()
  print(f"Neuronix implementation backward Takes {et-st} seconds")
  st = time.time()
  loss_torch.backward()
  et = time.time()
  print(f"Pytorch implementation backward Takes {et-st} seconds")

cnn_model.verify_all_parameters_are_on_the_same_device(device)

def compare_grads(param1, param2, name):
    grad1 = param1.grad
    grad2 = param2.grad
    if grad1 is None or grad2 is None:
        print(f"{name}: One of the gradients is None.")
    else:
        equal = torch.allclose(grad1, grad2, atol=1e-4)
        print(f"{name}: {'✅ Same' if equal else '❌ Different'}")

# Layer-wise comparisons
for i in range(1, 6):
    layer = f"layer{i}"
    torch_layer = getattr(cnn_model_pytorch, layer)
    custom_layer = cnn_model._modules[layer]

    compare_grads(torch_layer.conv.weight, custom_layer.conv.weight.tensor, f"{layer}.conv.weight")
    compare_grads(torch_layer.conv.bias, custom_layer.conv.bias.tensor, f"{layer}.conv.bias")
    compare_grads(torch_layer.batchnorm.weight, custom_layer.batchnorm.weight.tensor, f"{layer}.batchnorm.weight")
    compare_grads(torch_layer.batchnorm.bias, custom_layer.batchnorm.bias.tensor, f"{layer}.batchnorm.bias")

# Linear 1
compare_grads(cnn_model_pytorch.linear1.weight, cnn_model._modules['linear1'].weight.tensor, "linear1.weight")
compare_grads(cnn_model_pytorch.linear1.bias, cnn_model._modules['linear1'].bias.tensor, "linear1.bias")

# Output
compare_grads(cnn_model_pytorch.output.weight, cnn_model._modules['output'].weight.tensor, "output.weight")
compare_grads(cnn_model_pytorch.output.bias, cnn_model._modules['output'].bias.tensor, "output.bias")