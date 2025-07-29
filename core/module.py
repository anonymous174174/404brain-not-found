import torch
import math
import weakref
import torch.nn.functional as F
from custom_tensor import CustomTensor
from collections import OrderedDict
from functools import partial
from typing import Optional, Any
def get_activations(leaky_relu_slope=0.01, elu_alpha=1.0):
    return {
        "relu": F.relu,
        "gelu": F.gelu,
        "leaky_relu": partial(F.leaky_relu, negative_slope=leaky_relu_slope),
        "sigmoid": F.sigmoid,
        "tanh": F.tanh,
        "silu": F.silu,
        "elu": partial(F.elu, alpha=elu_alpha),
        "gelu_approx": partial(F.gelu, approximate='tanh')
    }


@torch.compile
def relu_derivative(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(x.dtype)

@torch.compile
def gelu_derivative(x: torch.Tensor) -> torch.Tensor:
    sqrt_2_pi = 2.5066282749176025 #torch.tensor(2 * torch.pi).sqrt()
    phi_x_cdf = 0.5 * (1 + torch.special.erf(x / 1.4142135381698608)) #torch.sqrt(torch.tensor(2.0))))
    phi_x_pdf = torch.exp(-0.5 * x**2) / sqrt_2_pi
    return phi_x_cdf + x * phi_x_pdf

@torch.compile
def leaky_relu_derivative(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, negative_slope))

@torch.compile
def sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1 - s)

@torch.compile
def tanh_derivative(x: torch.Tensor) -> torch.Tensor:
    t = torch.tanh(x).square()
    return 1 - t

@torch.compile
def silu_derivative(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s*( 1 + x * (1 - s))

@torch.compile
def elu_derivative(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), torch.exp(x) * alpha)

@torch.compile
def gelu_tanh_approx_derivative(x: torch.Tensor) -> torch.Tensor:
    sqrt_2_over_pi = 0.7978845238685608 #torch.tensor(2.0 / torch.pi).sqrt()
    coeff_cubic = 0.044715
    x2 = x.square()
    inner = x + coeff_cubic * x2 * x
    u = sqrt_2_over_pi * inner
    tanh_u = torch.tanh(u)
    poly = 1 + 3 * coeff_cubic * x2
    return 0.5 * tanh_u + 0.5 * (1 - tanh_u.square()) * (sqrt_2_over_pi * poly * x) + 0.5

def get_activation_derivatives(leaky_relu_slope=0.01, elu_alpha=1.0):
    """ keep in mind that these are not exact derivatives autograd derivatives are more stable 
    Checking manually implemented derivatives vs autograd:
    relu         | close: True | max abs diff: 0.000e+00
    gelu         | close: True | max abs diff: 1.192e-07
    leaky_relu   | close: True | max abs diff: 0.000e+00
    sigmoid      | close: True | max abs diff: 0.000e+00
    tanh         | close: True | max abs diff: 3.390e-07
    silu         | close: True | max abs diff: 1.192e-07
    elu          | close: True | max abs diff: 0.000e+00
    gelu_approx  | close: True | max abs diff: 5.960e-07
    """

    return {
        "relu": relu_derivative,
        "gelu": gelu_derivative,
        "leaky_relu": partial(leaky_relu_derivative, negative_slope=leaky_relu_slope),
        "sigmoid": sigmoid_derivative,
        "tanh": tanh_derivative,
        "silu": silu_derivative,
        "elu": partial(elu_derivative, alpha=elu_alpha),
        "gelu_approx": gelu_tanh_approx_derivative
    }
ACTIVATIONS = get_activations(leaky_relu_slope=0.01, elu_alpha=1.0)
ACTIVATIONS_GRADIENT = get_activation_derivatives(leaky_relu_slope=0.01, elu_alpha=1.0)
class Module:
    """
    Base class for all neural network modules. Your models should also subclass this class.
    Modules can also contain other Modules, allowing to nest them in a tree structure.
    """
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        # self._buffers = OrderedDict()
        self.training = True #
        
    def __setattr__(self, name, value):
        if isinstance(value, CustomTensor):
            if value._custom_requires_grad:
                self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        # Handle buffers (non-parameter tensors like running_mean in BatchNorm)
        # elif isinstance(value, torch.Tensor):
        #     self._buffers[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        """Returns a list of all parameters in the module and its submodules."""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            p._zero_grad()

    def train(self):
        """Sets the module and all its submodules to training mode."""
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        """Sets the module and all its submodules to evaluation mode."""
        self.training = False
        for module in self._modules.values():
            module.eval()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses of Module must implement a forward method.")
#_______________________________________________________________________________________________________________________________________--
#LINEAR AKA DENSE LAYER
class Linear(Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b
    types of activation relu,leaky_relu, gelu, sigmoid, tanh, silu,elu"""
    def __new__(cls, in_features, out_features, bias=True, *, graph=None,activation="relu"):
        assert activation in {"relu", "gelu", "leaky_relu", "sigmoid", "tanh", "silu", "elu", "gelu_approx"}
        return super().__new__(cls)
    def __init__(self, in_features, out_features, bias=True, *, graph=None,activation="relu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph = weakref.proxy(graph) if graph is not None else None
        
        self.weight = CustomTensor(torch.empty(out_features, in_features), _custom_requires_grad=True, graph=graph, is_leaf=True)
        if activation in {"relu", "gelu", "silu", "elu","gelu_approx"}:
            torch.nn.init.kaiming_uniform_(self.weight.tensor, nonlinearity="relu")
        elif activation == "leaky_relu":
            torch.nn.init.kaiming_uniform_(self.weight.tensor, nonlinearity="leaky_relu")
        elif activation == "sigmoid":
            torch.nn.init.xavier_uniform_(self.weight.tensor, gain=1.0)
        elif activation == "tanh":
            torch.nn.init.xavier_uniform_(self.weight.tensor, gain=5/3)
        
        if bias:
            self.bias = CustomTensor(torch.zeros(out_features), _custom_requires_grad=True, graph=graph, is_leaf=True)
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.tensor)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias.tensor, -bound, bound)
        else:
            self.bias = None
    def forward(self, input_tensor):
        
        output = input_tensor.tensor @ self.weight.tensor.transpose(-2, -1)
        if self.bias is not None:
            output.add_(self.bias.tensor)

        if not self.training:
            return CustomTensor(output, due_to_operation=True)
        graph = self.graph
        result = CustomTensor(output, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        
        if input_tensor._custom_requires_grad:
            graph.add_edge(input_tensor._node_id, result._node_id)
        graph.add_edge(self.weight._node_id, result._node_id)
        if self.bias is not None:
            graph.add_edge(self.bias._node_id, result._node_id)

        weight_ref = weakref.proxy(self.weight)
        input_tensor_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)
        bias_ref = weakref.proxy(self.bias) if self.bias is not None else None


        def _backward():

            if weight_ref._custom_requires_grad:
                if weight_ref.tensor.grad is None:
                    weight_ref._zero_grad()
                grad_w = torch.matmul(result_ref.tensor.grad.transpose(-2, -1), input_tensor_ref.tensor)
                weight_ref.tensor.grad.add_(weight_ref._reduce_grad_for_broadcast(grad_w, weight_ref.tensor.shape))
            

            if bias_ref is not None:
                if bias_ref._custom_requires_grad:
                    if bias_ref.tensor.grad is None:
                        bias_ref._zero_grad()
                    grad_b = bias_ref._reduce_grad_for_broadcast(result_ref.tensor.grad, bias_ref.tensor.shape)
                    bias_ref.tensor.grad.add_(grad_b)
            
            # Input gradient
            if input_tensor_ref._custom_requires_grad:
                if input_tensor_ref.tensor.grad is None:
                    input_tensor_ref._zero_grad()
                grad_in = torch.matmul(result_ref.tensor.grad, weight_ref.tensor)
                input_tensor_ref.tensor.grad.add_(input_tensor_ref._reduce_grad_for_broadcast(grad_in, input_tensor_ref.tensor.shape))
        
        result.backward = _backward
        return result
    # def forward(self, input_tensor):
    #     graph = self.graph
    #     if self.bias is not None:
    #         if not self.training:
    #             torch_result = input_tensor.tensor @ self.weight.tensor.transpose(-2,-1) + self.bias.tensor
    #             return CustomTensor(torch_result, due_to_operation=True)
    #         torch_result = input_tensor.tensor @ self.weight.tensor.transpose(-2,-1) + self.bias.tensor
    #         result = CustomTensor(torch_result, _custom_requires_grad=True, graph=self.graph, due_to_operation=True, is_leaf=False)
    #         if input_tensor._custom_requires_grad:
    #             graph.add_edge(input_tensor._node_id, result._node_id)
    #         graph.add_edge(self.weight._node_id, result._node_id)
    #         graph.add_edge(self.bias._node_id, result._node_id)
    #         weight_ref = weakref.proxy(self.weight)
    #         bias_ref = weakref.proxy(self.bias)
    #         input_tensor_ref = weakref.proxy(input_tensor)
    #         result_ref = weakref.proxy(result)
    #         def _backward():
    #             if weight_ref._custom_requires_grad:
    #                 if weight_ref.tensor.grad is None:
    #                     weight_ref._zero_grad()
    #                 grad_for_weight = torch.matmul(result_ref.tensor.grad.transpose(-2,-1),input_tensor_ref.tensor)
    #                 weight_ref.tensor.grad.add_(weight_ref._reduce_grad_for_broadcast(grad_for_weight,weight_ref.tensor.shape)) 
    #             if bias_ref._custom_requires_grad:
    #                 if bias_ref.tensor.grad is None:
    #                     bias_ref._zero_grad()
    #                 grad_for_bias = bias_ref._reduce_grad_for_broadcast(result_ref.tensor.grad, bias_ref.tensor.shape)
    #                 bias_ref.tensor.grad.add_(grad_for_bias)
    #             if input_tensor_ref._custom_requires_grad:
    #                 if input_tensor_ref.tensor.grad is None:
    #                     input_tensor_ref._zero_grad()
    #                 grad_for_input = torch.matmul(result_ref.tensor.grad, weight_ref.tensor)
    #                 input_tensor_ref.tensor.grad.add_(input_tensor_ref._reduce_grad_for_broadcast(grad_for_input, input_tensor_ref.tensor.shape))
    #         result.backward = _backward
    #         return result
    #     else:
    #         if not self.training:
    #             torch_result = input_tensor.tensor @ self.weight.tensor.transpose(-2,-1) 
    #             return CustomTensor(torch_result, due_to_operation=True)
    #         torch_result = input_tensor.tensor @ self.weight.tensor.transpose(-2,-1)
    #         result = CustomTensor(torch_result, _custom_requires_grad=True, graph=self.graph, due_to_operation=True, is_leaf=False)
    #         if input_tensor._custom_requires_grad:
    #             graph.add_edge(input_tensor._node_id, result._node_id)
    #         graph.add_edge(self.weight._node_id, result._node_id)
    #         weight_ref = weakref.proxy(self.weight)
    #         input_tensor_ref = weakref.proxy(input_tensor)
    #         result_ref = weakref.proxy(result)
    #         def _backward():
    #             if weight_ref._custom_requires_grad:
    #                 if weight_ref.tensor.grad is None:
    #                     weight_ref._zero_grad()
    #                 grad_for_weight = torch.matmul(result_ref.tensor.grad.transpose(-2,-1),input_tensor_ref.tensor)
    #                 weight_ref.tensor.grad.add_(weight_ref._reduce_grad_for_broadcast(grad_for_weight,weight_ref.tensor.shape)) 
    #             if input_tensor_ref._custom_requires_grad:
    #                 if input_tensor_ref.tensor.grad is None:
    #                     input_tensor_ref._zero_grad()
    #                 grad_for_input = torch.matmul(result_ref.tensor.grad, weight_ref.tensor)
    #                 input_tensor_ref.tensor.grad.add_(input_tensor_ref._reduce_grad_for_broadcast(grad_for_input, input_tensor_ref.tensor.shape))
    #         result.backward = _backward
    #         return result

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
# class Linear_with_activation(Module):
#     """Applies a linear transformation to the incoming data: y = xA^T + b and activation to it
#     types of activation relu,leaky_relu, gelu, sigmoid, tanh, silu,elu"""
#     def __new__(cls, in_features, out_features, bias=True, *, graph=None,activation="relu"):
#         assert activation in {"relu", "gelu", "leaky_relu", "sigmoid", "tanh", "silu", "elu", "gelu_approx"}
#         return super().__new__(cls)
#     def __init__(self, in_features, out_features, bias=True, *, graph=None,activation="relu"):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.graph = weakref.proxy(graph)
#         self.activation = activation
        
#         self.weight = CustomTensor(torch.empty(out_features, in_features), _custom_requires_grad=True, graph=graph, is_leaf=True)
#         if activation in {"relu", "gelu", "silu", "elu","gelu_approx"}:
#             torch.nn.init.kaiming_uniform_(self.weight.tensor, nonlinearity="relu")
#         elif activation == "leaky_relu":
#             torch.nn.init.kaiming_uniform_(self.weight.tensor, nonlinearity="leaky_relu")
#         elif activation == "sigmoid":
#             torch.nn.init.xavier_uniform_(self.weight.tensor, gain=1.0)
#         elif activation == "tanh":
#             torch.nn.init.xavier_uniform_(self.weight.tensor, gain=5/3)
        
#         if bias:
#             self.bias = CustomTensor(torch.zeros(out_features), _custom_requires_grad=True, graph=graph, is_leaf=True)
#             # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.tensor)
#             # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             # torch.nn.init.uniform_(self.bias.tensor, -bound, bound)
#         else:
#             self.bias = None
#     def forward(self, input_tensor):
        
#         activation = self.activation
#         pre_activation = input_tensor.tensor @ self.weight.tensor.transpose(-2, -1)
#         if self.bias is not None:
#             pre_activation.add_(self.bias.tensor)

#         output = ACTIVATIONS[activation](pre_activation)
#         if not self.training:
#             return CustomTensor(output, due_to_operation=True)
#         graph = self.graph
#         result = CustomTensor(output, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        
#         if input_tensor._custom_requires_grad:
#             graph.add_edge(input_tensor._node_id, result._node_id)
#         graph.add_edge(self.weight._node_id, result._node_id)
#         if self.bias is not None:
#             graph.add_edge(self.bias._node_id, result._node_id)

#         weight_ref = weakref.proxy(self.weight)
#         input_tensor_ref = weakref.proxy(input_tensor)
#         result_ref = weakref.proxy(result)
#         bias_ref = weakref.proxy(self.bias) if self.bias is not None else None


#         def _backward():
#             d_activation_d_pre_activation = ACTIVATIONS_GRADIENT[activation](pre_activation)
#             pre_activation_grad = d_activation_d_pre_activation*result_ref.tensor.grad
#             if weight_ref._custom_requires_grad:
#                 if weight_ref.tensor.grad is None:
#                     weight_ref._zero_grad()
                
#                 grad_w = torch.matmul(pre_activation_grad.transpose(-2, -1), input_tensor_ref.tensor)
#                 weight_ref.tensor.grad.add_(weight_ref._reduce_grad_for_broadcast(grad_w, weight_ref.tensor.shape))
            

#             if bias_ref is not None:
#                 if bias_ref._custom_requires_grad:
#                     if bias_ref.tensor.grad is None:
#                         bias_ref._zero_grad()
#                     grad_b = bias_ref._reduce_grad_for_broadcast(pre_activation_grad, bias_ref.tensor.shape)
#                     bias_ref.tensor.grad.add_(grad_b)
            
#             if input_tensor_ref._custom_requires_grad:
#                 if input_tensor_ref.tensor.grad is None:
#                     input_tensor_ref._zero_grad()
#                 grad_in = torch.matmul(pre_activation_grad, weight_ref.tensor)
#                 input_tensor_ref.tensor.grad.add_(input_tensor_ref._reduce_grad_for_broadcast(grad_in, input_tensor_ref.tensor.shape))
        
#         result.backward = _backward
#         return result

#     def __repr__(self):
#         return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

#____________________________________________________________________________________________________________________________
#CONVOLUTION LAYERS ✅ Conv → BatchNorm → ReLU → MaxPool
#complete but not tested
class Conv2d(Module):
    def __new__(cls, *,in_channels, out_channels, kernel_size, stride=1,dilation=1,groups=1,bias=True, padding=0, graph=None,activation="relu"):
        assert isinstance(kernel_size, int) or len(kernel_size) == 2
        assert isinstance(stride, int) or len(stride) == 2
        assert isinstance(dilation, int) or len(dilation) == 2
        assert isinstance(padding, int) or len(padding) == 2
        assert activation in {"relu", "gelu", "leaky_relu", "sigmoid", "tanh", "silu", "elu", "gelu_approx"}
        return super().__new__(cls)
    def __init__(self, *,in_channels, out_channels, kernel_size, stride=1,dilation=1,groups=1,bias=True, padding=0, graph=None,activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.groups = groups
        self.graph = weakref.proxy(graph) if graph is not None else None

        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = CustomTensor(torch.empty(weight_shape), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
        if activation in {"relu", "gelu", "silu", "elu","gelu_approx"}:
            torch.nn.init.kaiming_uniform_(self.weight.tensor, nonlinearity="relu")
        elif activation == "leaky_relu":
            torch.nn.init.kaiming_uniform_(self.weight.tensor, nonlinearity="leaky_relu")
        elif activation == "sigmoid":
            torch.nn.init.xavier_uniform_(self.weight.tensor, gain=1.0)
        elif activation == "tanh":
            torch.nn.init.xavier_uniform_(self.weight.tensor, gain=5/3)
        
        if bias:
            self.bias = CustomTensor(torch.zeros(out_channels), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
        else:
            self.bias = None

    def forward(self, input_tensor):
        output_tensor = F.conv2d(
            input = input_tensor.tensor,
            weight = self.weight.tensor,
            bias = self.bias.tensor if self.bias else None,
            stride = self.stride,
            padding = self.padding,
            groups=self.groups  
        )
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, due_to_operation=True, is_leaf=False)
        
        self.graph.add_edge(input_tensor._node_id, result._node_id)
        self.graph.add_edge(self.weight._node_id, result._node_id)
        if self.bias is not None:
            self.graph.add_edge(self.bias._node_id, result._node_id)

        
        input_ref = weakref.proxy(input_tensor)
        weight_ref = weakref.proxy(self.weight)
        bias_ref = weakref.proxy(self.bias) if self.bias is not None else None
        result_ref = weakref.proxy(result)

        def _backward():
            grad_output = result_ref.tensor.grad
            if bias_ref is not None:
                if bias_ref._custom_requires_grad:
                    if bias_ref.tensor.grad is None: bias_ref._zero_grad()
                    bias_ref.tensor.grad.add_(grad_output.sum(dim=[0, 2, 3]))
                
            if input_ref._custom_requires_grad:
                if input_ref.tensor.grad is None: input_ref._zero_grad()
                input_ref.tensor.grad.add_(
                    self._calculate_gradient_input_tensor(input_ref.tensor,weight_ref.tensor,grad_output)
                )

            if weight_ref._custom_requires_grad:
                if weight_ref.tensor.grad is None: weight_ref._zero_grad()
                # tried vectorizing groups but failed hence using autograd for computing weight for efficiency (NOTE This is considered cheating)
                weight_ref.tensor.grad.add_(
                    torch.nn.grad.conv2d_weight(
                    input=input_tensor.tensor,
                    weight_size=weight_ref.tensor.shape,
                    grad_output=grad_output,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                    )
                )

        result._backward = _backward
        return result
    @torch.compile
    def _calculate_gradient_input_tensor(self, input_tensor,weight_tensor,grad_output):
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        h_out, w_out = grad_output.shape[2], grad_output.shape[3]
        stride = self.stride
        padding = self.padding
        kernel_size = self.kernel_size
        dilation = self.dilation
        # The formula relating input size to output size in a transposed convolution is:
        # InputSize = (OutputSize - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
        # We rearrange this to solve for the required output_padding.
        output_padding_h = h_in - ((h_out - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1)
        output_padding_w = w_in - ((w_out - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1)
        output_padding = (output_padding_h, output_padding_w)

        grad_input = F.conv_transpose2d(
            grad_output,
            weight_tensor,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=self.groups
        )
        return grad_input
    @torch.compille
    def _calculate_gradient_weight_tensor_loop(self,input_tensor,grad_output):
        #The gradient w.r.t. the weights is a convolution
        # of the input (X) and the output gradient (grad_output).
        # For grouped convolutions, we must perform this calculation for each group separately.
        #O(b,co,oh,ow)=B(co)+ kh =0∑KH −1  kw =0∑KW −1  ci=(co/G)⋅(Cin/G)∑((co/G)+1)⋅(Cin/G)−1
        #  Ipadded(b,ci,ih,iw)K(co ,ci ,kh ,kw ),
        # where ih  = oh.sh+kh.dh, iw = ow.sw+kw.dw

        # ∂L/∂K(ci′ ,co′ ,kh′ ,kw′ ) =b,oh,ow∑ G(b,co',oh,ow) 
        # Ipadded(b,ci', oh.sh + kh'.dh, ow.sw + kw'.dw)

        # the original operation is a summation over kh and kw and the input image 
        # coordinates ih iw are sampled with dilation. (oh and ow for individual coordinates are constant)


        # the equation for the gradient is a summation over oh and ow and the input image 
        # coordinates ih iw are sampled with stride. 
        # (kh and kw are constant for individual coordinates are constant)

        # hence when calling conv2d we need to switch stride and dilation
        # and also transpose the dimensions of batch and channel as for derivative with respect to weight the channels are fixed in the summation 

        in_channels = self.in_channels
        groups = self.groups
        out_channels = self.out_channels
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups
        grad_W_groups = []

        for g in range(groups):
            # Slice the input tensor to get the channels for the current group
            start_in_ch = g * in_channels_per_group
            end_in_ch = start_in_ch + in_channels_per_group
            X_g = input_tensor[:, start_in_ch:end_in_ch, :, :]

            # Slice the output gradient tensor to get the channels for the current group
            start_out_ch = g * out_channels_per_group
            end_out_ch = start_out_ch + out_channels_per_group
            grad_output_g = grad_output[:, start_out_ch:end_out_ch, :, :]

            # To calculate the weight gradient via a convolution, we must cleverly
            # permute the input (X_g) and output gradient (grad_output_g) tensors.
            # We treat X_g as the input and grad_output_g as the kernel.
            # X_g: (N, Cin/g, H, W) -> permute -> (Cin/g, N, H, W)
            # grad_output_g: (N, Cout/g, oH, oW) -> permute -> (Cout/g, N, oH, oW)
            # The F.conv2d call then treats 'Cin/g' as the batch size and 'N' as the input channels.
            # The stride and dilation parameters from the original convolution are swapped.
            X_g_permuted = X_g.transpose(0, 1)
            grad_output_g_permuted = grad_output_g.transpose(0, 1)

            grad_W_g_permuted = F.conv2d(
                X_g_permuted,
                grad_output_g_permuted,
                stride=self.dilation,
                padding=self.padding,
                dilation=self.stride,
                groups=1 # The group calculation is handled by our loop, so this is a standard conv.
            )

            # The result has shape (Cin/g, Cout/g, kH, kW). We must permute it back to
            # the standard weight layout of (Cout/g, Cin/g, kH, kW).
            grad_W_g = grad_W_g_permuted.transpose(0, 1)
            grad_W_groups.append(grad_W_g)

        # Concatenate the gradients from all groups along the output channel dimension.
        # The weight tensor for grouped convolutions is laid out by stacking the weights
        # for each group, so we do the same for the gradient.
        grad_weight = torch.cat(grad_W_groups, dim=0)
        return grad_weight
    
    # def _calculate_gradient_weight_tensor_cheating(self,input_tensor,grad_output):
    #     return torch.nn.grad.conv2d_weight(
    #     input=input_tensor,
    #     weight_size=self.weight.tensor.shape,
    #     grad_output=grad_output,
    #     stride=self.stride,
    #     padding=self.padding,
    #     dilation=self.dilation,
    #     groups=self.groups
    #     )
#____________________________________________________________________________________________________________________________
# BATCHNORM LAYERS
# complete but not tested (gradient calculation logic has been tested)
class BatchNorm_Nd(Module):
    def __new__(cls, num_features, eps=1e-5, momentum=0.1, *, graph=None):
        assert num_features > 0
        return super().__new__(cls)
    def __init__(self, num_features, eps=1e-5, momentum=0.1, *, graph=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.graph = weakref.proxy(graph)

        self.weight = CustomTensor(torch.ones(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
        self.bias = CustomTensor(torch.zeros(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        self._channel_axis = 1  
        self._shape_cache = {}
    def _get_broadcast_shape(self, input_shape):

        if input_shape not in self._shape_cache:
            self._shape_cache[input_shape] = (1,) + (input_shape[1],) + (1,) * (len(input_shape) - 2)
        return self._shape_cache[input_shape]
    
    @torch.compile
    def _compute_stats(self, x: torch.Tensor):

        reduce_dims = tuple(i for i in range(x.dim()) if i != self._channel_axis)
        

        mean = x.mean(dim=reduce_dims, keepdim=False)
        var = x.var(dim=reduce_dims, keepdim=False, unbiased=False)
        
        return mean, var
    
    def forward(self, input_tensor):
        torch_input_tensor = input_tensor.tensor

        if self.training:
            batch_mean, batch_var = self._compute_stats(torch_input_tensor)
            total_elements = torch_input_tensor.numel() // torch_input_tensor.shape[self._channel_axis]
            unbiased_var = batch_var * total_elements / (total_elements - 1) if total_elements > 1 else batch_var
            
            self.running_mean.mul_(1-self.momentum).add_(batch_mean, alpha=self.momentum)
            self.running_var.mul_(1-self.momentum).add_(unbiased_var, alpha=self.momentum)
            
            mean, var = batch_mean, batch_var
        else:
            mean , var = self.running_mean, self.running_var
            shape_to = self._get_broadcast_shape(torch_input_tensor.shape)
            normalized = (torch_input_tensor - mean.view(shape_to)) / torch.sqrt(var.view(shape_to) + self.eps)
            result = normalized * self.weight.tensor.view(shape_to) + self.bias.tensor.view(shape_to)
            return CustomTensor(result, due_to_operation=True)
        
        shape_to = self._get_broadcast_shape(torch_input_tensor.shape)
        mean_shaped = mean.view(shape_to)
        var_shaped = var.view(shape_to)
        weight_shaped = self.weight.tensor.view(shape_to)
        bias_shaped = self.bias.tensor.view(shape_to)
        
        inv_std = torch.rsqrt(var_shaped + self.eps)  
        input_minus_mean = torch_input_tensor - mean_shaped
        normalized = input_minus_mean* inv_std
        output = normalized * weight_shaped + bias_shaped
        result = CustomTensor(output, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        
        graph = self.graph
        graph.add_edge(input_tensor._node_id, result._node_id)
        graph.add_edge(self.weight._node_id, result._node_id)
        graph.add_edge(self.bias._node_id, result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)
        weight_ref = weakref.proxy(self.weight)
        bias_ref = weakref.proxy(self.bias)

        def _backward():
            result_gradient = result_ref.tensor.grad

            if bias_ref._custom_requires_grad:
                if bias_ref.tensor.grad is None: 
                    bias_ref._zero_grad()
                grad_bias = bias_ref._reduce_grad_for_broadcast(result_gradient, shape_to)
                bias_ref.tensor.grad.add_(grad_bias.view(bias_ref.tensor.shape))

            if weight_ref._custom_requires_grad:
                if weight_ref.tensor.grad is None: 
                    weight_ref._zero_grad()
                grad_weight = weight_ref._reduce_grad_for_broadcast(result_gradient * normalized, shape_to)
                weight_ref.tensor.grad.add_(grad_weight.view(weight_ref.tensor.shape))

            if input_ref._custom_requires_grad:
                if input_ref.tensor.grad is None: 
                    input_ref._zero_grad()
                grad_input = self.batchnorm_gradient_for_input_tensor( 
                    result_gradient = result_gradient,
                    input_tensor=torch_input_tensor,
                    weight_shaped=weight_shaped,
                    input_minus_mean=input_minus_mean, 
                    inv_std=inv_std, 
                    total_elements=total_elements
                )
                input_ref.tensor.grad.add_(grad_input)

        result._backward = _backward
        return result
    @torch.compile
    def batchnorm_gradient_for_input_tensor(self,*,result_gradient, input_tensor, weight_shaped,input_minus_mean,inv_std,total_elements):
        reduce_dims = tuple(i for i in range(input_tensor.dim()) if i != self._channel_axis)

        outer_term = weight_shaped*inv_std
        term_1 = result_gradient
        term_2 = (-1/total_elements)*result_gradient.sum(dim=reduce_dims, keepdim=True)
        term3_sum_component = (input_minus_mean*result_gradient).sum(dim=reduce_dims, keepdim=True)
        term3 = inv_std**2 * (-1/total_elements) * input_minus_mean * term3_sum_component
        return outer_term * (term_1 + term_2 + term3)       
#____________________________________________________________________________________________________________________________
# POOLING LAYERS

class MaxPool2d(Module):
    def __new__(cls, *,kernel_size, stride=1, padding=0,dilation=1):
        assert isinstance(kernel_size, int) or len(kernel_size) == 2
        assert isinstance(stride, int) or len(stride) == 2
        assert isinstance(dilation, int) or len(dilation) == 2
        assert isinstance(padding, int) or len(padding) == 2
        super().__new__(cls)
    def __init__(self, *,kernel_size, stride=1, padding=0,dilation=1,graph=None):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self,input_tensor):
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        output_tensor, max_indices = F.max_pool2d(
            input=input_tensor.tensor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=True
        )
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        graph = self.graph
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=graph, is_leaf=False)
        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)
        cached_indices = max_indices

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            # max_unpool2d acts as a "gradient router", placing the incoming gradients
            # at the locations of the original maximum values.
            
            max_unpool = F.max_unpool2d(
                input=result_ref.tensor.grad,
                indices=cached_indices,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_size=input_ref.tensor.shape
            )
            input_ref.tensor.grad.add_(max_unpool)

        result._backward = _backward
        return result
    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"

class AvgPool2d(Module):

    def __new__(cls, *,kernel_size,stride=1,padding=0,graph = None):
        assert isinstance(kernel_size, int) or len(kernel_size) == 2
        assert isinstance(stride, int) or len(stride) == 2
        assert isinstance(padding, int) or len(padding) == 2
        super().__new__(cls)

    def __init__(self, *,kernel_size,stride=1,padding=0,graph = None):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self, input_tensor):
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        output_tensor = F.avg_pool2d(
            input=input_tensor.tensor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            count_include_pad=True
        )
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_output = result_ref.tensor.grad
            h_in, w_in = input_ref.shape[2], input_ref.shape[3]
            h_out, w_out = grad_output.shape[2], grad_output.shape[3]
            stride = self.stride
            padding = self.padding
            kernel_size = self.kernel_size
            # The formula relating input size to output size in a transposed convolution is:
            # InputSize = (OutputSize - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
            # We rearrange this to solve for the required output_padding.
            output_padding_h = h_in - ((h_out - 1) * stride[0] - 2 * padding[0] +  (kernel_size[0] - 1) + 1)
            output_padding_w = w_in - ((w_out - 1) * stride[1] - 2 * padding[1] +  (kernel_size[1] - 1) + 1)
            output_padding = (output_padding_h, output_padding_w)

            pool_size = self.kernel_size[0] * self.kernel_size[1]
            grad_kernel = torch.ones(grad_output.shape[1], 1, self.kernel_size[0], self.kernel_size[1]) / pool_size
            grad_input = F.conv_transpose2d(
                input= grad_output,
                weight = grad_kernel,
                stride = self.stride,
                padding = self.padding,
                output_padding=output_padding,
                groups = input_ref.tensor.shape[1] 
            )
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result 


class GlobalAvgPool2d(Module):
    def __new__(cls,*,graph=None):
        super().__new__(cls)
    def __init__(self,*,graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None
    def forward(self, input_tensor):
        output_tensor = input_tensor.tensor.mean(dim=[-2, -1], keepdim=True)
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        def _backward():
            output_grad= result_ref.tensor.grad 
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_input = output_grad/(input_ref.tensor.shape[-2] * input_ref.tensor.shape[-1])
            grad_input = grad_input.expand(input_ref.tensor.shape)
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result
    
class GlobalMaxPool2d(Module):
    def __new__(cls,*,graph=None):
        super().__new__(cls)
    def __init__(self,*,graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None
    def forward(self, input_tensor):
        output_tensor,indices = F.adaptive_max_pool2d(
            input=input_tensor.tensor,
            output_size=(1,1),
            return_indices=True
        )
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        def _backward():
            output_grad= result_ref.tensor.grad 
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_input = torch.zeros_like(input_ref.tensor)
            grad_input.view(-1)[indices.view(-1)] = output_grad.view(-1)
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result
    
#____________________________________________________________________________________________________________________________
# Activations
class ReLu(Module):
    def __init__(self,*,graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self, input_tensor):
        output_tensor = F.relu(input_tensor.tensor)
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_output = result_ref.tensor.grad
            grad_input = grad_output.clone()
            grad_input[input_ref.tensor < 0] = 0
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result
    
class Leaky_ReLu(Module):
    def __new__(cls,*,negative_slope=0.01,graph=None):
        assert negative_slope > 0
        super().__new__(cls)
    def __init__(self,*,negative_slope=0.01,graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None
        self.negative_slope = negative_slope

    def forward(self, input_tensor):
        output_tensor = F.leaky_relu(input_tensor.tensor, negative_slope=self.negative_slope)
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_output = result_ref.tensor.grad
            grad_input = grad_output.clone()
            grad_input[input_ref.tensor < 0] *= self.negative_slope
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result

class GeLu(Module):
    def __new__(cls,*,approximate='none',graph=None):
        assert approximate in {"none","tanh"}
        super().__new__(cls)
    def __init__(self,*,approximate='none',graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None
        self.approximate = approximate

    def forward(self, input_tensor):
        output_tensor = F.gelu(input_tensor.tensor,approximate=self.approximate)
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_output = result_ref.tensor.grad
            grad_input = GeLu.gelu_derivative(input_ref.tensor,grad_output,self.approximate) 
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result
    
    @torch.compile
    @staticmethod
    def gelu_derivative(x: torch.Tensor, grad_output: torch.Tensor,approximate: str) -> torch.Tensor:
        if approximate =="none":
            sqrt_2_pi = 2.5066282749176025 #torch.tensor(2 * torch.pi).sqrt()
            phi_x_cdf = 0.5 * (1 + torch.special.erf(x / 1.4142135381698608)) #torch.sqrt(torch.tensor(2.0))))
            phi_x_pdf = torch.exp(-0.5 * x**2) / sqrt_2_pi
            return (phi_x_cdf + x * phi_x_pdf) * grad_output
        else:
            sqrt_2_over_pi = 0.7978845238685608 #torch.tensor(2.0 / torch.pi).sqrt()
            coeff_cubic = 0.044715
            x2 = x.square()
            inner = x + coeff_cubic * x2 * x
            u = sqrt_2_over_pi * inner
            tanh_u = torch.tanh(u)
            poly = 1 + 3 * coeff_cubic * x2
            return (0.5 * tanh_u + 0.5 * (1 - tanh_u.square()) * (sqrt_2_over_pi * poly * x) + 0.5) * grad_output    
class Sigmoid(Module):
    def __new__(cls,*,graph=None):
        super().__new__(cls)
    def __init__(self,*,graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self, input_tensor):
        output_tensor = F.sigmoid(input_tensor.tensor)
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)
        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_output = result_ref.tensor.grad
            grad_input = grad_output * output_tensor * (1 - output_tensor)
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result

class Tanh(Module):
    def __new__(cls,*,graph=None):
        super().__new__(cls)
    def __init__(self,*,graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self,input_tensor):
        output_tensor = F.tanh(input_tensor.tensor)
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_output = result_ref.tensor.grad
            grad_input = grad_output * (1 - output_tensor**2)
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result
    
class Silu(Module):
    def __new__(cls,*,graph=None):
        super().__new__(cls)
    def __init__(self,*,graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self,input_tensor):
        output_tensor = F.silu(input_tensor.tensor)
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_output = result_ref.tensor.grad
            s_input_tensor = output_tensor/input_ref.tensor
            grad_input = grad_output*(s_input_tensor+output_tensor*(1-s_input_tensor))
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result

class Elu(Module):
    def __new__(cls,*,alpha=1.0,graph=None):
        assert alpha > 0
        super().__new__(cls)
    def __init__(self,*,alpha=1.0,graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None
        self.alpha = alpha

    def forward(self,input_tensor):
        output_tensor = F.elu(input_tensor.tensor, alpha=self.alpha)
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            grad_output = result_ref.tensor.grad
            grad_input = grad_output.clone()
            mask_neg = (input_ref.tensor.data < 0)
            grad_input[mask_neg] *= (self.alpha+output_tensor[mask_neg])
            input_ref.tensor.grad.add_(grad_input)
        result._backward = _backward
        return result

class Swish(Module):

    # TODO: implement in future
    def __new__(cls,*,B_initial=1.0,graph=None):
        assert B_initial > 0
        super().__new__(cls)
    def __init__(self,*,B_initial=1.0,graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None
        self.B = CustomTensor([B_initial], _custom_requires_grad=True, graph=graph, is_leaf=True)
        self.B_initial = B_initial
    
    def forward(self, input_tensor):
        scale = self.B.tensor.item()
        output_tensor = F.silu(scale*input_tensor.tensor)/scale
        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        graph = self.graph
        graph.add_edge(input_tensor._node_id,result._node_id)
        graph.add_edge(self.B._node_id,result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)
        B_ref = weakref.proxy(self.B)
        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            if B_ref.tensor.grad is None:
                B_ref._zero_grad()
            grad_output = result_ref.tensor.grad
            sig_B_x = output_tensor/input_ref.tensor
            common = sig_B_x*(1-sig_B_x)*grad_output

            grad_input = sig_B_x*grad_output+input_ref.tensor*B_ref.tensor*common
            grad_B = input_ref.tensor.square()*common
            input_ref.tensor.grad.add_(grad_input)
            B_ref.tensor.grad.add_(grad_B.sum())
        result._backward = _backward
        return result

