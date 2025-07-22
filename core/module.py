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
        self.graph = weakref.proxy(graph)
        
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
class Linear_with_activation(Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b and activation to it
    types of activation relu,leaky_relu, gelu, sigmoid, tanh, silu,elu"""
    def __new__(cls, in_features, out_features, bias=True, *, graph=None,activation="relu"):
        assert activation in {"relu", "gelu", "leaky_relu", "sigmoid", "tanh", "silu", "elu", "gelu_approx"}
        return super().__new__(cls)
    def __init__(self, in_features, out_features, bias=True, *, graph=None,activation="relu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph = weakref.proxy(graph)
        self.activation = activation
        
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
        
        activation = self.activation
        pre_activation = input_tensor.tensor @ self.weight.tensor.transpose(-2, -1)
        if self.bias is not None:
            pre_activation.add_(self.bias.tensor)

        output = ACTIVATIONS[activation](pre_activation)
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
            d_activation_d_pre_activation = ACTIVATIONS_GRADIENT[activation](pre_activation)
            pre_activation_grad = d_activation_d_pre_activation*result_ref.tensor.grad
            if weight_ref._custom_requires_grad:
                if weight_ref.tensor.grad is None:
                    weight_ref._zero_grad()
                
                grad_w = torch.matmul(pre_activation_grad.transpose(-2, -1), input_tensor_ref.tensor)
                weight_ref.tensor.grad.add_(weight_ref._reduce_grad_for_broadcast(grad_w, weight_ref.tensor.shape))
            

            if bias_ref is not None:
                if bias_ref._custom_requires_grad:
                    if bias_ref.tensor.grad is None:
                        bias_ref._zero_grad()
                    grad_b = bias_ref._reduce_grad_for_broadcast(pre_activation_grad, bias_ref.tensor.shape)
                    bias_ref.tensor.grad.add_(grad_b)
            
            if input_tensor_ref._custom_requires_grad:
                if input_tensor_ref.tensor.grad is None:
                    input_tensor_ref._zero_grad()
                grad_in = torch.matmul(pre_activation_grad, weight_ref.tensor)
                input_tensor_ref.tensor.grad.add_(input_tensor_ref._reduce_grad_for_broadcast(grad_in, input_tensor_ref.tensor.shape))
        
        result.backward = _backward
        return result

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

# class Linear_with_batch_norm_and_activations(Module):
#     """Applies a linear transformation to the incoming data: y = xA^T + b and batchnorm then activation
#     types of activation relu,leaky_relu, gelu, sigmoid, tanh, silu,elu, gelu_approx"""
#     def __new__(cls, in_features, out_features, bias=True, *, graph=None,activation="relu"):
#         assert activation in {"relu", "gelu", "leaky_relu", "sigmoid", "tanh", "silu", "elu", "gelu_approx"}
#         return super().__new__(cls)
#     def __init__(self, in_features, out_features, bias=True, *, graph=None,activation="relu"):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.graph = weakref.proxy(graph)
        
#         self.linear = Linear(in_features, out_features, bias=bias, graph=graph,activation=activation)

#____________________________________________________________________________________________________________________________
#CONVOLUTION LAYERS ✅ Conv → BatchNorm → ReLU → MaxPool
class Conv2d(Module):
    """Applies a 2D convolution over an input signal composed of several input planes."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *, graph=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.graph = graph

        # Weight and bias initialization
        self.weight = CustomTensor(torch.empty(out_channels, in_channels, *self.kernel_size), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
        self.bias = CustomTensor(torch.empty(out_channels), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
        torch.nn.init.kaiming_uniform_(self.weight.tensor, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.tensor)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias.tensor, -bound, bound)

    def forward(self, input_tensor):
        # We can use torch.nn.functional.conv2d and define a custom backward pass.
        # This is much simpler than implementing im2col/col2im manually.
        output_tensor = F.conv2d(input_tensor.tensor, self.weight.tensor, self.bias.tensor, self.stride, self.padding)

        requires_grad = input_tensor._custom_requires_grad or self.weight._custom_requires_grad
        if not requires_grad:
            return CustomTensor(output_tensor, due_to_operation=True)
        
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, due_to_operation=True, is_leaf=False)
        self.graph.add_edge(input_tensor._node_id, result._node_id)
        self.graph.add_edge(self.weight._node_id, result._node_id)
        self.graph.add_edge(self.bias._node_id, result._node_id)
        
        input_ref = weakref.proxy(input_tensor)
        weight_ref = weakref.proxy(self.weight)
        bias_ref = weakref.proxy(self.bias)
        result_ref = weakref.proxy(result)

        def _backward():
            grad_output = result_ref.tensor.grad
            
            # Gradient for input
            if input_ref._custom_requires_grad:
                if input_ref.tensor.grad is None: input_ref._zero_grad()
                input_ref.tensor.grad.add_(
                    torch.nn.grad.conv2d_input(input_ref.tensor.shape, weight_ref.tensor, grad_output, self.stride, self.padding)
                )

            # Gradient for weights
            if weight_ref._custom_requires_grad:
                if weight_ref.tensor.grad is None: weight_ref._zero_grad()
                weight_ref.tensor.grad.add_(
                    torch.nn.grad.conv2d_weight(input_ref.tensor, weight_ref.tensor.shape, grad_output, self.stride, self.padding)
                )

            # Gradient for bias
            if bias_ref._custom_requires_grad:
                if bias_ref.tensor.grad is None: bias_ref._zero_grad()
                bias_ref.tensor.grad.add_(grad_output.sum(dim=[0, 2, 3]))

        result._backward = _backward
        return result
#____________________________________________________________________________________________________________________________
# BATCHNORM LAYERS
class BatchNorm_Nd(Module):
    def __new__(cls, num_features, eps=1e-5, momentum=0.1, *, graph=None):
        assert num_features > 0
        return super().__new__(cls)
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, *, graph=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.graph = graph

        self.weight = CustomTensor(torch.ones(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
        self.bias = CustomTensor(torch.zeros(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self._channel_axis = 1
        self._shape_cache = {}
    
    def _get_broadcast_shape(self, input_shape):

        if input_shape not in self._shape_cache:
            self._shape_cache[input_shape] = (1,) + (input_shape[1],) + (1,) * (len(input_shape) - 2)
        return self._shape_cache[input_shape]
    
    @torch.jit.script_method if hasattr(torch.jit, 'script_method') else lambda x: x
    def _compute_stats(self, x: torch.Tensor):

        reduce_dims = tuple(i for i in range(x.dim()) if i != self._channel_axis)
        

        mean = x.mean(dim=reduce_dims, keepdim=False)
        var = x.var(dim=reduce_dims, keepdim=False, unbiased=False)
        
        return mean, var
    
    def forward(self, input_tensor):
        x = input_tensor.tensor
        
        if self.training:

            batch_mean, batch_var = self._compute_stats(x)
            

            n = x.numel() // x.shape[self._channel_axis]
            unbiased_var = batch_var * n / (n - 1) if n > 1 else batch_var
            

            self.running_mean.mul_(1 - self.momentum).add_(batch_mean, alpha=self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)
            
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

            shape = self._get_broadcast_shape(x.shape)
            normalized = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
            output = normalized * self.weight.tensor.view(shape) + self.bias.tensor.view(shape)
            return CustomTensor(output, due_to_operation=True)
        

        return self._forward_with_gradients(input_tensor, x, mean, var)
    
    def _forward_with_gradients(self, input_tensor, x, mean, var):

        shape = self._get_broadcast_shape(x.shape)
        

        mean_shaped = mean.view(shape)
        var_shaped = var.view(shape)
        weight_shaped = self.weight.tensor.view(shape)
        bias_shaped = self.bias.tensor.view(shape)
        

        inv_std = torch.rsqrt(var_shaped + self.eps)  
        centered = x - mean_shaped
        normalized = centered * inv_std
        output = normalized * weight_shaped + bias_shaped
        
        result = CustomTensor(output, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
    
        graph = self.graph
        graph.add_edge(input_tensor._node_id, result._node_id)
        graph.add_edge(self.weight._node_id, result._node_id)
        graph.add_edge(self.bias._node_id, result._node_id)

        # Store references for backward pass
        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)
        weight_ref = weakref.proxy(self.weight)
        bias_ref = weakref.proxy(self.bias)
        
        # Pre-compute values needed for backward pass
        n = x.numel() // x.shape[self._channel_axis]
        
        def _backward():
            grad_out = result_ref.tensor.grad
            
            # Bias gradient (most efficient)
            if bias_ref._custom_requires_grad:
                if bias_ref.tensor.grad is None: 
                    bias_ref._zero_grad()
                grad_bias = bias_ref._reduce_grad_for_broadcast(grad_out, shape)
                bias_ref.tensor.grad.add_(grad_bias.view(bias_ref.tensor.shape))

            # Weight gradient
            if weight_ref._custom_requires_grad:
                if weight_ref.tensor.grad is None: 
                    weight_ref._zero_grad()
                grad_weight = weight_ref._reduce_grad_for_broadcast(grad_out * normalized, shape)
                weight_ref.tensor.grad.add_(grad_weight.view(weight_ref.tensor.shape))

            # Input gradient (most expensive)
            if input_ref._custom_requires_grad:
                if input_ref.tensor.grad is None: 
                    input_ref._zero_grad()
                grad_input = self._compute_input_gradient(
                    grad_out, x, normalized, centered, weight_shaped, inv_std, n
                )
                input_ref.tensor.grad.add_(grad_input)

        result._backward = _backward
        return result
    
    @torch.jit.script_method if hasattr(torch.jit, 'script_method') else lambda x: x
    def _compute_input_gradient(self, grad_out: torch.Tensor, x: torch.Tensor, 
                                centered: torch.Tensor,weight_shaped: torch.Tensor, 
                                inv_std: torch.Tensor, n: int):
        """Optimized input gradient computation using efficient vectorized operations"""
        reduce_dims = tuple(i for i in range(x.dim()) if i != self._channel_axis)
        
        # More efficient gradient computation
        # Based on: https://arxiv.org/abs/1502.03167 (Batch Normalization paper)
        grad_normalized = grad_out * weight_shaped
        
        # Compute gradient components more efficiently
        sum_grad_normalized = grad_normalized.sum(dim=reduce_dims, keepdim=True)
        sum_grad_normalized_centered = (grad_normalized * centered).sum(dim=reduce_dims, keepdim=True)
        
        # Final gradient computation
        grad_input = inv_std * (
            grad_normalized 
            - sum_grad_normalized / n
            - centered * inv_std**2 * sum_grad_normalized_centered / n
        )
        
        return grad_input
# class BatchNorm_Nd(Module):
#     def __new__(cls, num_features, eps=1e-5, momentum=0.1, *, graph=None):
#         assert num_features > 0
#         return super().__new__(cls)
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, *, graph=None):
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.graph = weakref.proxy(graph)

#         self.weight = CustomTensor(torch.ones(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
#         self.bias = CustomTensor(torch.zeros(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)

#         self.running_mean = torch.zeros(num_features)
#         self.running_var = torch.ones(num_features)

#     def forward(self, input_tensor):
#         torch_input_tensor = input_tensor.tensor
#         if self.training:
#             total_elements = torch_input_tensor.numel() // torch_input_tensor.shape[1]
#             # total_elements = 0
#             # for i,size in enumerate(torch_input_tensor.shape):
#             #     if i != 1:
#             #         total_elements *= size
#             basel_correction_factor = total_elements / (total_elements - 1) if total_elements > 1 else 1

#             batch_mean = torch_input_tensor.mean(dim=[i for i in range(torch_input_tensor.dim()) if i != 1])
#             batch_var = torch_input_tensor.var(dim=[i for i in range(torch_input_tensor.dim()) if i != 1], unbiased=False)


#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean 
#             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var * basel_correction_factor

#             mean_to_use = batch_mean
#             var_to_use = batch_var
#         else:

#             mean_to_use = self.running_mean
#             var_to_use = self.running_var

#         shape_to =(1,) + (torch_input_tensor.shape[1],) + (1,) * (len(torch_input_tensor.shape) - 2)
#         mean_to_use = mean_to_use.reshape(shape_to)
#         var_to_use = var_to_use.reshape(shape_to)
#         weight_to_use = self.weight.tensor.reshape(shape_to)
#         bias_to_use = self.bias.tensor.reshape(shape_to)
#         normalizing_factor =torch.sqrt(var_to_use + self.eps)
#         normalized_tensor = (torch_input_tensor - mean_to_use) / normalizing_factor

#         # Apply weight and bias
#         output_tensor = normalized_tensor * weight_to_use + bias_to_use
#         if not self.training:
#             return CustomTensor(output_tensor, due_to_operation=True)
#         weight_to_use = None
#         bias_to_use = None
#         normalizing_factor = None
#         shape_to = None
#         graph = self.graph
#         result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=graph, is_leaf=False)
#         graph.add_edge(input_tensor._node_id, result._node_id)
#         graph.add_edge(self.weight._node_id, result._node_id)
#         graph.add_edge(self.bias._node_id, result._node_id)

#         input_ref = weakref.proxy(input_tensor)
#         result_ref = weakref.proxy(result)
#         weight_ref = weakref.proxy(self.weight)
#         bias_ref = weakref.proxy(self.bias)
#         def _backward():
#             if weight_ref._custom_requires_grad:
#                 if weight_ref.tensor.grad is None: weight_ref._zero_grad()
#                 grad_weight = weight_ref._reduce_grad_for_broadcast(result_ref.tensor.grad * normalized_tensor, shape_to)
#                 grad_weight = grad_weight.reshape(weight_ref.tensor.shape)
#                 weight_ref.tensor.grad.add_(grad_weight)

#             if bias_ref._custom_requires_grad:
#                 if self.bias.tensor.grad is None: self.bias._zero_grad()
#                 grad_bias = bias_ref._reduce_grad_for_broadcast(result_ref.tensor.grad, shape_to)
#                 grad_bias = grad_bias.reshape(bias_ref.tensor.shape)
#                 self.bias.tensor.grad.add_(grad_bias)
#             # Let the deep dive into math hell begin
#             # the 3 sources of gradients for the input tensor are normalized tensor, mean and variance 
#             if input_ref._custom_requires_grad:
#                 if input_ref.tensor.grad is None: input_ref._zero_grad()
#                 grad_input = BatchNorm_Nd.batchnorm_gradient_for_input_tensor(
#                     incoming_gradient=result_ref.tensor.grad,
#                     input_tensor=torch_input_tensor,
#                     weight=weight_ref.tensor,
#                     mean=mean_to_use,
#                     var=var_to_use,
#                     eps=self.eps,
#                 )
#                 input_ref.tensor.grad.add_(grad_input)

#         result._backward = _backward
#         return result
#     @staticmethod
#     @torch.compile
#     def batchnorm_gradient_for_input_tensor(*,incoming_gradient,input_tensor,weight,mean,var,eps):
#         dims_to_reduce = [i for i in range(input_tensor.dim()) if i != 1]
#         shape_to_broadcast = (1,) + (input_tensor.shape[1],) + (1,) * (len(input_tensor.shape) - 2)
#         total_elements = input_tensor.numel() // input_tensor.shape[1]
#         normalizing_factor = torch.sqrt(var + eps)
#         weight_reshaped = weight.reshape(shape_to_broadcast)
#         mean_reshaped = mean.reshape(shape_to_broadcast)
#         normalizing_factor_reshaped = normalizing_factor.reshape(shape_to_broadcast)
#         outer_term = weight_reshaped / normalizing_factor_reshaped
#         term1 = incoming_gradient
#         term2 = (-1 / total_elements) * incoming_gradient.sum(dim=dims_to_reduce, keepdim=True)
#         term3_sum_component = (incoming_gradient * (input_tensor - mean_reshaped)).sum(dim=dims_to_reduce, keepdim=True)
#         term3 = (-1 / (total_elements * normalizing_factor_reshaped**2)) * (input_tensor - mean_reshaped) * term3_sum_component
#         grad_input = outer_term * (term1 + term2 + term3)
#         return grad_input
#     # @staticmethod
#     # def batchnorm_gradient_for_input_tensor(*,incoming_gradient,input_tensor, mean_to_use,shape_to,normalizing_factor,weight_to_use,total_elements):
#     #     outer_term = weight_to_use/normalizing_factor
#     #     grad_input = outer_term * (incoming_gradient
#     #                                (-1/total_elements)
#     #                                *incoming_gradient.sum(dim=[i for i in range(incoming_gradient.dim()) if i != 1]).reshape(shape_to)
#     #                                - 1/(total_elements*normalizing_factor**2)
#     #                                *(input_tensor - mean_to_use)
#     #                                *(incoming_gradient*(input_tensor - mean_to_use)).sum(dim=[i for i in range(incoming_gradient.dim()) if i != 1]).reshape(shape_to))    
#     # #    (weight_to_use/normalizing_factor)*(incoming_gradient - 1/total_elements * incoming_gradient.sum(dim=[i for i in range(input_tensor.dim()) if i != 1]).reshape(shape_to) - 1/(total_elements*normalizing_factor**2)*(input_tensor - mean_to_use)*incoming_gradient*(input_tensor - mean_to_use).sum(dim=[i for i in range(input_tensor.dim()) if i != 1]).reshape(shape_to))    
#     #     return grad_input
