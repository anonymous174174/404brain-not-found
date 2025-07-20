import torch
import math
import weakref
import torch.nn.functional as F
from custom_tensor import CustomTensor
from collections import OrderedDict


class Module:
    """
    Base class for all neural network modules. Your models should also subclass this class.
    Modules can also contain other Modules, allowing to nest them in a tree structure.
    """
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._buffers = OrderedDict()
        self.training = True #
        
    def __setattr__(self, name, value):
        if isinstance(value, CustomTensor):
            if value._custom_requires_grad:
                self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        # Handle buffers (non-parameter tensors like running_mean in BatchNorm)
        elif isinstance(value, torch.Tensor):
            self._buffers[name] = value
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

class Linear(Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b"""
    def __init__(self, in_features, out_features, bias=True, *, graph=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph = weakref.proxy(graph)
        
        # Using Kaiming He initialization for weights
        self.weight = CustomTensor(torch.empty(out_features, in_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
        torch.nn.init.kaiming_uniform_(self.weight.tensor, a=math.sqrt(5))
        
        if bias:
            # Using uniform initialization for bias
            self.bias = CustomTensor(torch.empty(out_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.tensor)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias.tensor, -bound, bound)
        else:
            self.bias = None

    def forward(self, input_tensor):

        output = input_tensor.matmul(self.weight.T)
        if self.bias is not None:
            output = output + self.bias
        return output
        
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

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

class BatchNorm2d(Module):
    """Applies Batch Normalization over a 4D input."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, *, graph=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.graph = graph

        # Learnable parameters
        self.weight = CustomTensor(torch.ones(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
        self.bias = CustomTensor(torch.zeros(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)

        # Buffers for running statistics
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, input_tensor):
        if self.training:
            # Calculate batch statistics
            batch_mean = input_tensor.tensor.mean(dim=[0, 2, 3])
            batch_var = input_tensor.tensor.var(dim=[0, 2, 3], unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            mean_to_use = batch_mean
            var_to_use = batch_var
        else:
            # Use running statistics during evaluation
            mean_to_use = self.running_mean
            var_to_use = self.running_var
            
        # Reshape for broadcasting
        view_shape = (1, self.num_features, 1, 1)
        mean_r = mean_to_use.view(view_shape)
        var_r = var_to_use.view(view_shape)
        weight_r = self.weight.tensor.view(view_shape)
        bias_r = self.bias.tensor.view(view_shape)

        # Normalize
        x_hat = (input_tensor.tensor - mean_r) / torch.sqrt(var_r + self.eps)
        output_tensor = weight_r * x_hat + bias_r

        # --- Autograd part ---
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
            # This backward pass is only correct for the training case.
            # A full implementation requires caching intermediate values.
            # Using PyTorch's functional backward for simplicity and correctness.
            grad_output = result_ref.tensor.grad
            
            # Using autograd on the forward pass logic to get the gradients
            # This is a simplification; a manual implementation is very complex.
            temp_input = input_ref.tensor.clone().requires_grad_()
            temp_weight = weight_ref.tensor.clone().requires_grad_()
            temp_bias = bias_ref.tensor.clone().requires_grad_()

            out = F.batch_norm(temp_input, self.running_mean, self.running_var, temp_weight, temp_bias, self.training, self.momentum, self.eps)
            out.backward(grad_output)
            
            if input_ref._custom_requires_grad:
                if input_ref.tensor.grad is None: input_ref._zero_grad()
                input_ref.tensor.grad.add_(temp_input.grad)

            if weight_ref._custom_requires_grad:
                if weight_ref.tensor.grad is None: weight_ref._zero_grad()
                weight_ref.tensor.grad.add_(temp_weight.grad)

            if bias_ref._custom_requires_grad:
                if bias_ref.tensor.grad is None: bias_ref._zero_grad()
                bias_ref.tensor.grad.add_(temp_bias.grad)

        result._backward = _backward
        return result





if __name__ == "__main__":
    pass