import torch
import torch.nn.functional as F
import weakref
import math

# Assume Module, CustomTensor, and AutogradGraph are defined and available.

class MaxPool2d(Module):
    """Applies a 2D max pooling over an input signal."""
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, *, graph=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.graph = graph

    def forward(self, input_tensor):
        # Use return_indices=True to get the locations of the max values for the backward pass
        output_tensor, indices = F.max_pool2d(
            input_tensor.tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            return_indices=True
        )

        if not input_tensor._custom_requires_grad:
            return CustomTensor(output_tensor, due_to_operation=True)

        # --- Autograd setup ---
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        self.graph.add_edge(input_tensor._node_id, result._node_id)
        
        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        # Cache the indices for the backward pass
        cached_indices = indices

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            
            # max_unpool2d acts as a "gradient router", placing the incoming gradients
            # at the locations of the original maximum values.
            grad_input = F.max_unpool2d(
                result_ref.tensor.grad,
                cached_indices,
                self.kernel_size,
                self.stride,
                self.padding,
                output_size=input_ref.tensor.shape
            )
            input_ref.tensor.grad.add_(grad_input)

        result._backward = _backward
        return result

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool2d(Module):
    """Applies a 2D average pooling over an input signal."""
    def __init__(self, kernel_size, stride=None, padding=0, *, graph=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.graph = graph

    def forward(self, input_tensor):
        output_tensor = F.avg_pool2d(
            input_tensor.tensor,
            self.kernel_size,
            self.stride,
            self.padding
        )

        if not input_tensor._custom_requires_grad:
            return CustomTensor(output_tensor, due_to_operation=True)

        # --- Autograd setup ---
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        self.graph.add_edge(input_tensor._node_id, result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)
        
        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            
            # The gradient for average pooling is the incoming gradient divided by the
            # pool size, distributed over the pooling window. This is equivalent to a
            # transposed convolution with a uniform kernel.
            
            # Create the uniform kernel
            pool_size = self.kernel_size * self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0] * self.kernel_size[1]
            grad_kernel = torch.ones(input_ref.tensor.shape[1], 1, self.kernel_size, self.kernel_size, device=input_ref.tensor.device) / pool_size
            
            grad_input = F.conv_transpose2d(
                result_ref.tensor.grad,
                grad_kernel,
                stride=self.stride,
                padding=self.padding,
                groups=input_ref.tensor.shape[1] # Depthwise convolution
            )
            input_ref.tensor.grad.add_(grad_input)

        result._backward = _backward
        return result

    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class GlobalAvgPool2d(Module):
    """Applies a 2D global average pooling over an input signal."""
    def __init__(self, *, graph=None):
        super().__init__()
        self.graph = graph

    def forward(self, input_tensor):
        # Global average pooling is taking the mean over the spatial dimensions
        output_tensor = input_tensor.tensor.mean(dim=[-2, -1], keepdim=True)

        if not input_tensor._custom_requires_grad:
            return CustomTensor(output_tensor, due_to_operation=True)
            
        # --- Autograd setup ---
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        self.graph.add_edge(input_tensor._node_id, result._node_id)
        
        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)

        # Cache input shape for the backward pass
        cached_input_shape = input_tensor.shape

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()

            # The gradient is distributed equally to all spatial locations
            H, W = cached_input_shape[2], cached_input_shape[3]
            grad_input = result_ref.tensor.grad / (H * W)
            # Expand the (N, C, 1, 1) gradient back to (N, C, H, W)
            input_ref.tensor.grad.add_(grad_input.expand(cached_input_shape))

        result._backward = _backward
        return result
        
    def __repr__(self):
        return "GlobalAvgPool2d()"


class GlobalMaxPool2d(Module):
    """Applies a 2D global max pooling over an input signal."""
    def __init__(self, *, graph=None):
        super().__init__()
        self.graph = graph

    def forward(self, input_tensor):
        # Global max pooling is adaptive max pooling with an output size of (1, 1)
        output_tensor, indices = F.adaptive_max_pool2d(
            input_tensor.tensor,
            output_size=(1, 1),
            return_indices=True
        )

        if not input_tensor._custom_requires_grad:
            return CustomTensor(output_tensor, due_to_operation=True)

        # --- Autograd setup ---
        result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, is_leaf=False)
        self.graph.add_edge(input_tensor._node_id, result._node_id)

        input_ref = weakref.proxy(input_tensor)
        result_ref = weakref.proxy(result)
        
        # Cache the indices and original input shape for the backward pass
        cached_indices = indices
        cached_input_shape = input_tensor.shape

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()

            # The gradient is routed only to the single max value in each channel.
            # We can use max_unpool2d, but it requires a kernel size, which isn't
            # explicit here. A simpler way is to create a zero tensor and scatter
            # the gradients using the indices.
            grad_input = torch.zeros(cached_input_shape, device=input_ref.tensor.device)
            grad_input.view(-1)[cached_indices.view(-1)] = result_ref.tensor.grad.view(-1)
            
            input_ref.tensor.grad.add_(grad_input)

        result._backward = _backward
        return result

    def __repr__(self):
        return "GlobalMaxPool2d()"
