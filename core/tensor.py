from typing import Union, Optional, Tuple, Any
import torch
from . import RUN_ON_CPU, RUN_ON_GPU, RUN_ON_TPU, device

class Tensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
882e32



class CustomTensor(torch.Tensor):
#     CustomTensor Instance (Python object)
# │
# ├── PyTorch Native Tensor (C++ data)
# │   ├── storage: [1.0] (float32)
# │   ├── native requires_grad: False
# │
# ├── Python attributes:
# │   ├── requires_grad: True (your custom flag)
# │   ├── grad: None
# │   └── operation: None
# to safely create CustomTensor object of a pytorch tensor it is preffered to do CustomTensor(_tensor=pytorch_tensor);del pytorch_tensor
# this prevent modification of Custom Tensor object by pytorch autograd if the pytorch tensor is updates by autograd
    def __new__(cls, data=None, requires_grad=False, operation=None, _tensor=None):
        """
        Custom __new__ method for CustomTensor.
        It wraps a torch.Tensor instance, allowing you to leverage PyTorch's
        optimized kernels.

        Args:
            data: The data for the tensor (can be a list, numpy array, or torch.Tensor).
                  If _tensor is provided, data is ignored.
            requires_grad (bool): Whether this tensor requires gradients for custom autograd.
            operation (Operation): The operation that produced this tensor,
                                   used for building the computation graph.
            _tensor (torch.Tensor, optional): An existing torch.Tensor to wrap.
                                              If provided, 'data' is ignored.
        """
        if _tensor is not None:
            # If an existing torch.Tensor is provided, wrap it directly
            # This is crucial for returning CustomTensor instances from operations
            # that produce new torch.Tensors.
            _tensor = _tensor.detach()
            instance = torch.Tensor._make_subclass(cls, _tensor, requires_grad)
        else:
            # Otherwise, create a new torch.Tensor from the provided data
            # and then wrap it.
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data, dtype=torch.float32) # Ensure float type

            # Use torch.Tensor._make_subclass to create an instance of CustomTensor
            # that wraps the underlying torch.Tensor data.
            # We explicitly set requires_grad for the underlying torch.Tensor to False
            # because we are building a custom autograd system.
            instance = torch.Tensor._make_subclass(cls, data, requires_grad=False)

        instance.requires_grad = requires_grad
        instance.grad = None  # Initialize gradient
        instance.operation = operation # Store the operation that created this tensor
        return instance

    def __repr__(self):
        # Custom representation for better debugging
        return f"CustomTensor({super().__repr__()}, requires_grad={self.requires_grad}, grad={self.grad})"

    def zero_grad(self):
        """Zeros out the gradient of this tensor."""
        if self.grad is not None:
            self.grad.zero_()

    def backward(self, grad_output=None):
        """
        Performs backpropagation for this tensor.
        This is a simplified version; a full autograd would involve a topological sort.
        """
        if not self.requires_grad:
            print("Warning: Calling backward on a tensor that does not require grad.")
            return

        if grad_output is None:
            # Default gradient for scalar tensors
            if self.numel() == 1:
                grad_output = torch.tensor(1.0, dtype=self.dtype)
            else:
                raise RuntimeError("grad_output must be specified for non-scalar tensors")

        if self.grad is None:
            self.grad = grad_output
        else:
            self.grad += grad_output

        # Perform a simple depth-first traversal for backpropagation.
        # For a complete autograd, you'd build a graph and process nodes
        # in reverse topological order.
        if self.operation is not None:
            self.operation.backward(grad_output)


# Define a simple Operation class for your custom autograd graph
# This would track the function performed and its inputs for backpropagation.
class Operation:
    def __init__(self, inputs=()):
        self.inputs = inputs # Tensors involved in this operation

    def forward(self, *args):
        # This method would perform the actual computation.
        # For a custom autograd, you typically don't implement 'forward' here
        # in the Operation class, but rather within the Tensor's dunder methods
        # that create this operation.
        raise NotImplementedError

    def backward(self, grad_output):
        # This method computes and propagates gradients to inputs.
        raise NotImplementedError

# Example Operation: Addition
class Add(Operation):
    def __init__(self, input1, input2):
        super().__init__((input1, input2))

    def backward(self, grad_output):
        # For addition, the gradient is simply propagated to both inputs.
        # Make sure to handle cases where inputs might be None or already have gradients.
        if self.inputs[0].grad is None:
            self.inputs[0].grad = grad_output
        else:
            self.inputs[0].grad += grad_output

        if self.inputs[1].grad is None:
            self.inputs[1].grad = grad_output
        else:
            self.inputs[1].grad += grad_output


# Define your custom Tensor class