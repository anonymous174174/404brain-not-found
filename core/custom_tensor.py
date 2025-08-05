import torch
import torch.nn.functional as F
import weakref
import numbers
import math
from .__init__ import device, dtype
class CustomTensor:
    """
    A custom tensor class that wraps a PyTorch tensor to enable a custom
    autograd engine. It tracks operations to build a computation graph.
    """
    __slots__ = ('tensor', '_node_id', '_custom_requires_grad', '_backward', 'graph', '__weakref__','_is_leaf')

    def __new__(cls, data, *, _custom_requires_grad=False, device=device, dtype=dtype, graph=None, due_to_operation=False, is_leaf=False):
        assert device is not None
        assert dtype is not None
        if isinstance(data, CustomTensor):
            return data  # Don't rewrap
        return super().__new__(cls)

    def __init__(self, data, *, _custom_requires_grad=False, device=device, dtype=dtype, graph=None, due_to_operation=False, is_leaf=False):
        if isinstance(data, CustomTensor):
            return

        self.tensor = data if due_to_operation else torch.as_tensor(data, dtype=dtype, device=device)
        self.tensor.requires_grad_(False)
        self._custom_requires_grad = _custom_requires_grad
        self._node_id = None
        self._backward = CustomTensor._empty_backward_hook
        self.graph = None
        self._is_leaf = is_leaf

        if _custom_requires_grad:
            self._init_graph(graph)

    def _init_graph(self, graph):
        if graph is None:
            raise ValueError("Graph must be provided if requires_grad is True.")
        is_leaf=self._is_leaf
        if is_leaf:
            self.graph = weakref.proxy(graph)
        else:
            self.graph = graph # this line is only reached for tensors which are created by operations and graph passed is already a weakreference hence no need for wrapping
        graph.add_tensor_graph(self)
        if not is_leaf:
            graph.add_non_leaf_tensor_reference(self)
    @staticmethod
    def _empty_backward_hook():
      """A picklable placeholder for the backward function."""
      return None

    def clear(self):
        # NEVER CALL FOR LEAF TENSORS
        if self._is_leaf: return # ideally this line should never execute, this is just a gaurd rail
        self.tensor.grad = None
        self._custom_requires_grad = False
        self._node_id = None
        self._backward = CustomTensor._empty_backward_hook#lambda: None
        self.graph = None

    def clear_full(self):
        self.tensor = None
        self._custom_requires_grad = False
        self._node_id = None
        self._backward = CustomTensor._empty_backward_hook
        self.graph = None

    def _zero_grad(self):
        """Sets the gradient of the underlying tensor to zero."""
        if self.tensor.grad is None:
            self.tensor.grad = torch.zeros_like(self.tensor)
        else:
            self.tensor.grad.zero_()

    def zero_(self):
        """Sets the gradient of the underlying tensor to zero."""
        if self.tensor.grad is not None:
            self.tensor.grad.zero_()

    def to(self, device, dtype=None):
        if dtype is None:
            dtype = self.tensor.dtype
        self.tensor = self.tensor.to(device, dtype)
        return self


    # --- Broadcasting Helper ---
    @torch.compile
    def _reduce_grad_for_broadcast(self, grad, target_shape):
        """Reduces a gradient to match the shape of a tensor that was broadcasted."""
        if grad.shape == target_shape:
            return grad

        # Add singleton dimensions to the front of target_shape to match grad's ndim
        padded_target_shape = (1,) * (grad.ndim - len(target_shape)) + target_shape

        # Identify dimensions that were broadcasted
        sum_dims = [i for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, padded_target_shape)) if target_dim == 1 and grad_dim > 1]

        if sum_dims:
            grad = grad.sum(dim=sum_dims, keepdim=True)

        # Remove singleton dimensions to match the final target shape
        return grad.reshape(target_shape)



    def __add__(self, other):

        if isinstance(other, numbers.Number):
            return self._add_scalar(other)
        elif isinstance(other, CustomTensor):
            return self._add_tensor(other)
        return NotImplemented
    def __radd__(self,other):
        return self + other
    def __iadd__(self,other):
        if isinstance(other, numbers.Number):
            self.tensor.add_(other)
        elif isinstance(other,CustomTensor):
            self.tensor.add_(other.tensor)
    def _add_scalar(self, scalar):
        result_tensor = torch.add(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)
        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None: self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad)
        result._backward = _backward
        return result
    def _add_tensor(self, other):
        result_tensor = torch.add(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)
        graph = self.graph if self._custom_requires_grad else other.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        self_ref = weakref.proxy(self)
        other_ref = weakref.proxy(other)
        if self._custom_requires_grad:
            graph.add_edge(self._node_id, result._node_id)
        if other._custom_requires_grad:
            graph.add_edge(other._node_id, result._node_id)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref._custom_requires_grad:
                if self_ref.tensor.grad is None: self_ref._zero_grad()
                grad_for_self = self_ref._reduce_grad_for_broadcast(result_ref.tensor.grad, self_ref.tensor.shape)
                self_ref.tensor.grad.add_(grad_for_self)
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None: other_ref._zero_grad()
                grad_for_other = other_ref._reduce_grad_for_broadcast(result_ref.tensor.grad, other_ref.tensor.shape)
                other_ref.tensor.grad.add_(grad_for_other)
        result._backward = _backward
        return result

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return self._mul_scalar(other)
        elif isinstance(other, CustomTensor):
            return self._mul_tensor(other)
        return NotImplemented
    def __rmul__(self,other):
        return self*other
    def __imul__(self,other):
        if isinstance(other, numbers.Number):
            self.tensor.mul_(other)
        elif isinstance(other,CustomTensor):
            self.tensor.mul_(other.tensor)
    def _mul_scalar(self, scalar):
        result_tensor = torch.mul(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)
        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad * scalar)
        result._backward = _backward
        return result
    def _mul_tensor(self, other):
        result_tensor = torch.mul(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)
        graph = self.graph if self._custom_requires_grad else other.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        self_ref = weakref.proxy(self)
        other_ref = weakref.proxy(other)
        result_ref = weakref.proxy(result)
        if self._custom_requires_grad:
            graph.add_edge(self._node_id, result._node_id)
        if other._custom_requires_grad:
            graph.add_edge(other._node_id, result._node_id)
        def _backward():
            if self_ref._custom_requires_grad:
                if self_ref.tensor.grad is None: self_ref._zero_grad()
                grad_for_self = self_ref._reduce_grad_for_broadcast(result_ref.tensor.grad * other_ref.tensor, self_ref.tensor.shape)
                self_ref.tensor.grad.add_(grad_for_self)
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None: other_ref._zero_grad()
                grad_for_other = other_ref._reduce_grad_for_broadcast(result_ref.tensor.grad * self_ref.tensor, other_ref.tensor.shape)
                other_ref.tensor.grad.add_(grad_for_other)
        result._backward = _backward
        return result

    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            return self._sub_scalar(other)
        elif isinstance(other, CustomTensor):
            return self._sub_tensor(other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, numbers.Number):
            return self._rsub_scalar(other)

    def __isub__(self,other):
        if isinstance(other, numbers.Number):
            self.tensor.sub_(other)
        elif isinstance(other,CustomTensor):
            self.tensor.sub_(other.tensor)

    def _rsub_scalar(self, scalar):
        result_tensor = torch.sub(scalar, self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            # Derivative of scalar - x is -1
            self_ref.tensor.grad.sub_(result_ref.tensor.grad) # No broadcasting specific logic for scalar op

        result._backward = _backward
        return result


    def _sub_scalar(self, scalar):
        result_tensor = torch.sub(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad) # No broadcasting specific logic for scalar op
        result._backward = _backward
        return result

    def _sub_tensor(self, other):
        result_tensor = torch.sub(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = self.graph if self._custom_requires_grad else other.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)

        self_ref = weakref.proxy(self)
        other_ref = weakref.proxy(other)
        result_ref = weakref.proxy(result)

        if self._custom_requires_grad:
            graph.add_edge(self._node_id, result._node_id)
        if other._custom_requires_grad:
            graph.add_edge(other._node_id, result._node_id)

        def _backward():
            if self_ref._custom_requires_grad:
                if self_ref.tensor.grad is None:
                    self_ref._zero_grad()
                grad_for_self = self_ref._reduce_grad_for_broadcast(result_ref.tensor.grad, self_ref.tensor.shape)
                self_ref.tensor.grad.add_(grad_for_self)
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                grad_for_other = other_ref._reduce_grad_for_broadcast(-result_ref.tensor.grad, other_ref.tensor.shape)
                other_ref.tensor.grad.add_(grad_for_other)
        result._backward = _backward
        return result

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return self._div_scalar(other)
        elif isinstance(other, CustomTensor):
            return self._div_tensor(other)
        return NotImplemented
    def __itruediv__(self,other):
        if isinstance(other, numbers.Number):
            self.tensor.div_(other)
        elif isinstance(other,CustomTensor):
            self.tensor.div_(other.tensor)
    def _div_scalar(self, scalar):
        result_tensor = torch.div(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad / scalar)
        result._backward = _backward
        return result

    def _div_tensor(self,other):
        result_tensor = torch.div(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = self.graph if self._custom_requires_grad else other.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)

        self_ref = weakref.proxy(self)
        other_ref = weakref.proxy(other)
        result_ref = weakref.proxy(result)

        if self._custom_requires_grad:
            graph.add_edge(self._node_id, result._node_id)
        if other._custom_requires_grad:
            graph.add_edge(other._node_id, result._node_id)

        def _backward():
            if self_ref._custom_requires_grad:
                if self_ref.tensor.grad is None:
                    self_ref._zero_grad()
                grad_for_self = self_ref._reduce_grad_for_broadcast(result_ref.tensor.grad / other_ref.tensor, self_ref.tensor.shape)
                self_ref.tensor.grad.add_(grad_for_self)
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                grad_for_other = other_ref._reduce_grad_for_broadcast(-result_ref.tensor.grad * self_ref.tensor / other_ref.tensor.pow(2), other_ref.tensor.shape)
                other_ref.tensor.grad.add_(grad_for_other)
        result._backward = _backward
        return result

    def pow(self, scalar):
        result_tensor = torch.pow(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            grad_contrib = scalar * self_ref.tensor.pow(scalar - 1)
            self_ref.tensor.grad.add_(result_ref.tensor.grad * grad_contrib)
        result._backward = _backward
        return result
    def __ipow__(self,other):
        self.tensor.pow_(other)
    def __pow__(self,other):
      if isinstance(other, numbers.Number):
          return self.pow(other)
      return NotImplemented
    def exp(self):
        out = torch.exp(self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(out,due_to_operation=True)

        graph = self.graph
        result = CustomTensor(out, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad * out)
        result._backward = _backward
        return result

    def log(self):
        out = torch.log(self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(out,due_to_operation=True)

        graph = self.graph
        result = CustomTensor(out, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad / self_ref.tensor)
        result._backward = _backward
        return result

    def sin(self):
        out = torch.sin(self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(out,due_to_operation=True)

        graph = self.graph
        result = CustomTensor(out, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad * torch.cos(self_ref.tensor))
        result._backward = _backward
        return result

    def cos(self):
        out = torch.cos(self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(out,due_to_operation=True)

        graph = self.graph
        result = CustomTensor(out, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(-result_ref.tensor.grad*torch.sin(self_ref.tensor))
        result._backward = _backward
        return result

    def sqrt(self):
        out = torch.sqrt(self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(out,due_to_operation=True)

        graph = self.graph
        result = CustomTensor(out, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad*0.5*self_ref.tensor.pow(-0.5))
        result._backward = _backward
        return result
    def __matmul__(self,other):
        if isinstance(other, CustomTensor):
            return self.matmul(other)
        return NotImplemented
    def matmul(self, other):
        result_tensor = torch.matmul(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = self.graph if self._custom_requires_grad else other.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)

        self_ref = weakref.proxy(self)
        other_ref = weakref.proxy(other)
        result_ref = weakref.proxy(result)

        if self._custom_requires_grad:
            graph.add_edge(self._node_id, result._node_id)
        if other._custom_requires_grad:
            graph.add_edge(other._node_id, result._node_id)

        def _backward():
            if self_ref._custom_requires_grad:
                if self_ref.tensor.grad is None: self_ref._zero_grad()
                # Use robust broadcasting for matmul gradient
                grad_for_self = torch.matmul(result_ref.tensor.grad, other_ref.tensor.transpose(-2, -1))
                self_ref.tensor.grad.add_(self_ref._reduce_grad_for_broadcast(grad_for_self, self_ref.tensor.shape))
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None: other_ref._zero_grad()
                grad_for_other = torch.matmul(self_ref.tensor.transpose(-2, -1), result_ref.tensor.grad)
                other_ref.tensor.grad.add_(other_ref._reduce_grad_for_broadcast(grad_for_other, other_ref.tensor.shape))
        result._backward = _backward
        return result
    def dot(self, other):
        # torch.dot only works for 1D tensors, or for higher-D tensors,
        # it flattens them to 1D and then computes the dot product.
        # This means the gradients will also be 1D, so no complex broadcasting
        # reduction is needed on the output gradient itself.
        # However, the input tensors themselves could have been results of broadcasting ops.
        # For a truly general dot product, you'd use torch.matmul.
        result_tensor = torch.dot(self.tensor.reshape(-1), other.tensor.reshape(-1))
        requires_grad = self._custom_requires_grad or other._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = self.graph if self._custom_requires_grad else other.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)

        self_ref = weakref.proxy(self)
        other_ref = weakref.proxy(other)
        result_ref = weakref.proxy(result)

        if self._custom_requires_grad:
            graph.add_edge(self._node_id, result._node_id)
        if other._custom_requires_grad:
            graph.add_edge(other._node_id, result._node_id)

        def _backward():
            if self_ref._custom_requires_grad:
                if self_ref.tensor.grad is None:
                    self_ref._zero_grad()
                # The grad from result_ref.tensor.grad will be a scalar.
                # It needs to be multiplied by the other_ref.tensor (original shape)
                # and then potentially re-shaped if original was >1D
                grad_contrib = result_ref.tensor.grad * other_ref.tensor
                self_ref.tensor.grad.add_(grad_contrib)
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                grad_contrib = result_ref.tensor.grad * self_ref.tensor
                other_ref.tensor.grad.add_(grad_contrib)
        result._backward = _backward
        return result



    # --- Unary Operations ---

    def sum(self, dim=None, keepdim=False):
        """Computes the sum of elements along given dimensions."""
        result_tensor = self.tensor.sum(dim=dim, keepdim=keepdim)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)

        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()

            grad = result_ref.tensor.grad
            # If keepdim was false, the summed dim was squeezed. We need to unsqueeze it back for broadcasting.
            if not keepdim and dim is not None:
                grad = grad.unsqueeze(dim)

            self_ref.tensor.grad.add_(grad)

        result._backward = _backward
        return result

    def mean(self, dim=None, keepdim=False):
        """Computes the mean of elements along given dimensions."""
        result_tensor = self.tensor.mean(dim=dim, keepdim=keepdim)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)

        # Determine the number of elements that were averaged
        if dim is None:
            n = self.tensor.numel()
        else:
            n = self.tensor.shape[dim]

        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()

            grad = result_ref.tensor.grad
            if not keepdim and dim is not None:
                grad = grad.unsqueeze(dim)

            # Distribute gradient evenly
            self_ref.tensor.grad.add_(grad / n)

        result._backward = _backward
        return result

    def reshape(self, *shape):
        """Reshapes the tensor to the given shape."""
        original_shape = self.shape
        result_tensor = self.tensor.reshape(*shape)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)

        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad.reshape(original_shape))

        result._backward = _backward
        return result

    def transpose(self, dim0, dim1):
        """Transposes dimensions dim0 and dim1."""
        result_tensor = self.tensor.transpose(dim0, dim1)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)

        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            # The gradient operation for transpose is another transpose
            self_ref.tensor.grad.add_(result_ref.tensor.grad.transpose(dim0, dim1))

        result._backward = _backward
        return result

    @property
    def T(self):
        """Alias for transpose(-2, -1) for 2D or higher dimensional tensors."""
        if self.ndim < 2:
            raise ValueError("`.T` is only supported on tensors with 2 or more dimensions.")
        return self.transpose(-2, -1)

    def backward(self, weightage_tensor=1,retain_graph=False):
        if not self._custom_requires_grad:
            raise RuntimeError("Output tensor does not require grad.")
        if self.graph is None:
            raise RuntimeError("Output tensor is not part of a graph.")
        graph = self.graph

        # Initialize gradient for the output tensor
        if isinstance(weightage_tensor, numbers.Number):
            self.tensor.grad = torch.full_like(self.tensor, fill_value=weightage_tensor)
        elif isinstance(weightage_tensor, torch.Tensor):
            self.tensor.grad = weightage_tensor.to(self.tensor.device)#.clone()

        nodes_to_process = graph.reverse_toposort_from_tensor(self._node_id)

        for tensor_node in nodes_to_process:
            tensor_node._backward()
        if not retain_graph:
            graph.delete_all_non_leaf_nodes()

            #try:
                # The node is a weakref.proxy, check if it's still alive
                #if tensor_node.__class__ is weakref.ProxyType:
            #        tensor_node._backward()
            # except ReferenceError:
            #     # The tensor object was garbage collected, skip.
            #     print("dead reference node encountered")
            #     continue
    # --- Properties and Dunder Methods ---
    @property
    def dtype(self): return self.tensor.dtype
    @property
    def ndim(self): return self.tensor.ndim
    @property
    def shape(self): return self.tensor.shape
    @property
    def grad(self): return self.tensor.grad
    def __repr__(self): return f"CustomTensor({self.tensor}, grad_fn={self._backward != None}, requires_grad={self._custom_requires_grad})"
    # def __del__(self):
    #     if self._node_id is not None and self._is_leaf:
    #         try:
    #             if self.graph: self.graph.delete_node(self._node_id)
    #         except ReferenceError: # Graph might be gone first
    #             pass
if __name__ == "__main__":
    pass