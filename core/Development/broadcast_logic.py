import torch
import torch.nn.functional as F
import weakref
import numbers
import rustworkx as rx
import pytest
import math
from collections import OrderedDict

# Your existing AutogradGraph class (with minor improvements)
class AutogradGraph:
    """
    Manages the computation graph for automatic differentiation.
    It uses a directed acyclic graph to track dependencies between tensors.
    """
    __slots__ = ('graph', 'intermediate_tensors', '_check_cycles', '_auto_cleanup', '__weakref__')

    def __init__(self, check_for_cycles=True, auto_cleanup=True):
        self.graph = rx.PyDiGraph()
        self.intermediate_tensors = {}
        self._check_cycles = check_for_cycles
        self._auto_cleanup = auto_cleanup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._check_cycles and self.check_cycle():
            raise RuntimeError("Cycle detected in autograd graph on context exit.")
        if self._auto_cleanup:
            self.intermediate_tensors.clear()
            self.graph.clear()

    def add_tensor_graph(self, tensor):
        if not tensor._custom_requires_grad:
            raise ValueError("Tensor with requires_grad=False cannot be added to the graph.")
        ref = weakref.proxy(tensor)
        tensor_index = self.graph.add_node(ref)
        tensor._node_id = tensor_index

    def add_non_leaf_tensor_reference(self, tensor):
        if not tensor._custom_requires_grad:
            raise ValueError("Tensor must require grad.")
        if tensor._node_id in self.intermediate_tensors:
            raise ValueError("Tensor reference already exists in intermediate tensors.")
        self.intermediate_tensors[tensor._node_id] = tensor

    def add_edge(self, node_from, node_to, weight=None):
        if not all(isinstance(n, int) for n in (node_from, node_to)):
            raise TypeError("Node indices must be integers.")
        if not self.graph.has_node(node_from) or not self.graph.has_node(node_to):
            raise ValueError("Nodes must exist before adding edge.")
        self.graph.add_edge(node_from, node_to, weight)

    def check_cycle(self):
        return not rx.is_directed_acyclic_graph(self.graph)

    def reverse_toposort_from_tensor(self, tensor_index):
        graph=self.graph
        predecessors = list(rx.ancestors(graph, tensor_index))
        predecessors.append(tensor_index)
        sub_graph = graph.subgraph(predecessors)
        return [sub_graph[i] for i in reversed(rx.topological_sort(sub_graph))]
    # def alternative_reverse_toposort_from_tensor(self, tensor_index):
    #     graph = self.graph
    #     relevant_nodes = rx.ancestors(graph, tensor_index)
    #     relevant_nodes.add(tensor_index)
    #     full_topo = rx.topological_sort(graph)
    #     relevant_topo = [graph[_node_id] for _node_id in reversed(full_topo) if _node_id in relevant_nodes]
    #     return relevant_topo

    def delete_node(self, node_index):
        if not isinstance(node_index, int):
            raise TypeError("Node index must be an integer.")
        if self.graph.has_node(node_index):
             self.graph.remove_node(node_index)
    def delete_edge(self, node_from, node_to):
        if not self.graph.has_edge(node_from, node_to):
            raise ValueError("Edge does not exist.")
        self.graph.remove_edge(node_from, node_to)

    def del_non_leaf_tensor_reference(self, tensor_node_id):
        self.intermediate_tensors.pop(tensor_node_id, None)

    def delete_all_non_leaf_nodes(self):
        # removes non leaf nodes from graph and clears the intermediate_tensors dict
        self.graph.remove_nodes_from(list(self.intermediate_tensors.keys()))
        self.intermediate_tensors.clear()

    def __repr__(self):
        return f"CustomAutogradGraph(nodes={self.graph.num_nodes()}, edges={self.graph.num_edges()})"

# Your existing CustomTensor class, now enhanced with new methods
class CustomTensor:
    """
    A custom tensor class that wraps a PyTorch tensor to enable a custom
    autograd engine. It tracks operations to build a computation graph.
    """
    __slots__ = ('tensor', '_node_id', '_custom_requires_grad', '_backward', 'graph', '__weakref__','_is_leaf')

    def __new__(cls, data, *, _custom_requires_grad=False, device=None, dtype=None, graph=None, due_to_operation=False, is_leaf=False):
        if isinstance(data, CustomTensor):
            return data  # Don't rewrap
        return super().__new__(cls)

    def __init__(self, data, *, _custom_requires_grad=False, device=None, dtype=None, graph=None, due_to_operation=False, is_leaf=False):
        if isinstance(data, CustomTensor):
            return

        self.tensor = data if due_to_operation else torch.as_tensor(data, dtype=dtype, device=device)
        self.tensor.requires_grad_(False)
        self._custom_requires_grad = _custom_requires_grad
        self._node_id = None
        self._backward = lambda: None
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

    def _zero_grad(self):
        """Sets the gradient of the underlying tensor to zero."""
        if self.tensor.grad is not None:
            self.tensor.grad.zero_()

    # --- Broadcasting Helper ---
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



    # --- Basic Operators (from your original code, now compatible with new features) ---
    def __add__(self, other):
        # ... [Your original implementation]
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
        # ... [Your original implementation]
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

    def backward(self, weightage_tensor=1):
        if not self._custom_requires_grad:
            raise RuntimeError("Output tensor does not require grad.")
        if self.graph is None:
            raise RuntimeError("Output tensor is not part of a graph.")
        graph = self.graph
        
        # Initialize gradient for the output tensor
        if isinstance(weightage_tensor, numbers.Number):
            self.tensor.grad = torch.full_like(self.tensor, fill_value=weightage_tensor)
        elif isinstance(weightage_tensor, torch.Tensor):
            self.tensor.grad = weightage_tensor.clone()

        nodes_to_process = graph.reverse_toposort_from_tensor(self._node_id)
        
        for tensor_node in nodes_to_process:
            try:
                # The node is a weakref.proxy, check if it's still alive
                if tensor_node.__class__ is weakref.ProxyType:
                    tensor_node._backward()
            except ReferenceError:
                # The tensor object was garbage collected, skip.
                continue
    
    # --- New Unary Operations ---
    
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
        
    # --- Activation Functions ---

    def relu(self):
        """Applies the Rectified Linear Unit function element-wise."""
        result_tensor = F.relu(self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        
        def _backward():
            if self_ref.tensor.grad is None: self_ref._zero_grad()
            # Derivative is 1 for positive inputs, 0 otherwise
            grad_mask = (self_ref.tensor > 0).type(self_ref.tensor.dtype)
            self_ref.tensor.grad.add_(result_ref.tensor.grad * grad_mask)

        result._backward = _backward
        return result

    def tanh(self):
        """Applies the hyperbolic tangent function element-wise."""
        result_tensor = torch.tanh(self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        
        def _backward():
            if self_ref.tensor.grad is None: self_ref._zero_grad()
            # Derivative is 1 - tanh^2(x)
            local_grad = 1 - result_tensor.pow(2)
            self_ref.tensor.grad.add_(result_ref.tensor.grad * local_grad)

        result._backward = _backward
        return result

    def leaky_relu(self, negative_slope=0.01):
        """Applies the Leaky Rectified Linear Unit function element-wise."""
        result_tensor = F.leaky_relu(self.tensor, negative_slope)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)

        def _backward():
            if self_ref.tensor.grad is None: self_ref._zero_grad()
            # Derivative is 1 for positive, negative_slope for negative
            local_grad = torch.ones_like(self_ref.tensor)
            local_grad[self_ref.tensor < 0] = negative_slope
            self_ref.tensor.grad.add_(result_ref.tensor.grad * local_grad)

        result._backward = _backward
        return result

    def elu(self, alpha=1.0):
        """Applies the Exponential Linear Unit function element-wise."""
        result_tensor = F.elu(self.tensor, alpha)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)

        def _backward():
            if self_ref.tensor.grad is None: self_ref._zero_grad()
            # Derivative is 1 for positive, and output + alpha for negative
            local_grad = torch.ones_like(self_ref.tensor)
            neg_mask = self_ref.tensor < 0
            local_grad[neg_mask] = result_tensor[neg_mask] + alpha
            self_ref.tensor.grad.add_(result_ref.tensor.grad * local_grad)

        result._backward = _backward
        return result
        
    def silu(self):
        """Applies the Sigmoid-weighted Linear Unit function element-wise."""
        result_tensor = F.silu(self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        
        def _backward():
            if self_ref.tensor.grad is None: self_ref._zero_grad()
            # Derivative of x*sigmoid(x) is sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
            sig_x = torch.sigmoid(self_ref.tensor)
            local_grad = sig_x * (1 + self_ref.tensor * (1 - sig_x))
            self_ref.tensor.grad.add_(result_ref.tensor.grad * local_grad)

        result._backward = _backward
        return result
    
    # Add swish as an alias for silu
    swish = silu

    def gelu(self):
        """Applies the Gaussian Error Linear Unit function element-wise."""
        result_tensor = F.gelu(self.tensor)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        
        def _backward():
            if self_ref.tensor.grad is None: self_ref._zero_grad()
            # Derivative of GELU: 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
            x = self_ref.tensor
            cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
            pdf = torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)
            local_grad = cdf + x * pdf
            self_ref.tensor.grad.add_(result_ref.tensor.grad * local_grad)

        result._backward = _backward
        return result

    def softmax(self, dim=-1):
        """Applies the softmax function along a given dimension."""
        result_tensor = torch.softmax(self.tensor, dim=dim)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor, due_to_operation=True)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)
        
        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)

        def _backward():
            if self_ref.tensor.grad is None: self_ref._zero_grad()
            # For softmax, the jacobian-vector product is y * (grad - sum(grad * y))
            y = result_tensor
            grad_output = result_ref.tensor.grad
            grad_input = y * (grad_output - (grad_output * y).sum(dim=dim, keepdim=True))
            self_ref.tensor.grad.add_(grad_input)
            
        result._backward = _backward
        return result
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
    def __del__(self):
        if self._node_id is not None and self._is_leaf:
            try:
                if self.graph: self.graph.delete_node(self._node_id)
            except ReferenceError: # Graph might be gone first
                pass

# ## Neural Network Modules

# Here are the new `Module` and layer classes, designed to be similar to PyTorch's `nn` module.

# ```python
# class Module:
#     """
#     Base class for all neural network modules. Your models should also subclass this class.
#     Modules can also contain other Modules, allowing to nest them in a tree structure.
#     """
#     def __init__(self):
#         self._parameters = OrderedDict()
#         self._modules = OrderedDict()
#         self._buffers = OrderedDict()
#         self.training = True #
        
#     def __setattr__(self, name, value):
#         if isinstance(value, CustomTensor):
#             if value._custom_requires_grad:
#                 self._parameters[name] = value
#         elif isinstance(value, Module):
#             self._modules[name] = value
#         # Handle buffers (non-parameter tensors like running_mean in BatchNorm)
#         elif isinstance(value, torch.Tensor):
#             self._buffers[name] = value
#         super().__setattr__(name, value)

#     def parameters(self):
#         """Returns a list of all parameters in the module and its submodules."""
#         params = list(self._parameters.values())
#         for module in self._modules.values():
#             params.extend(module.parameters())
#         return params

#     def zero_grad(self):
#         """Sets gradients of all model parameters to zero."""
#         for p in self.parameters():
#             p._zero_grad()

#     def train(self):
#         """Sets the module and all its submodules to training mode."""
#         self.training = True
#         for module in self._modules.values():
#             module.train()

#     def eval(self):
#         """Sets the module and all its submodules to evaluation mode."""
#         self.training = False
#         for module in self._modules.values():
#             module.eval()

#     def __call__(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)

#     def forward(self, *args, **kwargs):
#         raise NotImplementedError("Subclasses of Module must implement a forward method.")

# class Linear(Module):
#     """Applies a linear transformation to the incoming data: y = xA^T + b"""
#     def __init__(self, in_features, out_features, bias=True, *, graph=None):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.graph = graph
        
#         # Using Kaiming He initialization for weights
#         self.weight = CustomTensor(torch.empty(out_features, in_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
#         torch.nn.init.kaiming_uniform_(self.weight.tensor, a=math.sqrt(5))
        
#         if bias:
#             # Using uniform initialization for bias
#             self.bias = CustomTensor(torch.empty(out_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
#             fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.tensor)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             torch.nn.init.uniform_(self.bias.tensor, -bound, bound)
#         else:
#             self.bias = None

#     def forward(self, input_tensor):
#         output = input_tensor.matmul(self.weight.T)
#         if self.bias is not None:
#             output = output + self.bias
#         return output
        
#     def __repr__(self):
#         return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

# class Conv2d(Module):
#     """Applies a 2D convolution over an input signal composed of several input planes."""
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *, graph=None):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
#         self.stride = (stride, stride) if isinstance(stride, int) else stride
#         self.padding = (padding, padding) if isinstance(padding, int) else padding
#         self.graph = graph

#         # Weight and bias initialization
#         self.weight = CustomTensor(torch.empty(out_channels, in_channels, *self.kernel_size), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
#         self.bias = CustomTensor(torch.empty(out_channels), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
#         torch.nn.init.kaiming_uniform_(self.weight.tensor, a=math.sqrt(5))
#         fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.tensor)
#         bound = 1 / math.sqrt(fan_in)
#         torch.nn.init.uniform_(self.bias.tensor, -bound, bound)

#     def forward(self, input_tensor):
#         # We can use torch.nn.functional.conv2d and define a custom backward pass.
#         # This is much simpler than implementing im2col/col2im manually.
#         output_tensor = F.conv2d(input_tensor.tensor, self.weight.tensor, self.bias.tensor, self.stride, self.padding)

#         requires_grad = input_tensor._custom_requires_grad or self.weight._custom_requires_grad
#         if not requires_grad:
#             return CustomTensor(output_tensor, due_to_operation=True)
        
#         result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, due_to_operation=True, is_leaf=False)
#         self.graph.add_edge(input_tensor._node_id, result._node_id)
#         self.graph.add_edge(self.weight._node_id, result._node_id)
#         self.graph.add_edge(self.bias._node_id, result._node_id)
        
#         input_ref = weakref.proxy(input_tensor)
#         weight_ref = weakref.proxy(self.weight)
#         bias_ref = weakref.proxy(self.bias)
#         result_ref = weakref.proxy(result)

#         def _backward():
#             grad_output = result_ref.tensor.grad
            
#             # Gradient for input
#             if input_ref._custom_requires_grad:
#                 if input_ref.tensor.grad is None: input_ref._zero_grad()
#                 input_ref.tensor.grad.add_(
#                     torch.nn.grad.conv2d_input(input_ref.tensor.shape, weight_ref.tensor, grad_output, self.stride, self.padding)
#                 )

#             # Gradient for weights
#             if weight_ref._custom_requires_grad:
#                 if weight_ref.tensor.grad is None: weight_ref._zero_grad()
#                 weight_ref.tensor.grad.add_(
#                     torch.nn.grad.conv2d_weight(input_ref.tensor, weight_ref.tensor.shape, grad_output, self.stride, self.padding)
#                 )

#             # Gradient for bias
#             if bias_ref._custom_requires_grad:
#                 if bias_ref.tensor.grad is None: bias_ref._zero_grad()
#                 bias_ref.tensor.grad.add_(grad_output.sum(dim=[0, 2, 3]))

#         result._backward = _backward
#         return result

# class BatchNorm2d(Module):
#     """Applies Batch Normalization over a 4D input."""
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, *, graph=None):
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.graph = graph

#         # Learnable parameters
#         self.weight = CustomTensor(torch.ones(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)
#         self.bias = CustomTensor(torch.zeros(num_features), _custom_requires_grad=True, graph=self.graph, is_leaf=True)

#         # Buffers for running statistics
#         self.running_mean = torch.zeros(num_features)
#         self.running_var = torch.ones(num_features)

#     def forward(self, input_tensor):
#         if self.training:
#             # Calculate batch statistics
#             batch_mean = input_tensor.tensor.mean(dim=[0, 2, 3])
#             batch_var = input_tensor.tensor.var(dim=[0, 2, 3], unbiased=False)
            
#             # Update running statistics
#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
#             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
#             mean_to_use = batch_mean
#             var_to_use = batch_var
#         else:
#             # Use running statistics during evaluation
#             mean_to_use = self.running_mean
#             var_to_use = self.running_var
            
#         # Reshape for broadcasting
#         view_shape = (1, self.num_features, 1, 1)
#         mean_r = mean_to_use.view(view_shape)
#         var_r = var_to_use.view(view_shape)
#         weight_r = self.weight.tensor.view(view_shape)
#         bias_r = self.bias.tensor.view(view_shape)

#         # Normalize
#         x_hat = (input_tensor.tensor - mean_r) / torch.sqrt(var_r + self.eps)
#         output_tensor = weight_r * x_hat + bias_r

#         # --- Autograd part ---
#         requires_grad = input_tensor._custom_requires_grad or self.weight._custom_requires_grad
#         if not requires_grad:
#             return CustomTensor(output_tensor, due_to_operation=True)

#         result = CustomTensor(output_tensor, _custom_requires_grad=True, graph=self.graph, due_to_operation=True, is_leaf=False)
#         self.graph.add_edge(input_tensor._node_id, result._node_id)
#         self.graph.add_edge(self.weight._node_id, result._node_id)
#         self.graph.add_edge(self.bias._node_id, result._node_id)
        
#         input_ref = weakref.proxy(input_tensor)
#         weight_ref = weakref.proxy(self.weight)
#         bias_ref = weakref.proxy(self.bias)
#         result_ref = weakref.proxy(result)

#         def _backward():
#             # This backward pass is only correct for the training case.
#             # A full implementation requires caching intermediate values.
#             # Using PyTorch's functional backward for simplicity and correctness.
#             grad_output = result_ref.tensor.grad
            
#             # Using autograd on the forward pass logic to get the gradients
#             # This is a simplification; a manual implementation is very complex.
#             temp_input = input_ref.tensor.clone().requires_grad_()
#             temp_weight = weight_ref.tensor.clone().requires_grad_()
#             temp_bias = bias_ref.tensor.clone().requires_grad_()

#             out = F.batch_norm(temp_input, self.running_mean, self.running_var, temp_weight, temp_bias, self.training, self.momentum, self.eps)
#             out.backward(grad_output)
            
#             if input_ref._custom_requires_grad:
#                 if input_ref.tensor.grad is None: input_ref._zero_grad()
#                 input_ref.tensor.grad.add_(temp_input.grad)

#             if weight_ref._custom_requires_grad:
#                 if weight_ref.tensor.grad is None: weight_ref._zero_grad()
#                 weight_ref.tensor.grad.add_(temp_weight.grad)

#             if bias_ref._custom_requires_grad:
#                 if bias_ref.tensor.grad is None: bias_ref._zero_grad()
#                 bias_ref.tensor.grad.add_(temp_bias.grad)

#         result._backward = _backward
#         return result

# # Note on Conv1d, Conv3d, and img_to_col:
# # The `Conv2d` implementation above uses `torch.nn.grad.conv2d_*` for the backward pass,
# # which is the most robust approach. A similar pattern can be followed for `Conv1d` and `Conv3d`
# # using their respective `torch.nn.grad` functions. Implementing `img_to_col` (or `unfold`) manually
# # and its counterpart `col2im` (`fold`) adds significant complexity but is the underlying
# # mechanism for these gradient calculations. Skip connections are handled automatically
# # by the existing `__add__` operator.

# def initialize_weights(module):
#     """
#     Initializes weights for Linear and Conv2d layers in a module.
#     - Kaiming (He) for Conv/Linear weights.
#     - Zeros for biases.
#     - Ones/Zeros for BatchNorm weights/biases.
#     """
#     for m in m.modules():
#         if isinstance(m, Conv2d):
#             torch.nn.init.kaiming_normal_(m.weight.tensor, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 torch.nn.init.constant_(m.bias.tensor, 0)
#         elif isinstance(m, BatchNorm2d):
#             torch.nn.init.constant_(m.weight.tensor, 1)
#             torch.nn.init.constant_(m.bias.tensor, 0)
#         elif isinstance(m, Linear):
#             torch.nn.init.kaiming_normal_(m.weight.tensor, nonlinearity='relu')
#             if m.bias is not None:
#                 torch.nn.init.constant_(m.bias.tensor, 0)