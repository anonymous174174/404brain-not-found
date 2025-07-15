import torch
import weakref
import numbers
import rustworkx as rx
import pytest

class AutogradGraph:
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

    def reverse_toposort(self):
        return [self.graph[n] for n in reversed(rx.topological_sort(self.graph))]

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
        if not self.graph.has_node(node_index):
            raise ValueError("Node does not exist.")
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

class CustomTensor:
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
        self.tensor.grad = torch.zeros_like(self.tensor)

    # --- Broadcasting Helper Function ---
    
    def _reduce_grad_for_broadcast(self,grad, target_shape):
      if grad.shape == target_shape:
          return grad
      padded_target_shape = (1,) * (grad.ndim - len(target_shape)) + target_shape
      sum_dims = []
      sum_dims = [i for i in range(grad.ndim) if padded_target_shape[i] == 1 and grad.shape[i] > 1] 
      if sum_dims:
          grad = grad.sum(dim=sum_dims, keepdim=True)    
      if grad.shape != target_shape:
        grad = grad.reshape(target_shape)    
      return grad
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
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            # Scalar addition doesn't change shape for the tensor, so no reduction needed
            self_ref.tensor.grad.add_(result_ref.tensor.grad)

        result._backward = _backward
        return result

    def _add_tensor(self, other):
        result_tensor = torch.add(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad

        if not requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = None
        if self._custom_requires_grad:
            graph = self.graph
        elif other._custom_requires_grad:
            graph = other.graph
        else:
            pass 
            
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
                if self_ref.tensor.grad is None:
                    self_ref._zero_grad()
                # Apply reduction if 'self' was broadcasted
                grad_for_self = self_ref._reduce_grad_for_broadcast(result_ref.tensor.grad, self_ref.tensor.shape)
                self_ref.tensor.grad.add_(grad_for_self)

            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                # Apply reduction if 'other' was broadcasted
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
            # Scalar multiplication doesn't change shape for the tensor, no reduction needed
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
                if self_ref.tensor.grad is None:
                    self_ref._zero_grad()
                grad_for_self = self_ref._reduce_grad_for_broadcast(result_ref.tensor.grad * other_ref.tensor, self_ref.tensor.shape)
                self_ref.tensor.grad.add_(grad_for_self)
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
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
                if self_ref.tensor.grad is None:
                    self_ref._zero_grad()
                # Matmul broadcasting for batch dimensions: no sum needed if shapes align after matmul
                grad_for_self = torch.matmul(result_ref.tensor.grad, other_ref.tensor.transpose(-2, -1))
                # If there were batch dimensions that were broadcasted in self, sum over them
                # This check can be more complex for general batch broadcasting in matmul
                if grad_for_self.shape != self_ref.tensor.shape:
                    grad_for_self = self_ref._reduce_grad_for_broadcast(grad_for_self, self_ref.tensor.shape)
                self_ref.tensor.grad.add_(grad_for_self)

            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                grad_for_other = torch.matmul(self_ref.tensor.transpose(-2, -1), result_ref.tensor.grad)
                # If there were batch dimensions that were broadcasted in other, sum over them
                if grad_for_other.shape != other_ref.tensor.shape:
                    grad_for_other = other_ref._reduce_grad_for_broadcast(grad_for_other, other_ref.tensor.shape)
                other_ref.tensor.grad.add_(grad_for_other)
        result._backward = _backward
        return result

    def apply_mask(self, mask):
        result_tensor = self.tensor * mask.tensor # This is element-wise multiplication
        requires_grad = self._custom_requires_grad or mask._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor,due_to_operation=True)

        graph = self.graph if self._custom_requires_grad else mask.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)

        self_ref = weakref.proxy(self)
        mask_ref = weakref.proxy(mask)
        result_ref = weakref.proxy(result)

        if self._custom_requires_grad:
            graph.add_edge(self._node_id, result._node_id)
        if mask._custom_requires_grad:
            graph.add_edge(mask._node_id, result._node_id)

        def _backward():
            if self_ref._custom_requires_grad:
                if self_ref.tensor.grad is None:
                    self_ref._zero_grad()
                # Apply reduction if 'self' was broadcasted
                grad_for_self = self_ref._reduce_grad_for_broadcast(result_ref.tensor.grad * mask_ref.tensor, self_ref.tensor.shape)
                self_ref.tensor.grad.add_(grad_for_self)
            if mask_ref._custom_requires_grad:
                if mask_ref.tensor.grad is None:
                    mask_ref._zero_grad()
                # Apply reduction if 'mask' was broadcasted
                grad_for_mask = mask_ref._reduce_grad_for_broadcast(result_ref.tensor.grad * self_ref.tensor, mask_ref.tensor.shape)
                mask_ref.tensor.grad.add_(grad_for_mask)
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

    def backward(self,weightage_tensor=1):
        if not self._custom_requires_grad:
            raise RuntimeError("Output tensor does not require grad.")
        if self.graph is None:
            raise RuntimeError("Output tensor is not part of a graph.")
        graph = self.graph

        # Initialize gradient for the output tensor
        if isinstance(weightage_tensor,numbers.Number):
            self.tensor.grad = torch.full_like(self.tensor, fill_value=weightage_tensor)
        elif isinstance(weightage_tensor,torch.Tensor):
            self.tensor.grad = weightage_tensor.clone() # we don't want to modify the original tensor data

        # Perform backward pass using topological sort

        nodes_to_process = graph.reverse_toposort_from_tensor(self._node_id)

        for tensor_node in nodes_to_process:
            # Check if the weak proxy is still valid (tensor is alive)
            if tensor_node.__class__ is weakref.ProxyType and tensor_node.__repr__() is None:
                continue # Skip if the weak reference is dead

            if tensor_node.tensor.grad is None and tensor_node is not self.tensor:
                pass 

            tensor_node._backward()

    def to_device(self, device):
        self.tensor = self.tensor.to(device)

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def shape(self):
        return self.tensor.shape


    def __del__(self):
      if self._node_id is not None and self._is_leaf: 
        try:
              self.graph.delete_node(self._node_id)
        except ReferenceError:
              pass
      print(f"Garbage Collector has decided that reference counts for {self._node_id} and id {id(self)} are zero so Goodbye!!")