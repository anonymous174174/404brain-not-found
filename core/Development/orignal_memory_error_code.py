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

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            return self._add_scalar(other)#, op=torch.add)#Operations.add_tensor_and_scalar)
        elif isinstance(other, CustomTensor):
            return self._add_tensor(other)#, op=torch.add)#Operations.add_tensor_and_tensor)
        return NotImplemented

    def _add_scalar(self, scalar):
        result_tensor = torch.add(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            # print(f"Backward for scalar add: result_grad={result.tensor.grad}, self_grad_before={self_ref.tensor.grad}") # Debugging
            self_ref.tensor.grad.add_(result_ref.tensor.grad)
            # print(f"Backward for scalar add: self_grad_after={self_ref.tensor.grad}") # Debugging

        result._backward = _backward
        return result

    def _add_tensor(self, other):
        result_tensor = torch.add(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad

        if not requires_grad:
            return CustomTensor(result_tensor)

        # Graph selection logic - assuming operations happen within a single graph context
        graph = None
        if self._custom_requires_grad:
            graph = self.graph
        elif other._custom_requires_grad:
            graph = other.graph
        else:
            # This case should ideally not be reached if requires_grad is True
            # and at least one operand has requires_grad
            pass # Or raise an error if graph is truly missing

        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)

        self_ref = weakref.proxy(self)
        other_ref = weakref.proxy(other)

        if self._custom_requires_grad:
            graph.add_edge(self._node_id, result._node_id)
        if other._custom_requires_grad:
            graph.add_edge(other._node_id, result._node_id)
        result_ref = weakref.proxy(result)
        def _backward():
            # print(f"Backward for tensor add: result_grad={result.tensor.grad}") # Debugging
            if self_ref._custom_requires_grad:
                if self_ref.tensor.grad is None:
                    self_ref._zero_grad()
                self_ref.tensor.grad.add_(result_ref.tensor.grad)
                # print(f"  self_grad_after={self_ref.tensor.grad}") # Debugging
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                other_ref.tensor.grad.add_(result_ref.tensor.grad)
                # print(f"  other_grad_after={other_ref.tensor.grad}") # Debugging

        result._backward = _backward
        return result
    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return self._mul_scalar(other)
        elif isinstance(other, CustomTensor):
            return self._mul_tensor(other)
        return NotImplemented

    def _mul_scalar(self, scalar):
        result_tensor = torch.mul(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor)

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
            return CustomTensor(result_tensor)

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
                self_ref.tensor.grad.add_(result_ref.tensor.grad * other_ref.tensor)
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                other_ref.tensor.grad.add_(result_ref.tensor.grad * self_ref.tensor)
        result._backward = _backward
        return result


    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            return self._sub_scalar(other)
        elif isinstance(other, CustomTensor):
            return self._sub_tensor(other)
        return NotImplemented

    def _sub_scalar(self, scalar):
        result_tensor = torch.sub(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(result_ref.tensor.grad)
        result._backward = _backward
        return result

    def _sub_tensor(self, other):
        result_tensor = torch.sub(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor)

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
                self_ref.tensor.grad.add_(result_ref.tensor.grad)
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                other_ref.tensor.grad.sub_(result_ref.tensor.grad)
        result._backward = _backward
        return result

    def __truediv__(self, scalar):
        return self._div_scalar(scalar)

    def _div_scalar(self, scalar):
        result_tensor = torch.div(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor)

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


    def pow(self, scalar):
        result_tensor = torch.pow(self.tensor, scalar)
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor)

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

    def exp(self):
        out = torch.exp(self.tensor)
        return self._unary_op(out, lambda grad, out_tensor: grad * out_tensor)

    def log(self):
        out = torch.log(self.tensor)
        return self._unary_op(out, lambda grad, input_tensor: grad / input_tensor)

    def sin(self):
        out = torch.sin(self.tensor)
        return self._unary_op(out, lambda grad, input_tensor: grad * torch.cos(input_tensor))

    def cos(self):
        out = torch.cos(self.tensor)
        return self._unary_op(out, lambda grad, input_tensor: -grad * torch.sin(input_tensor))

    def sqrt(self):
        out = torch.sqrt(self.tensor)
        return self._unary_op(out, lambda grad, out_tensor: grad * 0.5 / out_tensor)

    def _unary_op(self, result_tensor, backward_fn):
        if not self._custom_requires_grad:
            return CustomTensor(result_tensor)

        graph = self.graph
        result = CustomTensor(result_tensor, _custom_requires_grad=True, graph=graph, due_to_operation=True, is_leaf=False)
        graph.add_edge(self._node_id, result._node_id)

        self_ref = weakref.proxy(self)
        result_ref = weakref.proxy(result)
        def _backward():
            if self_ref.tensor.grad is None:
                self_ref._zero_grad()
            self_ref.tensor.grad.add_(backward_fn(result_ref.tensor.grad, self_ref.tensor))
        result._backward = _backward
        return result

    def matmul(self, other):
        result_tensor = torch.matmul(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor)

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
                self_ref.tensor.grad.add_(torch.matmul(result_ref.tensor.grad, other_ref.tensor.t()))
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                other_ref.tensor.grad.add_(torch.matmul(self_ref.tensor.t(), result_ref.tensor.grad))
        result._backward = _backward
        return result

    def apply_mask(self, mask):
        result_tensor = self.tensor * mask.tensor
        requires_grad = self._custom_requires_grad or mask._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor)

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
                self_ref.tensor.grad.add_(result_ref.tensor.grad * mask_ref.tensor)
            if mask_ref._custom_requires_grad:
                if mask_ref.tensor.grad is None:
                    mask_ref._zero_grad()
                mask_ref.tensor.grad.add_(result_ref.tensor.grad * self_ref.tensor)
        result._backward = _backward
        return result

    def dot(self, other):
        result_tensor = torch.dot(self.tensor, other.tensor)
        requires_grad = self._custom_requires_grad or other._custom_requires_grad
        if not requires_grad:
            return CustomTensor(result_tensor)

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
                self_ref.tensor.grad.add_(result_ref.tensor.grad * other_ref.tensor)
            if other_ref._custom_requires_grad:
                if other_ref.tensor.grad is None:
                    other_ref._zero_grad()
                other_ref.tensor.grad.add_(result_ref.tensor.grad * self_ref.tensor)
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

        # Create a strong reference to intermediate tensors needed for backward pass
        # This simulates how a real autograd engine would keep track of them
        # The graph context's intermediate_tensors dict already serves this purpose.

        for tensor_node in nodes_to_process:
            # Check if the weak proxy is still valid (tensor is alive)
            if tensor_node.__class__ is weakref.ProxyType and tensor_node.__repr__() is None:
                # print(f"Skipping dead proxy: {tensor_node}") # Debugging
                continue # Skip if the weak reference is dead

            if tensor_node.tensor.grad is None and tensor_node is not self.tensor:
                # This can happen if a tensor is part of the graph but its grad hasn't been set yet
                # and it's not the root of the backward call. This typically means it's a leaf
                # that wasn't used to compute the output or an intermediate that accumulated no grad.
                # For simplicity in this test, we assume grads propagate.
                # print(f"Warning: Tensor node {tensor_node._node_id} has no grad before _backward call.")
                pass # A no-op for now. In a real system, you might want to handle this.

            # Ensure that non-leaf tensors are still alive when their _backward is called
            # The `intermediate_tensors` in `AutogradGraph` should keep them alive.
            tensor_node._backward()

        # Clean up intermediate tensors references after backward pass
        # This would typically be handled by the graph context's exit, but
        # if `_auto_cleanup` is False, you might need manual cleanup.
        # Here, for testing GC, we'll let the context manager handle it.



    def __del__(self):
      if self._node_id is not None and self._is_leaf and self.graph: #must remove leaf tensor from graph before it is deleted from memory
        self.graph.delete_node(self._node_id)
        # try:
        #       # Check if graph is still alive before trying to delete
        #       self.graph.delete_node(self._node_id)
        # except ReferenceError:
        #       # Graph context has already been cleaned up, so do nothing.
        #       pass
      print(f"Garbage Collector has decided that reference counts for {self._node_id} are zero so Goodbye!!")

