from typing import Union, Optional, Tuple, Any
import torch
import rustworkx as rx
from . import RUN_ON_CPU, RUN_ON_GPU, RUN_ON_TPU, device
# from weakref import WeakValueDictionary
import weakref
class Tensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
882e32
"""
for an autograd if i want to do something similar

1. Creating a graph context manager class

2. for first fowrward pass create the tensors in memory

3. after the graph is created and temporary tensors are in place

4. at each backward pass instead of deleting the temp tensors created due to operations modify their attributes in memory instead

5. store the result of toposort and keep using that for every backward pass

6. after calling gradfn for a node reset it's gradiants to 0 within the same loop

7. profit
"""

class CustomAutogradGraph:
    def __init__(self):
        self.graph = rx.PyDiGraph() # Initialize a new directed graph
        #self.graph = None#rx.PyDiGraph()  # Using Rustworkx for graph representation
        self._is_active = False # Track if the graph is active
        self.nodes = None  # No of Nodes
        self.edges = None  # No of Edges
    
    def __enter__(self):
        """
        Start a new autograd graph context.
        This method is called when entering the 'with' block.
        """
        global _autograd_graph_instance
        if _autograd_graph_instance is not None and _autograd_graph_instance._is_active:
            raise RuntimeError("An AutogradGraph is already active. Nested contexts are not supported directly.")
        _autograd_graph_instance = self
        self._is_active = True
        return self
    def __exit__(self, exc_type,exc_value,exc_tb):
        """
        End the autograd graph context.
        This method is called when exiting the 'with' block.
        """
        global _autograd_graph_instance
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to exit.")
        self._is_active = False
        _autograd_graph_instance = None
    
    def add_tensor(self,tensor):
        """
        Add a tensor to the autograd graph.
        This method should be called when a new tensor is created.
        """
        if not isinstance(tensor, CustomTensor):
            raise TypeError("Only CustomTensor instances can be added to the graph.")
        if self.current_node_id is None:
            self.nodes= 0
        self.nodes += 1
        if not tensor._custom_requires_grad:
            raise ValueError("Tensor must have requires_grad set to True to be added to the graph.")
        
        tensor_index=self.graph.add_node(tensor)#weakref.ref(tensor)) #addding tensors as weak_references to prevent memory leaks
        return tensor_index 
    def add_edge(self, node_from, node_to,weight=None):
        """
        Add an edge between two nodes in the graph.
        This method should be called when a tensor operation is performed.
        """
        if not self._is_active:
            raise RuntimeError("Cannot add edges outside an active AutogradGraph context.")
        if self.edges is None:
            self.edges = 0
        self.edges += 1
        if not isinstance(node_from, int) or not isinstance(node_to, int):
            raise TypeError("Node indices must be integers.")
        self.graph.add_edge(node_from, node_to,weight)
    @classmethod
    def check_cycle(cls):
        """
        Check if the current graph has a cycle.
        This method can be used to validate the graph structure.
        """
        if not cls._is_active:
            raise RuntimeError("No active AutogradGraph to check for cycles.")
        return rx.is_directed_acyclic_graph(cls.graph)
    
    def reverse_toposort(self):
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to perform topological sort.")
        if self.check_cycle():
            raise RuntimeError("Cannot perform topological sort on a graph with cycles.")
        node_indexes=rx.toposort_directed(self.graph)
        node_indexes.reverse() # must reverse the order to get the correct order for backpropagation
        # Convert node indexes to tensor references
        tensor_references = [self.graph[node_index] for node_index in node_indexes]
        return  tensor_references
    def clear_graph(self):
        """
        Clear the current graph.
        This method can be used to reset the graph for a new computation.
        """
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to clear.")
        self.graph.clear()
        self.nodes = None
        self.edges = None
        self._is_active = False
    
    def __repr__(self):
        """
        Custom representation of the AutogradGraph.
        """
        return f"CustomAutogradGraph(nodes={self.nodes}, edges={self.edges}, active={self._is_active})"
    def delete_node(self, node_index):
        """
        Delete a node from the graph.
        This method can be used to remove a tensor from the graph.
        """
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to delete nodes from.")
        if not isinstance(node_index, int):
            raise TypeError("Node index must be an integer.")
        if node_index not in self.graph:
            raise ValueError(f"Node index {node_index} does not exist in the graph.")
        self.graph.remove_node(node_index)
        self.nodes -= 1
    def delete_edge(self, node_from, node_to):
        """
        Delete an edge from the graph.
        This method can be used to remove an edge from the graph.
        """
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to delete edges from.")
        if not isinstance(node_from, int) or not isinstance(node_to, int):
            raise TypeError("Node indices must be integers.")
        if (node_from, node_to) not in self.graph.edges():
            raise ValueError(f"Edge ({node_from}, {node_to}) does not exist in the graph.")
        self.graph.remove_edge(node_from, node_to)
        self.edges -= 1

graph = CustomAutogradGraph() # Global instance of the autograd graph
graph._is_active =  True
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
    def __new__(cls, data=None, requires_grad=False, operation=None, _tensor=None,_is_leaf=False,dtype=None,device=device):
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
        dtype = dtype if dtype is not None else torch.float32 # Ensure dtype is set, default to float32
        if _tensor is not None:
            # If an existing torch.Tensor is provided, wrap it directly
            # This is crucial for returning CustomTensor instances from operations
            # that produce new torch.Tensors.
            _tensor = _tensor.detach()
            # _tensor.to(device)
            # _tensor.requires_grad=False
            # _tensor.dtype = dtype if dtype is not None else torch.float32 # Ensure dtype is set, default to float32
            instance = torch.Tensor._make_subclass(cls, _tensor)#, requires_grad)
            # instance.requires_grad=requires_grad
            instance.requires_grad_(False) # disable pytorch's autograd from recording anything for this tensor
            # instance.grad = None  # Initialize gradient
            # instance.operation = operation # Store the operation that created this tensor
        else:
            # Otherwise, create a new torch.Tensor from the provided data
            # and then wrap it.
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data, dtype=dtype,device=device) # Ensure float type

            # Use torch.Tensor._make_subclass to create an instance of CustomTensor
            # that wraps the underlying torch.Tensor data.
            # We explicitly set requires_grad for the underlying torch.Tensor to False
            # because we are building a custom autograd system.
            instance = torch.Tensor._make_subclass(cls, data)
            instance.requires_grad_(False) # disable pytorch's autograd from recording anything for this tensor

        instance._custom_requires_grad = requires_grad
        #instance.dtype = dtype if dtype is not None else torch.float32 # Ensure dtype is set, default to float32
        instance.node_id = graph.add_tensor(instance) if requires_grad else None # Add to the autograd graph if requires_grad is True
        instance._is_leaf = _is_leaf

        # instance.grad = None  # Initialize gradient
        instance.operation = operation # Store the operation that created this tensor
        return instance
    def __matmul__(self,other):
        if not isinstance(other, CustomTensor):
            raise TypeError("CustomTensor can only be multiplied by another CustomTensor or a compatible type.")
        output = super().__matmul__(other)
        if other._custom_requires_grad or self._custom_requires_grad:
            def _matmulbackward():
                if self._custom_requires_grad:
                    if self.grad is None:
                        self.grad = CustomTensor(_tensor=torch.zeros(self.shape),requires_grad=False)
                    self.grad += output.grad @ other.T
                if other._custom_requires_grad:
                    if other.grad is None:
                        other.grad = CustomTensor(_tensor=torch.zeros(other.shape),requires_grad=False)
                    other.grad += output.grad.T @ self

            output._custom_backward = _matmulbackward
        return output
    def __repr__(self):
        # Custom representation for better debugging
        return f"CustomTensor({super().__repr__()}, requires_grad={self._custom_requires_grad}, grad={self.grad})"
    
    def zero_grad(self):
        """Zeros out the gradient of this tensor."""
        if self.grad is not None:
            self.grad = CustomTensor(_tensor=torch.zeros(self.shape)) #torch.zeros_like(self.grad,requires_grad=False) # the gradient attribute must be a pytorch tensor



    def __add__(self,other):
      if not isinstance(other, CustomTensor):
          raise TypeError("CustomTensor can only be added to another CustomTensor or a compatible type.")
      output = super().__add__(other)
      if other._custom_requires_grad or self._custom_requires_grad:
        def _addbackward():
            if self._custom_requires_grad:
                if self.grad is None:
                    self.grad = CustomTensor(_tensor=torch.zeros(self.shape),requires_grad=False)
                    #torch.zeros_like(self,requires_grad=False,)#self.grad.zero_()#torch.zeros(self.shape,requires_grad=False) # the gradient attribute msut be a pytorch tensor 
                self.grad+=output.grad
            if other._custom_requires_grad:
                if other.grad is None:
                    other.grad = CustomTensor(_tensor=torch.zeros(other.shape),requires_grad=False)#torch.zeros_like(other)#other.grad.zero_()#torch.zeros(self.shape,requires_grad=False) # the gradient attribute msut be a pytorch tensor 
                other.grad+=output.grad
    
        output._custom_backward=_addbackward
      return output
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if not isinstance(other, CustomTensor):
            raise TypeError("CustomTensor can only be multiplied by another CustomTensor or a compatible type.")
        output = super().__mul__(other)
        if other._custom_requires_grad or self._custom_requires_grad:
            def _mulbackward():
                if self._custom_requires_grad:
                    if self.grad is None:
                        self.grad = CustomTensor(_tensor=torch.zeros(self.shape),requires_grad=False)
                    self.grad += output.grad * other
                if other._custom_requires_grad:
                    if other.grad is None:
                        other.grad = CustomTensor(_tensor=torch.zeros(other.shape),requires_grad=False)
                    other.grad += output.grad * self

            output._custom_backward = _mulbackward
        return output
    def __rmul__(self, other):
        return self * other

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


# # Define a simple Operation class for your custom autograd graph
# # This would track the function performed and its inputs for backpropagation.
# class Operation:
#     def __init__(self, inputs=()):
#         self.inputs = inputs # Tensors involved in this operation

#     def forward(self, *args):
#         # This method would perform the actual computation.
#         # For a custom autograd, you typically don't implement 'forward' here
#         # in the Operation class, but rather within the Tensor's dunder methods
#         # that create this operation.
#         raise NotImplementedError

#     def backward(self, grad_output):
#         # This method computes and propagates gradients to inputs.
#         raise NotImplementedError

# # Example Operation: Addition
# class Add(Operation):
#     def __init__(self, input1, input2):
#         super().__init__((input1, input2))

#     def backward(self, grad_output):
#         # For addition, the gradient is simply propagated to both inputs.
#         # Make sure to handle cases where inputs might be None or already have gradients.
#         if self.inputs[0].grad is None:
#             self.inputs[0].grad = grad_output
#         else:
#             self.inputs[0].grad += grad_output

#         if self.inputs[1].grad is None:
#             self.inputs[1].grad = grad_output
#         else:
#             self.inputs[1].grad += grad_output


# # Define your custom Tensor class




class CustomAutogradGraph:
    def __init__(self):
        self.graph = rx.PyDiGraph() # Initialize a new directed graph
        #self.graph = None#rx.PyDiGraph()  # Using Rustworkx for graph representation
        self._is_active = False # Track if the graph is active
        self.nodes = None  # No of Nodes
        self.edges = None  # No of Edges
    
    def __enter__(self):
        """
        Start a new autograd graph context.
        This method is called when entering the 'with' block.
        """
        global _autograd_graph_instance
        if _autograd_graph_instance is not None and _autograd_graph_instance._is_active:
            raise RuntimeError("An AutogradGraph is already active. Nested contexts are not supported directly.")
        _autograd_graph_instance = self
        self._is_active = True
        return self
    def __exit__(self, exc_type,exc_value,exc_tb):
        """
        End the autograd graph context.
        This method is called when exiting the 'with' block.
        """
        global _autograd_graph_instance
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to exit.")
        self._is_active = False
        _autograd_graph_instance = None
    
    def add_tensor(self,tensor):
        """
        Add a tensor to the autograd graph.
        This method should be called when a new tensor is created.
        """
        if not isinstance(tensor, CustomTensor):
            raise TypeError("Only CustomTensor instances can be added to the graph.")
        if self.current_node_id is None:
            self.nodes= 0
        self.nodes += 1
        if not tensor._custom_requires_grad:
            raise ValueError("Tensor must have requires_grad set to True to be added to the graph.")
        
        tensor_index=self.graph.add_node(tensor)#weakref.ref(tensor)) #addding tensors as weak_references to prevent memory leaks
        return tensor_index 
    def add_edge(self, node_from, node_to,weight=None):
        """
        Add an edge between two nodes in the graph.
        This method should be called when a tensor operation is performed.
        """
        if not self._is_active:
            raise RuntimeError("Cannot add edges outside an active AutogradGraph context.")
        if self.edges is None:
            self.edges = 0
        self.edges += 1
        if not isinstance(node_from, int) or not isinstance(node_to, int):
            raise TypeError("Node indices must be integers.")
        self.graph.add_edge(node_from, node_to,weight)
    @classmethod
    def check_cycle(cls):
        """
        Check if the current graph has a cycle.
        This method can be used to validate the graph structure.
        """
        if not cls._is_active:
            raise RuntimeError("No active AutogradGraph to check for cycles.")
        return rx.is_directed_acyclic_graph(cls.graph)
    
    def reverse_toposort(self):
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to perform topological sort.")
        if self.check_cycle():
            raise RuntimeError("Cannot perform topological sort on a graph with cycles.")
        node_indexes=rx.toposort_directed(self.graph)
        node_indexes.reverse() # must reverse the order to get the correct order for backpropagation
        # Convert node indexes to tensor references
        tensor_references = [self.graph[node_index] for node_index in node_indexes]
        return  tensor_references
    def clear_graph(self):
        """
        Clear the current graph.
        This method can be used to reset the graph for a new computation.
        """
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to clear.")
        self.graph.clear()
        self.nodes = None
        self.edges = None
        self._is_active = False
    
    def __repr__(self):
        """
        Custom representation of the AutogradGraph.
        """
        return f"CustomAutogradGraph(nodes={self.nodes}, edges={self.edges}, active={self._is_active})"
    def delete_node(self, node_index):
        """
        Delete a node from the graph.
        This method can be used to remove a tensor from the graph.
        """
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to delete nodes from.")
        if not isinstance(node_index, int):
            raise TypeError("Node index must be an integer.")
        if node_index not in self.graph:
            raise ValueError(f"Node index {node_index} does not exist in the graph.")
        self.graph.remove_node(node_index)
        self.nodes -= 1
    def delete_edge(self, node_from, node_to):
        """
        Delete an edge from the graph.
        This method can be used to remove an edge from the graph.
        """
        if not self._is_active:
            raise RuntimeError("No active AutogradGraph to delete edges from.")
        if not isinstance(node_from, int) or not isinstance(node_to, int):
            raise TypeError("Node indices must be integers.")
        if (node_from, node_to) not in self.graph.edges():
            raise ValueError(f"Edge ({node_from}, {node_to}) does not exist in the graph.")
        self.graph.remove_edge(node_from, node_to)
        self.edges -= 1