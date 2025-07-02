import rustworkx as rx
from tensor import CustomTensor
class AutogradGraph:
    def __init__(self):
        self.graph = rx.PyDiGraph()
        self.tensor_to_node_idx = {}  # Map CustomTensor object ID to node index
        self.op_to_node_idx = {}      # Map Operation object ID to node index
        self._is_active = False # To check if the graph is being built dynamically

    def __enter__(self):
        global _autograd_graph_instance
        if _autograd_graph_instance is not None and _autograd_graph_instance._is_active:
            raise RuntimeError("An AutogradGraph is already active. Nested contexts are not supported directly.")
        _autograd_graph_instance = self
        self._is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _autograd_graph_instance
        self._is_active = False
        _autograd_graph_instance = None
        # You might choose to clear the graph here automatically,
        # but for flexibility, we'll leave it to a manual call to clear_graph().
        # self.clear_graph()

    def add_tensor(self, tensor):
        """
        Adds a CustomTensor as a node to the graph if it doesn't already exist.
        Returns the node index.
        """
        tensor_id = id(tensor)
        if tensor_id not in self.tensor_to_node_idx:
            node_idx = self.graph.add_node(tensor)
            self.tensor_to_node_idx[tensor_id] = node_idx
        return self.tensor_to_node_idx[tensor_id]

    def add_operation(self, op):
        """
        Adds an Operation as a node to the graph if it doesn't already exist.
        Returns the node index.
        """
        op_id = id(op)
        if op_id not in self.op_to_node_idx:
            node_idx = self.graph.add_node(op)
            self.op_to_node_idx[op_id] = node_idx
        return self.op_to_node_idx[op_id]

    def add_edge(self, u_node_idx, v_node_idx, weight=None):
        """
        Adds a directed edge between two nodes.
        """
        self.graph.add_edge(u_node_idx, v_node_idx, weight)

    def build_dynamic_graph(self, output_tensor):
        """
        Recursively builds the computational graph starting from an output tensor
        by tracing back its operation and input tensors.
        This is typically called implicitly during operation execution
        when the graph is active.
        """
        if not self._is_active:
            # If not in active context, assume we're adding to a static graph
            # or this is a standalone tensor not part of an operation being tracked
            return

        output_node_idx = self.add_tensor(output_tensor)

        op = output_tensor.operation
        if op is not None:
            op_node_idx = self.add_operation(op)
            self.add_edge(op_node_idx, output_node_idx, "produces")

            for input_tensor in op.inputs:
                input_node_idx = self.add_tensor(input_tensor)
                self.add_edge(input_node_idx, op_node_idx, "input_to")
                # Recursively build for inputs that require gradients
                if input_tensor.requires_grad or input_tensor.operation is not None:
                    self.build_dynamic_graph(input_tensor)

    def topological_sort(self, start_nodes):
        """
        Performs a topological sort of the graph.
        If start_nodes is None, sorts the entire graph.
        Otherwise, sorts reachable nodes from start_nodes.
        Returns a list of nodes in topological order.
        """
        if start_nodes is None:
            return rx.topological_sort(self.graph)
        else:
            # If specific start_nodes are given, we need to ensure they are in the graph
            # and then sort the subgraph reachable from them.
            # rustworkx.topological_sort_by_dfs can be used if we need to start from specific nodes.
            # For a general topological sort of the full graph that considers all dependencies,
            # the standard topological_sort on the whole graph is appropriate for autograd.
            # For backward pass, we often need a reverse topological sort from the output.
            # Let's return the full graph's topological sort and let the caller filter.
            # Or, for backward pass, we can just use the operation chain.
            print("Note: For autograd backward, a reverse topological sort from the output is typically needed.")
            return rx.topological_sort(self.graph)


    # def backward(self, output_tensor, grad_output= None):
    #     """
    #     Performs a backward pass through the graph to compute gradients.
    #     Starts from the output_tensor and propagates gradients.
    #     """
    #     if not output_tensor.requires_grad:
    #         print("Output tensor does not require gradients. Skipping backward pass.")
    #         return

    #     if grad_output is None:
    #         # Default gradient for scalar output
    #         grad_output = CustomTensor(data=torch.ones_like(output_tensor._tensor))

    #     output_tensor.grad = grad_output

    #     # Perform a reverse topological sort from the output node
    #     # We need to find the path of operations that led to this output.
    #     # A simpler way for autograd is to traverse the 'operation' links.
    #     # This avoids needing to define complex start_nodes for rustworkx
    #     # and directly follows the computational chain.

    #     # Collect all operations in reverse topological order by tracing back
    #     operations_to_process = []
    #     visited_ops = set()

    #     def collect_ops(tensor: CustomTensor):
    #         if tensor.operation is not None and tensor.operation not in visited_ops:
    #             visited_ops.add(tensor.operation)
    #             # Recursively collect operations for inputs
    #             for inp in tensor.operation.inputs:
    #                 collect_ops(inp)
    #             operations_to_process.append(tensor.operation)

    #     collect_ops(output_tensor)

    #     # The operations_to_process list is now in a topological order.
    #     # For backward pass, we need to process them in reverse.
    #     operations_to_process.reverse()

    #     print(f"Performing backward pass. Operations to process: {[op.name for op in operations_to_process]}")
    #     for op in operations_to_process:
    #         if op.grad_fn: # In a real system, this would be a callable for backward
    #             # Pass the gradient of the output of this operation to its backward function
    #             # This is a simplified example. Real autograd handles multiple outputs and inputs
    #             # and accumulates gradients.
    #             op.grad_fn(op.output.grad)
    #         else:
    #             # If no specific grad_fn, just pass the gradient down to inputs that require it
    #             for inp_tensor in op.inputs:
    #                 if inp_tensor.requires_grad:
    #                     if inp_tensor.grad is None:
    #                         inp_tensor.grad = op.output.grad
    #                     else:
    #                         inp_tensor.grad += op.output.grad # Accumulate gradients


    def clear_graph(self):
        """
        Clears the graph and attempts to break all strong references to
        CustomTensor and Operation objects stored as node weights.
        This aids in garbage collection.
        """
        # Iterate through all nodes and set their weights to None or similar
        # to explicitly break references.
        for node_idx in list(self.graph.node_indices()):
            node_weight = self.graph[node_idx]
            if isinstance(node_weight, CustomTensor):
                # For CustomTensor, you might also want to explicitly clear its .grad and .operation
                # if they hold strong references back to other graph components, though
                # our design here minimizes that for .operation.
                # For .grad, if it's a CustomTensor, its data will be released with the tensor.
                node_weight.grad = None # Explicitly clear gradient reference
                node_weight.operation = None # Clear operation reference

            elif isinstance(node_weight, Operation):
                node_weight.inputs = []  # Clear references to input tensors
                node_weight.output = None # Clear reference to output tensor
                node_weight.grad_fn = None # Clear any function reference

            # Remove the node from the graph. This is the primary way to break rx's internal references.
            self.graph.remove_node(node_idx)

        self.graph = rx.PyDiGraph() # Reinitialize an empty graph
        self.tensor_to_node_idx = {}
        self.op_to_node_idx = {}
        gc.collect() # Suggest garbage collection to reclaim memory immediately


# # Global instance to be used by operations
# autograd_graph = AutogradGraph()


# # Example of how an operation would interact with the global graph instance
# def add_operation_func(a: CustomTensor, b: CustomTensor) -> CustomTensor:
#     global autograd_graph

#     # Perform the actual PyTorch computation
#     result_tensor_native = a._tensor + b._tensor
#     # Wrap the result in CustomTensor
#     result = CustomTensor(_tensor=result_tensor_native)

#     # Create the operation object
#     op = Operation(name="Add", inputs=[a, b], output=result)
#     result.operation = op

#     # If the graph is active, add the operation and tensors to it
#     if autograd_graph._is_active:
#         autograd_graph.build_dynamic_graph(result)

#     # Define a simplified grad_fn for this operation
#     def add_backward(grad_output_val):
#         if a.requires_grad:
#             if a.grad is None:
#                 a.grad = grad_output_val
#             else:
#                 a.grad += grad_output_val
#         if b.requires_grad:
#             if b.grad is None:
#                 b.grad = grad_output_val
#             else:
#                 b.grad += grad_output_val
#     op.grad_fn = add_backward

#     return result

# def mul_operation_func(a: CustomTensor, b: CustomTensor) -> CustomTensor:
#     global autograd_graph

#     result_tensor_native = a._tensor * b._tensor
#     result = CustomTensor(_tensor=result_tensor_native)

#     op = Operation(name="Mul", inputs=[a, b], output=result)
#     result.operation = op

#     if autograd_graph._is_active:
#         autograd_graph.build_dynamic_graph(result)

#     def mul_backward(grad_output_val):
#         if a.requires_grad:
#             if a.grad is None:
#                 a.grad = grad_output_val * b._tensor
#             else:
#                 a.grad += grad_output_val * b._tensor
#         if b.requires_grad:
#             if b.grad is None:
#                 b.grad = grad_output_val * a._tensor
#             else:
#                 b.grad += grad_output_val * a._tensor
#     op.grad_fn = mul_backward

#     return result