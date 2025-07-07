# import rustworkx as rx
# import weakref

# class AutogradGraph:
#     __slots__ = ("graph", "intermediate_tensors")

#     def __init__(self):
#         self.graph = rx.PyDiGraph()
#         self.intermediate_tensors = {}

#     def add_non_leaf_tensor_graph(self, tensor):
#         requires_grad = tensor._custom_requires_grad
#         is_leaf = tensor._is_leaf

#         if not requires_grad or is_leaf:
#             raise ValueError("Tensor must be a non leaf tensor to be added to the graph.")

#         graph = self.graph
#         tensor_index = graph.add_node(weakref.ref(tensor))
#         tensor.node_id = tensor_index

#     def add_non_leaf_tensor_references(self, tensor):
#         requires_grad = tensor._custom_requires_grad
#         is_leaf = tensor._is_leaf
#         node_id = tensor.node_id

#         if not requires_grad or is_leaf:
#             raise ValueError("Tensor must be a non leaf tensor.")

#         if node_id in self.intermediate_tensors:
#             raise ValueError("Tensor reference to persist in memory already exists.")

#         self.intermediate_tensors[node_id] = tensor

#     def add_edge(self, node_from, node_to, weight=None):
#         graph = self.graph

#         if not isinstance(node_from, int) or not isinstance(node_to, int):
#             raise TypeError("Node indices must be integers.")

#         if not graph.has_node(node_from) or not graph.has_node(node_to):
#             raise ValueError("Both nodes must exist in the graph before adding an edge.")

#         graph.add_edge(node_from, node_to, weight)

#     def check_cycle(self):
#         return rx.is_directed_acyclic_graph(self.graph)

#     def __repr__(self):
#         graph = self.graph
#         return f"CustomAutogradGraph(nodes={graph.num_nodes()}, edges={graph.num_edges()})"

#     def reverse_toposort(self):
#         graph = self.graph
#         if not self.check_cycle():
#             raise RuntimeError("Cannot perform topological sort on a graph with cycles.")

#         node_indexes = rx.topological_sort(graph)
#         return [graph[node_index] for node_index in reversed(node_indexes)]

#     def delete_node(self, node_index):
#         if not isinstance(node_index, int):
#             raise TypeError("Node index must be an integer.")

#         graph = self.graph
#         if not graph.has_node(node_index):
#             raise ValueError(f"Node index {node_index} does not exist in the graph.")

#         graph.remove_node(node_index)

#     def delete_edge(self, node_from, node_to):
#         if not isinstance(node_from, int) or not isinstance(node_to, int):
#             raise TypeError("Node indices must be integers.")

#         graph = self.graph
#         if not graph.has_edge(node_from, node_to):
#             raise ValueError(f"Edge ({node_from}, {node_to}) does not exist in the graph.")

#         graph.remove_edge(node_from, node_to)

#     def del_non_leaf_tensor_reference(self, tensor_node_id):
#         try:
#             del self.intermediate_tensors[tensor_node_id]
#         except KeyError:
#             raise KeyError(f"No tensor reference found for node ID {tensor_node_id}")

import rustworkx as rx
import weakref


class AutogradGraph:
    __slots__ = ("graph", "intermediate_tensors", "_check_cycles", "_auto_cleanup")

    def __init__(self, check_for_cycles=True, auto_cleanup=True):
        self.graph = rx.PyDiGraph()
        self.intermediate_tensors = {}
        self._check_cycles = check_for_cycles
        self._auto_cleanup = auto_cleanup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._check_cycles:
            if not self.check_cycle():
                raise RuntimeError("Cycle detected in autograd graph on context exit.")

        if self._auto_cleanup:
            self.intermediate_tensors.clear()
            self.graph.clear()  # Clears all nodes and edges

    def add_tensor_graph(self, tensor):
        requires_grad = tensor._custom_requires_grad

        if not requires_grad:
            raise ValueError("Tensor with require grad False cannot to be added to the graph.")

        tensor_index = self.graph.add_node(weakref.ref(tensor))
        tensor.node_id = tensor_index

    def add_non_leaf_tensor_references(self, tensor):
        requires_grad = tensor._custom_requires_grad
        is_leaf = tensor._is_leaf
        node_id = tensor.node_id

        if not requires_grad or is_leaf:
            raise ValueError("Tensor must be a non leaf tensor.")

        if node_id in self.intermediate_tensors:
            raise ValueError("Tensor reference to persist in memory already exists.")

        self.intermediate_tensors[node_id] = tensor

    def add_edge(self, node_from, node_to, weight=None):
        if not isinstance(node_from, int) or not isinstance(node_to, int):
            raise TypeError("Node indices must be integers.")

        graph = self.graph
        if not graph.has_node(node_from) or not graph.has_node(node_to):
            raise ValueError("Both nodes must exist in the graph before adding an edge.")

        graph.add_edge(node_from, node_to, weight)

    def check_cycle(self):
        return rx.is_directed_acyclic_graph(self.graph)

    def reverse_toposort(self):
        graph = self.graph
        if not self.check_cycle():
            raise RuntimeError("Cannot perform topological sort on a graph with cycles.")

        node_indexes = rx.topological_sort(graph)
        return [graph[node_index] for node_index in reversed(node_indexes)]

    def delete_node(self, node_index):
        if not isinstance(node_index, int):
            raise TypeError("Node index must be an integer.")

        graph = self.graph
        if not graph.has_node(node_index):
            raise ValueError(f"Node index {node_index} does not exist in the graph.")

        graph.remove_node(node_index)

    def delete_edge(self, node_from, node_to):
        if not isinstance(node_from, int) or not isinstance(node_to, int):
            raise TypeError("Node indices must be integers.")

        graph = self.graph
        if not graph.has_edge(node_from, node_to):
            raise ValueError(f"Edge ({node_from}, {node_to}) does not exist in the graph.")

        graph.remove_edge(node_from, node_to)

    def del_non_leaf_tensor_reference(self, tensor_node_id):
        try:
            del self.intermediate_tensors[tensor_node_id]
        except KeyError:
            raise KeyError(f"No tensor reference found for node ID {tensor_node_id}")

    def __repr__(self):
        graph = self.graph
        return f"CustomAutogradGraph(nodes={graph.num_nodes()}, edges={graph.num_edges()})"
