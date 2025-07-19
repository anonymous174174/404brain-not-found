import rustworkx as rx
import weakref
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
if __name__ == "__main__":
    pass