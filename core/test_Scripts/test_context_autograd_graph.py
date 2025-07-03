import sys
sys.path.append(r"c:\Users\darth\Desktop\Deep Learning\New folder\404brain-not-found\core")

import rustworkx as rx
import weakref

# Dummy CustomTensor class for testing
class CustomTensor:
    def __init__(self, name, requires_grad=True, is_leaf=False):
        self.name = name
        self._custom_requires_grad = requires_grad
        self._is_leaf = is_leaf
        self.node_id = None

    def __repr__(self):
        return f"CustomTensor({self.name})"
    def __del__(self):
        print(f"CustomTensor {self.name} being garbage collected.")

from autograd_graph import AutogradGraph

def test_autograd_graph_context_manager():
    print("Starting AutogradGraph context-managed test...\n")

    t1 = CustomTensor("t1", requires_grad=True, is_leaf=False)
    t2 = CustomTensor("t2", requires_grad=True, is_leaf=False)
    t3 = CustomTensor("t3", requires_grad=True, is_leaf=False)

    try:
        with AutogradGraph(check_for_cycles=True, auto_cleanup=True) as graph:
            print("Graph initialized inside context block.")

            print("Adding tensors to graph...")
            graph.add_non_leaf_tensor_graph(t1)
            graph.add_non_leaf_tensor_graph(t2)
            graph.add_non_leaf_tensor_graph(t3)

            print("Adding tensor references...")
            graph.add_non_leaf_tensor_references(t1)
            graph.add_non_leaf_tensor_references(t2)
            graph.add_non_leaf_tensor_references(t3)

            print("Adding edges...")
            graph.add_edge(t1.node_id, t2.node_id)
            graph.add_edge(t2.node_id, t3.node_id)

            print("Checking for cycles (should be True):", graph.check_cycle())

            print("Reverse topological sort (expecting [t3, t2, t1]):")
            tensors = graph.reverse_toposort()
            print([t().name for t in tensors if t is not None])

            print("Deleting edge t2 -> t3...")
            graph.delete_edge(t2.node_id, t3.node_id)

            print("Attempting to delete non-existent edge (expect error)...")
            try:
                graph.delete_edge(t2.node_id, t3.node_id)
            except ValueError as e:
                print("Caught expected error:", e)

            print("Deleting node t3...")
            graph.delete_node(t3.node_id)

            print("Attempting to delete already deleted node (expect error)...")
            try:
                graph.delete_node(t3.node_id)
            except ValueError as e:
                print("Caught expected error:", e)

            print("Deleting tensor reference for t1...")
            graph.del_non_leaf_tensor_reference(t1.node_id)

            print("Exiting context block... cleanup and cycle check will run.\n")

    except RuntimeError as e:
        print("Cycle check failed on context exit:", e)
    print(f"Reference counts of the three tensors are t1 {sys.getrefcount(t1)}, t2 {sys.getrefcount(t2)}, t3 {sys.getrefcount(t3)}")
    # t1=None
    # t2=None
    # t3=None

    #print(f"Reference counts of the three tensors are t1 {sys.getrefcount(t1)}, t2 {sys.getrefcount(t2)}, t3 {sys.getrefcount(t3)}")
    del t1
    del t2
    del t3
    print("Context exited. Validating cleanup...")

    # Confirm that intermediate tensors and graph were cleaned up
    # This block checks side effects post-exit (auto_cleanup)
    # print("Graph and references should be cleared:")
    # print("t1.node_id:", t1.node_id)
    print("Test complete.\n")


if __name__ == "__main__":
    test_autograd_graph_context_manager()
