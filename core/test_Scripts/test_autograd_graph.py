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

from autograd_graph import AutogradGraph

def test_autograd_graph():
    print("Creating AutogradGraph...")
    graph = AutogradGraph()

    # Create dummy tensors
    t1 = CustomTensor("t1", requires_grad=True, is_leaf=False)
    t2 = CustomTensor("t2", requires_grad=True, is_leaf=False)
    t3 = CustomTensor("t3", requires_grad=True, is_leaf=False)

    # Add tensors as nodes
    print("Adding non-leaf tensors to graph...")
    graph.add_non_leaf_tensor_graph(t1)
    graph.add_non_leaf_tensor_graph(t2)
    graph.add_non_leaf_tensor_graph(t3)

    # Add tensor references
    print("Adding tensor references...")
    graph.add_non_leaf_tensor_references(t1)
    graph.add_non_leaf_tensor_references(t2)
    graph.add_non_leaf_tensor_references(t3)

    # Add edges
    print("Adding edges...")
    graph.add_edge(t1.node_id, t2.node_id)
    graph.add_edge(t2.node_id, t3.node_id)

    # Check cycle (should be acyclic)
    print("Checking for cycles (should be True):", graph.check_cycle())

    # Topological sort
    print("Reverse topological sort (should be [t3, t2, t1]):")
    tensors = graph.reverse_toposort()
    print([t().name for t in tensors if t is not None])

    # Delete edge
    print("Deleting edge t2 -> t3...")
    graph.delete_edge(t2.node_id, t3.node_id)

    # Try deleting a non-existent edge (should raise)
    try:
        graph.delete_edge(t2.node_id, t3.node_id)
    except ValueError as e:
        print("Caught expected error:", e)

    # Delete node
    print("Deleting node t3...")
    graph.delete_node(t3.node_id)

    # Try deleting a non-existent node (should raise)
    try:
        graph.delete_node(t3.node_id)
    except ValueError as e:
        print("Caught expected error:", e)

    # Delete tensor reference
    print("Deleting tensor reference for t1...")
    graph.del_non_leaf_tensor_reference(t1.node_id)
    print("Test complete.")

if __name__ == "__main__":
    test_autograd_graph()