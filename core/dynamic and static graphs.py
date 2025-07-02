# print("--- Dynamic Graph Building ---")
# with AutogradGraph() as graph:
#     a = CustomTensor(data=[2.0], requires_grad=True)
#     b = CustomTensor(data=[3.0], requires_grad=True)
#     c = CustomTensor(data=[4.0], requires_grad=False)

#     x = add_operation_func(a, b) # x = a + b = 5
#     y = mul_operation_func(x, c) # y = x * c = 5 * 4 = 20

#     print(f"a: {a}, requires_grad: {a.requires_grad}, op: {a.operation}")
#     print(f"b: {b}, requires_grad: {b.requires_grad}, op: {b.operation}")
#     print(f"c: {c}, requires_grad: {c.requires_grad}, op: {c.operation}")
#     print(f"x: {x}, requires_grad: {x.requires_grad}, op: {x.operation}")
#     print(f"y: {y}, requires_grad: {y.requires_grad}, op: {y.operation}")

#     print("\nGraph nodes (ID and type):")
#     for node_idx in graph.graph.node_indices():
#         node_data = graph.graph[node_idx]
#         print(f"  Node {node_idx}: {node_data} (Type: {type(node_data).__name__})")

#     print("\nGraph edges:")
#     for edge in graph.graph.edge_list():
#         u, v, weight = edge
#         u_data = graph.graph[u]
#         v_data = graph.graph[v]
#         print(f"  {u_data} ({type(u_data).__name__}) --[{weight}]--> {v_data} ({type(v_data).__name__})")

#     print("\nTopological Sort:")
#     sorted_nodes = graph.topological_sort()
#     for node_idx in sorted_nodes:
#         print(f"  {graph.graph[node_idx]}")

#     print("\nPerforming backward pass for y:")
#     graph.backward(y)

#     print(f"a.grad: {a.grad}")
#     print(f"b.grad: {b.grad}")
#     print(f"c.grad: {c.grad}") # Should be None or 0, as c.requires_grad is False

#     print("\nClearing graph...")
#     graph.clear_graph()
#     print(f"Graph after clearing: {graph.graph.node_count()} nodes")

#     # Verify that references are broken by trying to access something that was in the graph
#     # This might still show the object if other strong references exist in your script,
#     # but the graph's internal references should be gone.
#     # If `y` goes out of scope here, it should be garbage collected.
#     del a, b, c, x, y # Explicitly delete variables to help GC
#     gc.collect() # Trigger garbage collection

# # After exiting the context, the global _autograd_graph_instance is None
# print("\nIs autograd_graph active after context exit?", autograd_graph._is_active)
# print("Are there any nodes left in the global graph?", autograd_graph.graph.node_count())


# print("\n--- Static Graph Building ---")
# static_graph = AutogradGraph()

# t1 = CustomTensor(data=[10.0], requires_grad=True)
# t2 = CustomTensor(data=[20.0], requires_grad=True)
# t_out = CustomTensor(data=[30.0]) # Output of some operation

# # Manually create an operation
# op1 = Operation(name="ManualOp", inputs=[t1, t2], output=t_out)
# t_out.operation = op1

# # Add nodes
# node_t1 = static_graph.add_tensor(t1)
# node_t2 = static_graph.add_tensor(t2)
# node_op1 = static_graph.add_operation(op1)
# node_t_out = static_graph.add_tensor(t_out)

# # Add edges
# static_graph.add_edge(node_t1, node_op1, "input_to")
# static_graph.add_edge(node_t2, node_op1, "input_to")
# static_graph.add_edge(node_op1, node_t_out, "produces")

# print("\nStatic graph nodes:")
# for node_idx in static_graph.graph.node_indices():
#     node_data = static_graph.graph[node_idx]
#     print(f"  Node {node_idx}: {node_data} (Type: {type(node_data).__name__})")

# print("\nStatic graph edges:")
# for edge in static_graph.graph.edge_list():
#     u, v, weight = edge
#     u_data = static_graph.graph[u]
#     v_data = static_graph.graph[v]
#     print(f"  {u_data} ({type(u_data).__name__}) --[{weight}]--> {v_data} ({type(v_data).__name__})")

# print("\nTopological Sort of Static Graph:")
# sorted_static_nodes = static_graph.topological_sort()
# for node_idx in sorted_static_nodes:
#     print(f"  {static_graph.graph[node_idx]}")

# print("\nClearing static graph...")
# static_graph.clear_graph()
# print(f"Static graph after clearing: {static_graph.graph.node_count()} nodes")

# del t1, t2, t_out, op1 # Explicitly delete variables to help GC
# gc.collect()