import torch
import rustworkx as rx
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import random

class AutogradTensor(torch.Tensor):
    """Subclass of torch.Tensor to track autograd operations"""
    
    def __new__(cls, data, op_name="leaf", parents=None, function=None):
        # Create tensor using __new__
        if isinstance(data, torch.Tensor):
            obj = data.as_subclass(cls)
        else:
            obj = torch.tensor(data).as_subclass(cls)
        
        # Add custom attributes
        obj.op_name = op_name
        obj.parents = parents or []
        obj.function = function
        obj.tensor_id = id(obj)
        
        return obj
    
    def __repr__(self):
        return f"AutogradTensor({self.op_name}, id={self.tensor_id}, shape={self.shape})"

def create_linear_layer_graph(input_size: int, hidden_sizes: List[int], output_size: int) -> rx.PyDiGraph:
    """Creates a feedforward neural network computation graph"""
    g = rx.PyDiGraph()
    node_map = {}
    
    # Input
    x = AutogradTensor(torch.randn(1, input_size), "input")
    x_node = g.add_node(x)
    node_map[x.tensor_id] = x_node
    
    current = x
    current_node = x_node
    
    # Hidden layers
    for i, hidden_size in enumerate(hidden_sizes):
        # Weight matrix
        w = AutogradTensor(torch.randn(current.shape[-1], hidden_size), f"weight_{i}")
        w_node = g.add_node(w)
        node_map[w.tensor_id] = w_node
        
        # Bias vector
        b = AutogradTensor(torch.randn(hidden_size), f"bias_{i}")
        b_node = g.add_node(b)
        node_map[b.tensor_id] = b_node
        
        # Linear transformation: x @ w
        matmul = AutogradTensor(torch.randn(1, hidden_size), f"matmul_{i}", [current, w])
        matmul_node = g.add_node(matmul)
        node_map[matmul.tensor_id] = matmul_node
        g.add_edge(current_node, matmul_node, "matmul")
        g.add_edge(w_node, matmul_node, "matmul")
        
        # Add bias: (x @ w) + b
        add_bias = AutogradTensor(torch.randn(1, hidden_size), f"add_bias_{i}", [matmul, b])
        add_bias_node = g.add_node(add_bias)
        node_map[add_bias.tensor_id] = add_bias_node
        g.add_edge(matmul_node, add_bias_node, "add")
        g.add_edge(b_node, add_bias_node, "add")
        
        # Activation: ReLU
        relu = AutogradTensor(torch.randn(1, hidden_size), f"relu_{i}", [add_bias])
        relu_node = g.add_node(relu)
        node_map[relu.tensor_id] = relu_node
        g.add_edge(add_bias_node, relu_node, "relu")
        
        current = relu
        current_node = relu_node
    
    # Output layer
    w_out = AutogradTensor(torch.randn(current.shape[-1], output_size), "weight_out")
    w_out_node = g.add_node(w_out)
    
    b_out = AutogradTensor(torch.randn(output_size), "bias_out")
    b_out_node = g.add_node(b_out)
    
    output = AutogradTensor(torch.randn(1, output_size), "output", [current, w_out, b_out])
    output_node = g.add_node(output)
    g.add_edge(current_node, output_node, "matmul")
    g.add_edge(w_out_node, output_node, "matmul")
    g.add_edge(b_out_node, output_node, "add")
    
    return g

def create_residual_block_graph(channels: int, num_blocks: int) -> rx.PyDiGraph:
    """Creates a ResNet-style residual block computation graph"""
    g = rx.PyDiGraph()
    
    # Input
    x = AutogradTensor(torch.randn(1, channels, 32, 32), "input")
    x_node = g.add_node(x)
    
    current = x
    current_node = x_node
    
    for block_idx in range(num_blocks):
        # Store the residual connection
        residual = current
        residual_node = current_node
        
        # First conv layer
        w1 = AutogradTensor(torch.randn(channels, channels, 3, 3), f"conv1_w_{block_idx}")
        w1_node = g.add_node(w1)
        
        conv1 = AutogradTensor(torch.randn(1, channels, 32, 32), f"conv1_{block_idx}", [current, w1])
        conv1_node = g.add_node(conv1)
        g.add_edge(current_node, conv1_node, "conv2d")
        g.add_edge(w1_node, conv1_node, "conv2d")
        
        # BatchNorm + ReLU
        bn1 = AutogradTensor(torch.randn(1, channels, 32, 32), f"bn1_{block_idx}", [conv1])
        bn1_node = g.add_node(bn1)
        g.add_edge(conv1_node, bn1_node, "batchnorm")
        
        relu1 = AutogradTensor(torch.randn(1, channels, 32, 32), f"relu1_{block_idx}", [bn1])
        relu1_node = g.add_node(relu1)
        g.add_edge(bn1_node, relu1_node, "relu")
        
        # Second conv layer
        w2 = AutogradTensor(torch.randn(channels, channels, 3, 3), f"conv2_w_{block_idx}")
        w2_node = g.add_node(w2)
        
        conv2 = AutogradTensor(torch.randn(1, channels, 32, 32), f"conv2_{block_idx}", [relu1, w2])
        conv2_node = g.add_node(conv2)
        g.add_edge(relu1_node, conv2_node, "conv2d")
        g.add_edge(w2_node, conv2_node, "conv2d")
        
        # BatchNorm
        bn2 = AutogradTensor(torch.randn(1, channels, 32, 32), f"bn2_{block_idx}", [conv2])
        bn2_node = g.add_node(bn2)
        g.add_edge(conv2_node, bn2_node, "batchnorm")
        
        # Residual connection: x + F(x)
        residual_add = AutogradTensor(torch.randn(1, channels, 32, 32), f"residual_add_{block_idx}", [residual, bn2])
        residual_add_node = g.add_node(residual_add)
        g.add_edge(residual_node, residual_add_node, "add")
        g.add_edge(bn2_node, residual_add_node, "add")
        
        # Final ReLU
        relu_final = AutogradTensor(torch.randn(1, channels, 32, 32), f"relu_final_{block_idx}", [residual_add])
        relu_final_node = g.add_node(relu_final)
        g.add_edge(residual_add_node, relu_final_node, "relu")
        
        current = relu_final
        current_node = relu_final_node
    
    return g

def create_attention_graph(seq_len: int, d_model: int, num_heads: int) -> rx.PyDiGraph:
    """Creates a multi-head attention computation graph"""
    g = rx.PyDiGraph()
    
    # Input embeddings
    x = AutogradTensor(torch.randn(seq_len, d_model), "input_embeddings")
    x_node = g.add_node(x)
    
    d_k = d_model // num_heads
    
    # Query, Key, Value projections
    wq = AutogradTensor(torch.randn(d_model, d_model), "weight_query")
    wk = AutogradTensor(torch.randn(d_model, d_model), "weight_key")
    wv = AutogradTensor(torch.randn(d_model, d_model), "weight_value")
    
    wq_node = g.add_node(wq)
    wk_node = g.add_node(wk)
    wv_node = g.add_node(wv)
    
    # Q, K, V computations
    q = AutogradTensor(torch.randn(seq_len, d_model), "query", [x, wq])
    k = AutogradTensor(torch.randn(seq_len, d_model), "key", [x, wk])
    v = AutogradTensor(torch.randn(seq_len, d_model), "value", [x, wv])
    
    q_node = g.add_node(q)
    k_node = g.add_node(k)
    v_node = g.add_node(v)
    
    g.add_edge(x_node, q_node, "matmul")
    g.add_edge(wq_node, q_node, "matmul")
    g.add_edge(x_node, k_node, "matmul")
    g.add_edge(wk_node, k_node, "matmul")
    g.add_edge(x_node, v_node, "matmul")
    g.add_edge(wv_node, v_node, "matmul")
    
    attention_heads = []
    
    for head in range(num_heads):
        # Reshape for multi-head attention
        q_head = AutogradTensor(torch.randn(seq_len, d_k), f"q_head_{head}", [q])
        k_head = AutogradTensor(torch.randn(seq_len, d_k), f"k_head_{head}", [k])
        v_head = AutogradTensor(torch.randn(seq_len, d_k), f"v_head_{head}", [v])
        
        q_head_node = g.add_node(q_head)
        k_head_node = g.add_node(k_head)
        v_head_node = g.add_node(v_head)
        
        g.add_edge(q_node, q_head_node, "view")
        g.add_edge(k_node, k_head_node, "view")
        g.add_edge(v_node, v_head_node, "view")
        
        # Attention scores: Q @ K^T
        scores = AutogradTensor(torch.randn(seq_len, seq_len), f"scores_{head}", [q_head, k_head])
        scores_node = g.add_node(scores)
        g.add_edge(q_head_node, scores_node, "matmul")
        g.add_edge(k_head_node, scores_node, "transpose_matmul")
        
        # Scale scores
        scale_factor = AutogradTensor(torch.tensor(1.0 / np.sqrt(d_k)), f"scale_{head}")
        scale_factor_node = g.add_node(scale_factor)
        
        scaled_scores = AutogradTensor(torch.randn(seq_len, seq_len), f"scaled_scores_{head}", [scores, scale_factor])
        scaled_scores_node = g.add_node(scaled_scores)
        g.add_edge(scores_node, scaled_scores_node, "mul")
        g.add_edge(scale_factor_node, scaled_scores_node, "mul")
        
        # Softmax
        attention_weights = AutogradTensor(torch.randn(seq_len, seq_len), f"attention_weights_{head}", [scaled_scores])
        attention_weights_node = g.add_node(attention_weights)
        g.add_edge(scaled_scores_node, attention_weights_node, "softmax")
        
        # Apply attention to values
        attended_values = AutogradTensor(torch.randn(seq_len, d_k), f"attended_values_{head}", [attention_weights, v_head])
        attended_values_node = g.add_node(attended_values)
        g.add_edge(attention_weights_node, attended_values_node, "matmul")
        g.add_edge(v_head_node, attended_values_node, "matmul")
        
        attention_heads.append((attended_values, attended_values_node))
    
    # Concatenate heads
    if len(attention_heads) > 1:
        concat_input = [head[0] for head in attention_heads]
        concat_output = AutogradTensor(torch.randn(seq_len, d_model), "concat_heads", concat_input)
        concat_output_node = g.add_node(concat_output)
        for _, head_node in attention_heads:
            g.add_edge(head_node, concat_output_node, "concat")
    else:
        concat_output_node = attention_heads[0][1]
    
    # Output projection
    wo = AutogradTensor(torch.randn(d_model, d_model), "weight_output")
    wo_node = g.add_node(wo)
    
    final_output = AutogradTensor(torch.randn(seq_len, d_model), "attention_output", [concat_output, wo])
    final_output_node = g.add_node(final_output)
    g.add_edge(concat_output_node, final_output_node, "matmul")
    g.add_edge(wo_node, final_output_node, "matmul")
    
    return g

def create_complex_dag_graph(num_inputs: int, num_intermediate: int, num_outputs: int, connectivity: float = 0.3) -> rx.PyDiGraph:
    """Creates a complex DAG with multiple paths and shared computations"""
    g = rx.PyDiGraph()
    
    # Input nodes
    inputs = []
    for i in range(num_inputs):
        tensor = AutogradTensor(torch.randn(10, 10), f"input_{i}")
        node = g.add_node(tensor)
        inputs.append((tensor, node))
    
    # Intermediate computation nodes
    intermediates = []
    for i in range(num_intermediate):
        # Randomly select parents from existing nodes
        available_parents = inputs + intermediates
        num_parents = random.randint(1, min(3, len(available_parents)))
        parents = random.sample(available_parents, num_parents)
        
        parent_tensors = [p[0] for p in parents]
        parent_nodes = [p[1] for p in parents]
        
        # Create different types of operations
        ops = ["add", "mul", "matmul", "relu", "tanh", "sigmoid"]
        op = random.choice(ops)
        
        tensor = AutogradTensor(torch.randn(10, 10), f"intermediate_{i}_{op}", parent_tensors)
        node = g.add_node(tensor)
        
        # Add edges with some probability
        for parent_node in parent_nodes:
            if random.random() < connectivity:
                g.add_edge(parent_node, node, op)
        
        intermediates.append((tensor, node))
    
    # Output nodes
    outputs = []
    for i in range(num_outputs):
        # Outputs depend on multiple intermediate nodes
        available_parents = intermediates
        num_parents = random.randint(2, min(5, len(available_parents)))
        parents = random.sample(available_parents, num_parents)
        
        parent_tensors = [p[0] for p in parents]
        parent_nodes = [p[1] for p in parents]
        
        tensor = AutogradTensor(torch.randn(10, 10), f"output_{i}", parent_tensors)
        node = g.add_node(tensor)
        
        for parent_node in parent_nodes:
            g.add_edge(parent_node, node, "final_op")
        
        outputs.append((tensor, node))
    
    return g

def benchmark_autograd_graphs():
    """Benchmark different autograd graph structures"""
    
    # Different graph types and their configurations
    graph_configs = [
        ("Linear Network", lambda: create_linear_layer_graph(100, [64, 32, 16], 10)),
        ("Residual Network", lambda: create_residual_block_graph(32, 4)),
        ("Attention", lambda: create_attention_graph(50, 128, 8)),
        ("Complex DAG", lambda: create_complex_dag_graph(10, 50, 5, 0.4)),
    ]
    
    results = {}
    
    for name, create_fn in graph_configs:
        print(f"\nBenchmarking {name}...")
        
        # Create graph
        start_time = time.perf_counter()
        graph = create_fn()
        creation_time = time.perf_counter() - start_time
        
        # Topological sort
        start_time = time.perf_counter()
        topo_order = rx.topological_sort(graph)
        topo_time = time.perf_counter() - start_time
        
        # Graph stats
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        
        results[name] = {
            'nodes': num_nodes,
            'edges': num_edges,
            'creation_time': creation_time,
            'topo_time': topo_time,
            'graph': graph
        }
        
        print(f"  Nodes: {num_nodes}, Edges: {num_edges}")
        print(f"  Creation time: {creation_time:.6f}s")
        print(f"  Topological sort time: {topo_time:.6f}s")
        
        # Show first few nodes
        print(f"  Sample nodes:")
        for i, node_idx in enumerate(topo_order[:3]):
            node = graph[node_idx]
            print(f"    {i}: {node}")
    
    return results

def visualize_results(results):
    """Create visualization of benchmark results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    names = list(results.keys())
    nodes = [results[name]['nodes'] for name in names]
    edges = [results[name]['edges'] for name in names]
    creation_times = [results[name]['creation_time'] * 1000 for name in names]  # Convert to ms
    topo_times = [results[name]['topo_time'] * 1000 for name in names]  # Convert to ms
    
    # Graph size comparison
    ax1.bar(names, nodes, alpha=0.7, color='skyblue')
    ax1.set_ylabel('Number of Nodes')
    ax1.set_title('Graph Size (Nodes)')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(names, edges, alpha=0.7, color='lightcoral')
    ax2.set_ylabel('Number of Edges')
    ax2.set_title('Graph Size (Edges)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Performance comparison
    ax3.bar(names, creation_times, alpha=0.7, color='lightgreen')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Graph Creation Time')
    ax3.tick_params(axis='x', rotation=45)
    
    ax4.bar(names, topo_times, alpha=0.7, color='gold')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Topological Sort Time')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    print("Creating and benchmarking autograd graph structures...")
    results = benchmark_autograd_graphs()
    
    print("\nVisualizing results...")
    visualize_results(results)
    
    # Example: Show the structure of one graph
    print(f"\nDetailed view of Linear Network graph:")
    linear_graph = results["Linear Network"]["graph"]
    topo_order = rx.topological_sort(linear_graph)
    
    print("Forward pass order:")
    for i, node_idx in enumerate(topo_order[:10]):  # Show first 10 nodes
        node = linear_graph[node_idx]
        in_edges = linear_graph.in_edges(node_idx)
        print(f"  Step {i}: {node}")
        if in_edges:
            print(f"    Dependencies: {len(in_edges)} nodes")