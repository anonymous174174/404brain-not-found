# Neuronix 🧠

A custom deep learning framework built from scratch in Python for educational and research purposes. This project implements a complete neural network library with automatic differentiation, demonstrating the inner workings of modern ML frameworks like PyTorch.

## ✨ What This Is

This is a learning-focused implementation that recreates core deep learning functionality from the ground up. It's designed to help understand how autograd engines, neural network modules, and optimization algorithms actually work under the hood.

**⚠️ Important:** This is an educational tool, not a production framework. Use PyTorch, TensorFlow, or JAX for real applications.

## 🚀 Features

### Core Autograd System

- **Custom computation graph** with automatic differentiation
- **Broadcasting-aware gradient computation** for complex tensor operations
- **Memory management** with automatic cleanup and cycle detection
- **Graph validation** with topological sorting for correct backward passes

### Neural Network Components

- **Tensor operations**: Standard arithmetic, matrix ops, and mathematical functions
- **Layers**: Linear, Conv2d, BatchNorm, pooling layers
- **Activations**: ReLU, GELU, Sigmoid, Tanh, SiLU, ELU, Leaky ReLU
- **Loss functions**: MSE, CrossEntropy, BCEWithLogits
- **Optimizers**: SGD, Momentum, Nesterov, AdamW, Lion

### Performance Features

- **JIT compilation** integration with `torch.compile` for critical paths
- **Efficient memory patterns** with weak references
- **Broadcasting optimization** for gradient reduction

## 🏗️ Architecture Overview

The framework consists of three main components:

1. **AutogradGraph**: Manages the computation graph for automatic differentiation
2. **CustomTensor**: Core tensor class with gradient tracking capabilities  
3. **Modules**: Neural network layers and operations built on top of the tensor system


## 🧪 What You Can Learn

This implementation helps understand:

- **Automatic differentiation**: How gradients are computed and backpropagated
- **Computation graphs**: How modern ML frameworks track operations
- **Memory management**: Preventing memory leaks in dynamic graphs
- **Broadcasting**: How tensor operations work with different shapes
- **Optimization algorithms**: Implementation details of popular optimizers
- **Neural network modules**: How layers compose and interact

## ⚡ Technical Highlights

### Memory Management

```python
def __exit__(self, exc_type, exc_value, traceback):
    if self._check_cycles and self.check_cycle():
        raise RuntimeError("Cycle detected in autograd graph")
    if self._auto_cleanup:
        self.intermediate_tensors.clear()
        self.graph.clear()
```

### JIT-Compiled Operations

```python
@torch.compile
def _reduce_grad_for_broadcast(self, grad, target_shape):
    """JIT-compiled gradient reduction for broadcasting operations."""
    # Custom broadcasting logic for arbitrary dimensional tensors
```

### Graph-Based Backward Pass

```python
def reverse_toposort_from_tensor(self, tensor_index):
    predecessors = list(rx.ancestors(self.graph, tensor_index))
    predecessors.append(tensor_index)
    sub_graph = self.graph.subgraph(predecessors)
    return [sub_graph[i] for i in reversed(rx.topological_sort(sub_graph))]
```

## 🔍 Testing & Validation

The implementation has been validated against PyTorch for:

- ✅ Gradient correctness across all operations upto rtol 1e-4
- ✅ Broadcasting behavior consistency  
- ✅ Memory usage patterns
- ✅ Numerical stability

## 📂 Project Structure

```bash
404brain-not-found/
├── core/
│   ├── __init__.py          # Core tensor and graph classes
│   ├── module.py            # Neural network layers
│   ├── losses.py            # Loss functions
│   └── optimizers.py        # Optimization algorithms
├── tests/                   # Test scripts and notebooks
├── examples/               # Usage examples and tutorials
└── README.md
```

## 🤝 Contributing

This is a learning project! Contributions, bug reports, and feature requests are welcome. Areas for exploration:

- [ ] Second-order derivatives (Hessian computation)
- [ ] Sparse gradient operations  
- [ ] Additional optimization algorithms
- [ ] More activation functions
- [ ] Custom loss functions

## 📚 Learning Resources

This project implements concepts from:

- PyTorch autograd internals
- Modern optimization papers (Adam, Lion, etc.)
- Numerical methods for gradient computation

## 🎯 Why Build This?

Understanding how autograd works is crucial for:

- Debugging gradient issues in real projects
- Implementing custom operations efficiently
- Contributing to ML frameworks
- Research involving novel differentiation techniques
- Building domain-specific ML tools

## 📄 License

MIT License - feel free to use this for learning and experimentation.

---

*Built to demystify the magic behind modern deep learning frameworks. Every operation implemented from first principles with educational clarity in mind.* 🎓
