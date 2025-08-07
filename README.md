# Neuronix 🧠

A custom deep learning framework built from first principles in Python for Learning and research purposes. This project implements a complete neural network library with a custom automatic differentiation engine, trying to deconstruct the inner workings of modern ML frameworks like PyTorch.

## ✨ What This Is

This is a learning-focused implementation that recreates core deep learning functionality from the ground up. It's designed to help understand how autograd engines, neural network modules, and optimization algorithms actually work under the hood.

**⚠️ Disclaimer:** Neuronix is for learning and research purposes only. For production applications, use established frameworks like PyTorch, TensorFlow, or JAX.

## 🚀 Features

### Core Autograd System

- **Custom computation graph** with automatic differentiation
- **Broadcasting-aware gradient computation** for complex tensor operations
- **Memory management** with automatic cleanup and cycle detection
- **Graph validation** with topological sorting for correct backward passes

### Neural Network Components

- **Tensor operations**: Standard arithmetic, matrix ops, and mathematical functions
- **Layers**: Linear, Conv2d, BatchNorm, pooling layers
- **Activations**: ReLU, GELU, Sigmoid, Tanh, SiLU, ELU, Leaky ReLU, Swish
- **Loss functions**: MSE, CrossEntropy, BCEWithLogits
- **Optimizers**: SGD, Momentum, Nesterov, AdamW, Lion

### Performance Features

- **Efficient memory patterns** with weak references
- **Broadcasting optimization** for gradient reduction

## 🏗️ Architecture Overview

The framework consists of three main components:

1. **AutogradGraph**: Manages the computation graph for automatic differentiation
2. **CustomTensor**: Core tensor class with gradient tracking capabilities  
3. **Modules**: Neural network layers and operations built on top of the tensor system


## 🧪 What I Learned

- **Automatic differentiation**: How gradients are computed and backpropagated
- **Computation graphs**: How modern ML frameworks track operations
- **Memory management**: Preventing memory leaks in dynamic graphs
- **Broadcasting**: How tensor operations work with different shapes
- **Optimization algorithms**: Implementation details of popular optimizers
- **Neural network modules**: How layers compose and interact


## 🔍 Testing & Validation

The implementation has been validated against PyTorch for:

- ✅ Gradient correctness across all operations upto rtol 1e-6 and atol 1e-6 on float32 and rtol 1e-7 and atol 1e-12 on float 64
- ✅ Broadcasting behavior consistency  
- ✅ Memory usage patterns
- ✅ Numerical stability
You can verify the same by running /tests/test_comprehensive.py

## 📂 Project Structure

```bash
404brain-not-found/
├── src/neuronix/
│   ├── autograd_graph.py    # Autograd Graph Context Manager
│   ├── custom_tensor.py     # Tensor object (A wrapper around torch tensor)
│   ├── module.py            # Neural network layers
│   ├── losses.py            # Loss functions
│   └── optimizers.py        # Optimization algorithms
├── tests/                   # Test scripts and notebooks
├── examples/                # Example Architectures trained on Cifar Datasets (To be added)
└── README.md
```

## 🤝 Contributing

This is a learning project Contributions, bug reports are welcome. Areas for exploration for future:

- [ ] Higher-order derivatives
- [ ] Additional Modules
- [ ] Mutli-Gpu Graph Management
- [ ] Object Pooling for Efficiency
- [ ] Better Methods for saving Models

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

*Built to understand the madness and brilliance behind modern deep learning frameworks.* 🎓
