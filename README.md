# Neuronix

A custom deep learning framework built from first principles in Python for Learning purposes. This project implements a complete neural network library with a custom automatic differentiation engine, trying to deconstruct the inner workings of modern ML frameworks like PyTorch.

**⚠️**  Neuronix is built for learning, experimentation, and understanding deep learning internals. It is not intended for production use.

## 🚀 What It Does
Neuronix reimplements the core building blocks of modern ML frameworks from scratch, including:

- A fully custom **autograd engine**

- A computation graph that supports **broadcast-aware backward passes**

- Basic and advanced **neural network** layers

- Optimizers like AdamW and Lion

- A tensor abstraction wrapping low-level PyTorch ops while managing its own graph and gradient system

## 🚀 Key Features

### Autograd Engine

- Pure-Python computation graph built independently of PyTorch autograd
- Reverse-mode differentiation using topological sort
- Supports broadcasting-aware gradient propagation
- Automatic cleanup and memory management using weak references

### Core Components

- **CustomTensor**: Gradient-tracking tensor wrapper
- **AutogradGraph**: Manages dynamic computation graph
- **Module**: Neural network layers and forward hooks
- **Layers**: Linear, Conv2d, BatchNorm, Pooling, etc.
- **Activations**: ReLU, GELU, Tanh, SiLU, ELU, Leaky ReLU, Swish
- **Losses**: MSE, CrossEntropy, BCEWithLogits
- **Optimizers**: SGD, Momentum, Nesterov, AdamW, Lion


## 🏗️ Why Build This?

Neuronix was created to:

1. Deconstruct how autograd engines actually compute gradients
2. Understand the internals of layers, modules, optimizers, and tensor ops
3. Build a usable (and testable) DL framework from scratch
Trained ResNet-18 on CIFAR-10 using Neuronix with over **88% test accuracy** — validating the correctness and expressiveness of the engine.

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
│   ├── lr_scheduler.py      # Learning Rate Schedulers
│   └── optimizers.py        # Optimization algorithms
├── tests/                   # Test scripts and notebooks
├── examples/                # Example Architectures trained on Cifar Datasets (To be added)
└── README.md
```

## 🧪 Key Concepts Explored

- **Automatic differentiation**: How gradients are computed and backpropagated
- **Computation graphs**: How modern ML frameworks track operations
- **Memory management**: Preventing memory leaks in dynamic graphs
- **Broadcasting**: How tensor operations work with different shapes
- **Optimization algorithms**: Implementation details of popular optimizers
- **Neural network modules**: How layers compose and interact

## 🎯 Future Directions
Areas for exploration for future:
- [ ] Higher-order derivatives
- [ ] Additional Modules
- [ ] Mutli-Gpu Graph Management
- [ ] Object Pooling for Efficiency
- [ ] Better Methods for saving Models

## 📚 References & Inspiration
- PyTorch & JAX internals
- Backpropagation from first principles
- Optimizer research: Adam, Lion, Momentum variants
- Stanford CS231n, Karpathy's micrograd, MIT 6.S191

## 📄 License

MIT License - feel free to use this for learning and experimentation.

---

*Built to understand the madness and brilliance behind modern deep learning frameworks.* 🎓
