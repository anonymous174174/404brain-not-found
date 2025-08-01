# 404brain-not-found

## Overview

`404brain-not-found` is a research-oriented deep learning framework implemented in Python, inspired by PyTorch. It is designed for educational and experimental purposes, focusing on building a minimal yet extensible neural network library from scratch. The project aims to demystify the inner workings of neural network modules, custom autograd systems, and optimization algorithms by providing clear, readable, and modifiable code.

## Key Features

- **Custom Autograd Engine:**  
  Implements a computation graph and automatic differentiation system, allowing for custom tensor operations and gradient tracking.

- **Neural Network Modules:**  
  Includes core layers such as `Linear`, `Conv2d`, `BatchNorm`, pooling layers, and various activation functions, all built to be modular and extensible.

- **Loss Functions:**  
  Provides implementations for common loss functions like Mean Squared Error (MSE), Cross Entropy, and Binary Cross Entropy, with support for weighting and reduction modes.

- **Optimizers:**  
  Contains several optimization algorithms including SGD, Momentum, Nesterov, AdamW, and Lion, each with configurable hyperparameters.

- **Device and Dtype Management:**  
  Supports device (CPU/GPU) and dtype management for tensors, similar to PyTorch.

- **Testing and Experimentation:**  
  Includes a suite of test scripts and Jupyter notebooks for validating gradients, module behaviors, and experimenting with new features.

## Directory Structure

```
core/
    __init__.py
    1.py
    2.py
    autograd_graph.py
    custom_tensor.py
    losses.py
    module.py
    optimizers.py
    temp.py
    Development/
        Welcome_To_Colab (2).ipynb
        Welcome_To_Colab (5).ipynb
        working_1st august.ipynb
        working_31st_july.ipynb
    test_Scripts/
        max_poolwd_gradient.py
        test_autograd_graph.py
        test_conv_gradients.py
        testing_batchnorm.py
```

- **core/**: Main source code for the framework.
- **Development/**: Jupyter notebooks for prototyping, debugging, and documenting features.
- **test_Scripts/**: Standalone scripts for unit and integration testing of core components.

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd 404brain-not-found
   ```

2. **Install dependencies:**
   - Python 3.8+
   - PyTorch
   - rustworkx (for graph operations)
   - numpy

   You can install dependencies using pip:
   ```sh
   pip install torch rustworkx numpy
   ```

3. **Explore the code:**
   - Start with `core/module.py` and `core/custom_tensor.py` to understand the basic building blocks.
   - Review `core/autograd_graph.py` for the custom autograd implementation.
   - Check out the notebooks in `core/Development/` for usage examples and experiments.



## Contributing

This project is intended for learning and experimentation. Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

## License

This project is released under the MIT License.

---

**Note:**  
This framework is not intended for production use. It is a learning tool to understand the internals of deep learning libraries and to experiment with new ideas in neural network design and optimization.
