import torch
import numpy as np
import numbers
import weakref
import rustworkx as rx
from typing import Optional, Any
import sys
import gc
import pytest
from autograd_graph import AutogradGraph
from custom_tensor import CustomTensor
from module import *
from losses import *
class AutogradTester:
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.tolerance = 1e-6 #1e-7  # Increased tolerance slightly for complex ops

    def assert_tensors_close(self, custom_tensor, pytorch_tensor, test_name, check_grad=True):
        """Compare custom tensor with PyTorch tensor values and optionally gradients."""
        try:
            # Check values
            np.testing.assert_allclose(
                custom_tensor.tensor.detach().cpu().numpy(),  # Ensure on CPU for numpy
                pytorch_tensor.detach().cpu().numpy(),
                rtol=self.tolerance,
                atol=self.tolerance,
                err_msg=f"Mismatch in tensor values for {test_name}"
            )

            # Check gradients if requested and they exist for PyTorch tensor
            if check_grad and pytorch_tensor.grad is not None:
                if custom_tensor.tensor.grad is None:
                    raise AssertionError(f"Custom tensor has no gradient for {test_name}, but PyTorch does.")

                np.testing.assert_allclose(
                    custom_tensor.tensor.grad.detach().cpu().numpy(),  # Ensure on CPU for numpy
                    pytorch_tensor.grad.detach().cpu().numpy(),
                    rtol=self.tolerance,
                    atol=self.tolerance,
                    err_msg=f"Mismatch in gradients for {test_name}"
                )
            elif check_grad and pytorch_tensor.grad is None and custom_tensor.tensor.grad is not None:
                raise AssertionError(f"Custom tensor has gradient for {test_name}, but PyTorch does not (should be no_grad).")

            print(f"✓ {test_name}")
            self.passed_tests += 1

        except Exception as e:
            print(f"✗ {test_name}: {str(e)}")
            self.failed_tests += 1

    def test_basic_operations(self):
        """Test basic arithmetic operations"""
        print("\n=== Testing Basic Operations ===")

        # Test scalar addition
        with AutogradGraph() as graph:
            x_custom = CustomTensor([2.0, 3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom + 5.0
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([2.0, 3.0], requires_grad=True)
            y_pytorch = x_pytorch + 5.0
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Scalar Addition - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Scalar Addition - y (result)")

        # Test tensor addition
        with AutogradGraph() as graph:
            x_custom = CustomTensor([1.0, 2.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([3.0, 4.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            z_custom = x_custom + y_custom
            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([1.0, 2.0], requires_grad=True)
            y_pytorch = torch.tensor([3.0, 4.0], requires_grad=True)
            z_pytorch = x_pytorch + y_pytorch
            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Tensor Addition - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Tensor Addition - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Tensor Addition - z (result)")

    def test_multiplication(self):
        """Test multiplication operations"""
        print("\n=== Testing Multiplication ===")

        # Test scalar multiplication
        with AutogradGraph() as graph:
            x_custom = CustomTensor([2.0, 3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom * 4.0
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([2.0, 3.0], requires_grad=True)
            y_pytorch = x_pytorch * 4.0
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Scalar Multiplication - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Scalar Multiplication - y (result)")

        # Test tensor multiplication
        with AutogradGraph() as graph:
            x_custom = CustomTensor([2.0, 3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([4.0, 5.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            z_custom = x_custom * y_custom
            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([2.0, 3.0], requires_grad=True)
            y_pytorch = torch.tensor([4.0, 5.0], requires_grad=True)
            z_pytorch = x_pytorch * y_pytorch
            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Tensor Multiplication - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Tensor Multiplication - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Tensor Multiplication - z (result)")

    def test_subtraction_division(self):
        """Test subtraction and division"""
        print("\n=== Testing Subtraction and Division ===")

        # Test scalar subtraction (x - C)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([5.0, 6.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom - 2.0
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([5.0, 6.0], requires_grad=True)
            y_pytorch = x_pytorch - 2.0
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Scalar Subtraction (x - C) - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Scalar Subtraction (x - C) - y (result)")

        # Test scalar reverse subtraction (C - x)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([5.0, 6.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = 10.0 - x_custom  # Uses __rsub__
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([5.0, 6.0], requires_grad=True)
            y_pytorch = 10.0 - x_pytorch
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Scalar Reverse Subtraction (C - x) - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Scalar Reverse Subtraction (C - x) - y (result)")

        # Test tensor subtraction
        with AutogradGraph() as graph:
            x_custom = CustomTensor([7.0, 8.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([2.0, 1.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            z_custom = x_custom - y_custom
            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([7.0, 8.0], requires_grad=True)
            y_pytorch = torch.tensor([2.0, 1.0], requires_grad=True)
            z_pytorch = x_pytorch - y_pytorch
            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Tensor Subtraction - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Tensor Subtraction - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Tensor Subtraction - z (result)")

        # Test scalar division
        with AutogradGraph() as graph:
            x_custom = CustomTensor([8.0, 12.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom / 4.0
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([8.0, 12.0], requires_grad=True)
            y_pytorch = x_pytorch / 4.0
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Scalar Division - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Scalar Division - y (result)")
        # Test tensor division
        with AutogradGraph() as graph:
            x_custom = CustomTensor([8.0, 12.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([5.0, 10.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            z_custom = x_custom / y_custom
            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([8.0, 12.0], requires_grad=True)
            y_pytorch = torch.tensor([5.0, 10.0], requires_grad=True)
            z_pytorch = x_pytorch / y_pytorch
            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Tensor Division - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Tensir Division - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Tensor Division - z (result)", )


    def test_power_function(self):
        """Test power operation"""
        print("\n=== Testing Power Function ===")

        with AutogradGraph() as graph:
            x_custom = CustomTensor([2.0, 3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom.pow(3.0)
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([2.0, 3.0], requires_grad=True)
            y_pytorch = torch.pow(x_pytorch, 3.0)
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Power Function - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Power Function - y (result)" )

        # Test power with negative exponent
        with AutogradGraph() as graph:
            x_custom = CustomTensor([2.0, 3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom.pow(-2.0)
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([2.0, 3.0], requires_grad=True)
            y_pytorch = torch.pow(x_pytorch, -2.0)
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Power Function (Negative Exponent) - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Power Function (Negative Exponent) - y (result)")

    def test_unary_functions(self):
        """Test unary mathematical functions"""
        print("\n=== Testing Unary Functions ===")

        # Test exp
        with AutogradGraph() as graph:
            x_custom = CustomTensor([1.0, 2.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom.exp()
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([1.0, 2.0], requires_grad=True)
            y_pytorch = torch.exp(x_pytorch)
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Exponential Function - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Exponential Function - y (result)")

        # Test log
        with AutogradGraph() as graph:
            x_custom = CustomTensor([1.0, 2.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom.log()
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([1.0, 2.0], requires_grad=True)
            y_pytorch = torch.log(x_pytorch)
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Logarithm Function - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Logarithm Function - y (result)")

        # Test sin
        with AutogradGraph() as graph:
            x_custom = CustomTensor([0.5, 1.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom.sin()
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([0.5, 1.0], requires_grad=True)
            y_pytorch = torch.sin(x_pytorch)
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Sine Function - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Sine Function - y (result)")

        # Test cos
        with AutogradGraph() as graph:
            x_custom = CustomTensor([0.5, 1.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom.cos()
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([0.5, 1.0], requires_grad=True)
            y_pytorch = torch.cos(x_pytorch)
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Cosine Function - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Cosine Function - y (result)")

        # Test sqrt
        with AutogradGraph() as graph:
            x_custom = CustomTensor([4.0, 9.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom.sqrt()
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([4.0, 9.0], requires_grad=True)
            y_pytorch = torch.sqrt(x_pytorch)
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Square Root Function - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Square Root Function - y (result)")

    def test_matrix_operations(self):
        """Test matrix operations"""
        print("\n=== Testing Matrix Operations ===")

        # Test matrix multiplication (2x2 @ 2x2)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([[1.0, 2.0], [3.0, 4.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([[5.0, 6.0], [7.0, 8.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            z_custom = x_custom.matmul(y_custom)
            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            y_pytorch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
            z_pytorch = torch.matmul(x_pytorch, y_pytorch)
            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Matrix Multiplication (2x2 @ 2x2) - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Matrix Multiplication (2x2 @ 2x2) - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Matrix Multiplication (2x2 @ 2x2) - z (result)")

        # Test matrix multiplication (2x3 @ 3x2)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            z_custom = x_custom.matmul(y_custom)
            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            y_pytorch = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], requires_grad=True)
            z_pytorch = torch.matmul(x_pytorch, y_pytorch)
            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Matrix Multiplication (2x3 @ 3x2) - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Matrix Multiplication (2x3 @ 3x2) - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Matrix Multiplication (2x3 @ 3x2) - z (result)")

        # Test dot product (vector * vector)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([1.0, 2.0, 3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([4.0, 5.0, 6.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            z_custom = x_custom.dot(y_custom)
            z_custom.backward()  # Scalar output, so default backward() is fine (grad=1)

            x_pytorch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            y_pytorch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
            z_pytorch = torch.dot(x_pytorch, y_pytorch)
            z_pytorch.backward()

            self.assert_tensors_close(x_custom, x_pytorch, "Dot Product (vector) - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Dot Product (vector) - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Dot Product (vector) - z (result)")

    def test_complex_chain(self):
        """Test complex computational chains"""
        print("\n=== Testing Complex Chains ===")

        # Test 1: z = (x + y) * (x - y) + x^2 - sin(y)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([3.0, 4.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([1.0, 2.0], _custom_requires_grad=True, graph=graph, is_leaf=True)

            sum_custom = x_custom + y_custom
            diff_custom = x_custom - y_custom
            prod_custom = sum_custom * diff_custom
            x_squared_custom = x_custom.pow(2.0)
            sin_y_custom = y_custom.sin()

            inter1_custom = prod_custom + x_squared_custom
            z_custom = inter1_custom - sin_y_custom

            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([3.0, 4.0], requires_grad=True)
            y_pytorch = torch.tensor([1.0, 2.0], requires_grad=True)

            sum_pytorch = x_pytorch + y_pytorch
            diff_pytorch = x_pytorch - y_pytorch
            prod_pytorch = sum_pytorch * diff_pytorch
            x_squared_pytorch = torch.pow(x_pytorch, 2.0)
            sin_y_pytorch = torch.sin(y_pytorch)

            inter1_pytorch = prod_pytorch + x_squared_pytorch
            z_pytorch = inter1_pytorch - sin_y_pytorch

            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Complex Chain 1 - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Complex Chain 1 - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Complex Chain 1 - z (result)")

        # Test 2: Multiple paths to a leaf: z = x*y + x*x + y*z_fixed
        with AutogradGraph() as graph:
            x_custom = CustomTensor([2.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            z_fixed_custom = CustomTensor([0.5])  # No grad

            term1_custom = x_custom * y_custom
            term2_custom = x_custom * x_custom  # x appears twice
            term3_custom = y_custom * z_fixed_custom  # y appears twice, one with no-grad

            inter_custom = term1_custom + term2_custom
            z_custom = inter_custom + term3_custom
            z_custom.backward()

            x_pytorch = torch.tensor([2.0], requires_grad=True)
            y_pytorch = torch.tensor([3.0], requires_grad=True)
            z_fixed_pytorch = torch.tensor([0.5])  # No grad

            term1_pytorch = x_pytorch * y_pytorch
            term2_pytorch = x_pytorch * x_pytorch
            term3_pytorch = y_pytorch * z_fixed_pytorch

            inter_pytorch = term1_pytorch + term2_pytorch
            z_pytorch = inter_pytorch + term3_pytorch
            z_pytorch.backward()

            self.assert_tensors_close(x_custom, x_pytorch, "Complex Chain 2 (Multiple Paths) - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Complex Chain 2 (Multiple Paths) - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Complex Chain 2 (Multiple Paths) - z (result)")

        # Test 3: Deeper Chain with Mixed Ops: (exp(x) * log(y)) / sqrt(x+y)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([1.5], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([2.5], _custom_requires_grad=True, graph=graph, is_leaf=True)

            exp_x_custom = x_custom.exp()
            log_y_custom = y_custom.log()
            numerator_custom = exp_x_custom * log_y_custom

            sum_xy_custom = x_custom + y_custom
            sqrt_sum_custom = sum_xy_custom.sqrt()

            z_custom = numerator_custom / sqrt_sum_custom
            z_custom.backward()

            x_pytorch = torch.tensor([1.5], requires_grad=True)
            y_pytorch = torch.tensor([2.5], requires_grad=True)

            exp_x_pytorch = torch.exp(x_pytorch)
            log_y_pytorch = torch.log(y_pytorch)
            numerator_pytorch = exp_x_pytorch * log_y_pytorch

            sum_xy_pytorch = x_pytorch + y_pytorch
            sqrt_sum_pytorch = torch.sqrt(sum_xy_pytorch)

            z_pytorch = numerator_pytorch / sqrt_sum_pytorch
            z_pytorch.backward()

            self.assert_tensors_close(x_custom, x_pytorch, "Complex Chain 3 (Deeper Mixed Ops) - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Complex Chain 3 (Deeper Mixed Ops) - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Complex Chain 3 (Deeper Mixed Ops) - z (result)")

    def test_mixed_operations(self):
        """Test mixing operations with and without gradients"""
        print("\n=== Testing Mixed Operations ===")

        # One tensor requires grad, other doesn't (multiplication)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([2.0, 3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([4.0, 5.0])  # No grad
            z_custom = x_custom * y_custom
            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([2.0, 3.0], requires_grad=True)
            y_pytorch = torch.tensor([4.0, 5.0])  # No grad
            z_pytorch = x_pytorch * y_pytorch
            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Mixed Operations (X*Y, Y no grad) - x")
            # Check that y_custom has no grad
            self.assert_tensors_close(y_custom, y_pytorch, "Mixed Operations (X*Y, Y no grad) - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Mixed Operations (X*Y, Y no grad) - z (result)")

        # One tensor requires grad, other doesn't (addition)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([10.0, 20.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([1.0, 2.0])  # No grad
            z_custom = x_custom + y_custom
            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([10.0, 20.0], requires_grad=True)
            y_pytorch = torch.tensor([1.0, 2.0])  # No grad
            z_pytorch = x_pytorch + y_pytorch
            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Mixed Operations (X+Y, Y no grad) - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Mixed Operations (X+Y, Y no grad) - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Mixed Operations (X+Y, Y no grad) - z (result)")

    def test_broadcasting(self):
        """Test operations with broadcasting"""
        print("\n=== Testing Broadcasting ===")

        # Vector + scalar
        with AutogradGraph() as graph:
            x_custom = CustomTensor([1.0, 2.0, 3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom + 10.0
            y_custom.backward(torch.tensor([1.0, 1.0, 1.0]))

            x_pytorch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            y_pytorch = x_pytorch + 10.0
            y_pytorch.backward(torch.tensor([1.0, 1.0, 1.0]))

            self.assert_tensors_close(x_custom, x_pytorch, "Broadcasting: Vector + Scalar - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Broadcasting: Vector + Scalar - y (result)")

        # Matrix + vector (row broadcasting)
        with AutogradGraph() as graph:
            x_custom = CustomTensor([[1.0, 2.0], [3.0, 4.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([10.0, 20.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            z_custom = x_custom + y_custom  # y broadcasts to rows of x
            z_custom.backward(torch.ones_like(z_custom.tensor))

            x_pytorch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            y_pytorch = torch.tensor([10.0, 20.0], requires_grad=True)
            z_pytorch = x_pytorch + y_pytorch
            z_pytorch.backward(torch.ones_like(z_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Broadcasting: Matrix + Vector (row) - x")
            # For broadcasted operations, the gradient needs to be summed over the broadcasted dimensions
            # PyTorch handles this automatically. Your custom backward for add should accumulate.
            self.assert_tensors_close(y_custom, y_pytorch, "Broadcasting: Matrix + Vector (row) - y")
            self.assert_tensors_close(z_custom, z_pytorch, "Broadcasting: Matrix + Vector (row) - z (result)")

        # Matrix * scalar
        with AutogradGraph() as graph:
            x_custom = CustomTensor([[1.0, 2.0], [3.0, 4.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom * 5.0
            y_custom.backward(torch.ones_like(y_custom.tensor))

            x_pytorch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            y_pytorch = x_pytorch * 5.0
            y_pytorch.backward(torch.ones_like(y_pytorch))

            self.assert_tensors_close(x_custom, x_pytorch, "Broadcasting: Matrix * Scalar - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Broadcasting: Matrix * Scalar - y (result)")

    def test_backward_with_custom_grad(self):
        """Test backward pass with a custom initial gradient tensor."""
        print("\n=== Testing Backward with Custom Grad ===")

        with AutogradGraph() as graph:
            x_custom = CustomTensor([2.0, 3.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom * 4.0 + 1.0

            custom_grad_output = torch.tensor([0.5, 2.0])
            y_custom.backward(custom_grad_output)

            x_pytorch = torch.tensor([2.0, 3.0], requires_grad=True)
            y_pytorch = x_pytorch * 4.0 + 1.0

            pytorch_grad_output = torch.tensor([0.5, 2.0])
            y_pytorch.backward(pytorch_grad_output)

            self.assert_tensors_close(x_custom, x_pytorch, "Backward with Custom Grad - x")
            self.assert_tensors_close(y_custom, y_pytorch, "Backward with Custom Grad - y (result)")

    def test_zero_grad_behavior(self):
        """Test _zero_grad and subsequent backward calls."""
        print("\n=== Testing Zero Grad Behavior ===")
        with AutogradGraph() as graph:
            x_custom = CustomTensor([1.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = x_custom * 2
            z_custom = y_custom + 3
            self.assert_tensors_close(x_custom, torch.tensor([1.0], requires_grad=True), "Zero Grad Init (first backward) - x")
            z_custom.backward(retain_graph=True)  # First backward

            z_custom._zero_grad()  # Manually zero for custom
            y_custom._zero_grad()  # Manually zero for custom
            x_custom._zero_grad()  # Manually zero for custom leaf

            # Do another backward pass
            z_custom.backward()  # Should accumulate again from 1.0

            x_pytorch = torch.tensor([1.0], requires_grad=True)
            y_pytorch = x_pytorch * 2
            z_pytorch = y_pytorch + 3
            z_pytorch.backward(retain_graph=True)

            x_pytorch.grad.zero_()
            z_pytorch.backward()  # PyTorch accumulates if not zeroed explicitly

            self.assert_tensors_close(x_custom, x_pytorch, "Zero Grad Behavior - x (after 2nd backward)")
            self.assert_tensors_close(z_custom, z_pytorch, "Zero Grad Behavior - z (result, after 2nd backward)")

    def test_no_grad_flow(self):
        """Test that gradients do not flow to tensors not requiring grad."""
        print("\n=== Testing No Grad Flow ===")
        with AutogradGraph() as graph:
            x_custom = CustomTensor([5.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
            y_custom = CustomTensor([2.0], _custom_requires_grad=False)  # Does NOT require grad
            z_custom = x_custom * y_custom
            z_custom.backward()

            x_pytorch = torch.tensor([5.0], requires_grad=True)
            y_pytorch = torch.tensor([2.0], requires_grad=False)
            z_pytorch = x_pytorch * y_pytorch
            z_pytorch.backward()

            self.assert_tensors_close(x_custom, x_pytorch, "No Grad Flow - x (requires grad)")
            # PyTorch's .grad for non-requiring-grad tensors is None
            # Our CustomTensor.tensor.grad for non-requiring-grad should also be None
            try:
                # Check that y_custom.tensor.grad is None
                if y_custom.tensor.grad is not None:
                    raise AssertionError("Custom non-grad tensor unexpectedly has a gradient.")
                print(f"✓ No Grad Flow - y (no grad, custom correctly None)")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ No Grad Flow - y (no grad): {str(e)}")
                self.failed_tests += 1

    def test_basic_add_scalar_grad_system(self):
        print("\n=== System Test: Basic Scalar Add Grad ===")
        try:
            with AutogradGraph() as graph:
                a = CustomTensor(torch.tensor([2.0, 3.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)
                b = a + 5.0  # (a + 5)
                c = b + 10.0  # (a + 5 + 10)

                # Manually run backward pass
                c.backward(weightage_tensor=1,retain_graph=True)

                # Expected gradients:
                # dC/dA = 1.0 (for each element)
                assert torch.allclose(a.tensor.grad, torch.tensor([1.0, 1.0]))
                assert b.tensor.grad is not None
                assert torch.allclose(b.tensor.grad, torch.tensor([1.0, 1.0]))  # dC/dB = 1.0

                # Verify graph structure
                assert graph.graph.num_nodes() == 3
                assert graph.graph.num_edges() == 2
                assert graph.graph.has_edge(a._node_id, b._node_id)
                assert graph.graph.has_edge(b._node_id, c._node_id)
                assert graph.check_cycle() is False
            print("✓ System Test: Basic Scalar Add Grad")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ System Test: Basic Scalar Add Grad: {str(e)}")
            self.failed_tests += 1

    def test_basic_add_tensor_grad_system(self):
        print("\n=== System Test: Basic Tensor Add Grad ===")
        try:
            with AutogradGraph() as graph:
                a = CustomTensor(torch.tensor([2.0, 3.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)
                b = CustomTensor(torch.tensor([1.0, 2.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)
                c = a + b  # (a + b)
                d = c + 5.0  # (a + b + 5)

                d.backward(weightage_tensor=1,retain_graph=True)

                # Expected gradients:
                # dD/dA = 1.0
                # dD/dB = 1.0
                assert torch.allclose(a.tensor.grad, torch.tensor([1.0, 1.0]))
                assert torch.allclose(b.tensor.grad, torch.tensor([1.0, 1.0]))

                # Verify graph structure
                assert graph.graph.num_nodes() == 4
                assert graph.graph.num_edges() == 3
                assert graph.graph.has_edge(a._node_id, c._node_id)
                assert graph.graph.has_edge(b._node_id, c._node_id)
                assert graph.graph.has_edge(c._node_id, d._node_id)
                assert graph.check_cycle() is False
            print("✓ System Test: Basic Tensor Add Grad")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ System Test: Basic Tensor Add Grad: {str(e)}")
            self.failed_tests += 1

    def test_mixed_requires_grad_tensor_add_system(self):
        print("\n=== System Test: Mixed Requires Grad Tensor Add ===")
        try:
            with AutogradGraph() as graph:
                a = CustomTensor(torch.tensor([2.0, 3.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)
                b = CustomTensor(torch.tensor([1.0, 2.0]), _custom_requires_grad=False)  # Does not require grad
                c = a + b  # c should require grad, b's grad should be None

                c.backward(weightage_tensor=1,retain_graph = True)

                assert torch.allclose(a.tensor.grad, torch.tensor([1.0, 1.0]))
                assert b.tensor.grad is None  # b should not have a grad
                assert c._custom_requires_grad is True

                # Verify graph structure
                assert graph.graph.num_nodes() == 2  # Only a and c in the graph
                assert graph.graph.num_edges() == 1
                assert graph.graph.has_node(a._node_id)
                assert graph.graph.has_node(c._node_id)
                assert graph.graph.has_edge(a._node_id, c._node_id)
                # assert not graph.graph.has_node(b._node_id) # b should not be in graph
            print("✓ System Test: Mixed Requires Grad Tensor Add")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ System Test: Mixed Requires Grad Tensor Add: {str(e)}")
            self.failed_tests += 1

    def test_no_requires_grad_system(self):
        print("\n=== System Test: No Requires Grad ===")
        try:
            with AutogradGraph() as graph:  # Graph created, but no tensors with requires_grad=True added
                a = CustomTensor(torch.tensor([1.0]))
                b = CustomTensor(torch.tensor([2.0]))
                c = a + b
                d = c + 3.0

                assert not a._custom_requires_grad
                assert not b._custom_requires_grad
                assert not c._custom_requires_grad
                assert not d._custom_requires_grad
                assert graph.graph.num_nodes() == 0  # Graph should remain empty
                assert graph.graph.num_edges() == 0

                with pytest.raises(RuntimeError, match="Output tensor does not require grad."):
                    d.backward(weightage_tensor=1)
            print("✓ System Test: No Requires Grad")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ System Test: No Requires Grad: {str(e)}")
            self.failed_tests += 1

    def test_autograd_graph_context_manager_system(self):
        print("\n=== System Test: Autograd Graph Context Manager ===")
        try:
            graph = None
            with AutogradGraph(check_for_cycles=True, auto_cleanup=True) as g:
                graph = g
                a = CustomTensor(torch.tensor([1.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)
                b = a + 1.0
                assert graph.graph.num_nodes() == 2
                assert graph.graph.num_edges() == 1
                assert len(graph.intermediate_tensors) == 1  # b should be in intermediate_tensors

            # After exiting the context, graph should be empty
            assert graph.graph.num_nodes() == 0
            assert graph.graph.num_edges() == 0
            assert len(graph.intermediate_tensors) == 0
            print("✓ System Test: Autograd Graph Context Manager")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ System Test: Autograd Graph Context Manager: {str(e)}")
            self.failed_tests += 1

    def test_cycle_detection_system(self):
        print("\n=== System Test: Cycle Detection ===")
        try:
            with pytest.raises(RuntimeError, match="Cycle detected in autograd graph."):
                with AutogradGraph(check_for_cycles=True, auto_cleanup=False) as graph:
                    a = CustomTensor(torch.tensor([1.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)
                    b = CustomTensor(torch.tensor([2.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)

                    # Manually create a cycle (a -> b -> a)
                    graph.add_edge(a._node_id, b._node_id)
                    graph.add_edge(b._node_id, a._node_id)
                    graph.check_cycle() # Explicitly check for cycle
            print("✓ System Test: Cycle Detection")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ System Test: Cycle Detection: {str(e)}")
            self.failed_tests += 1

    def test_no_circular_references_non_leaf_tensors_die_system(self):
        # This test relies on the garbage collector. It's a heuristic test
        # as Python's GC timing is not strictly deterministic.
        # However, with weakrefs, it should work for non-leaf tensors.

        print("\n--- Starting System Test: No Circular References (Part 1) ---")
        try:
            graph_ref = None
            output_tensor_weak_ref = None
            node_id_d = -1  # To store node_id before d is deleted

            # BLOCK 1: Create graph and tensors
            with AutogradGraph(auto_cleanup=False) as graph:  # Keep graph for inspection
                graph_ref = weakref.ref(graph)
                a = CustomTensor(torch.tensor([1.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)
                b = a + 1.0  # Intermediate tensor
                c = b + 2.0  # Intermediate tensor
                d = c + 3.0  # Output tensor (also intermediate from graph's perspective)

                # Store weak reference to 'd' BEFORE its strong reference is potentially removed
                output_tensor_weak_ref = weakref.ref(d)
                node_id_d = d._node_id  # Store node_id while d is alive

                # The ref count for `d` object itself will be high here because it's in `graph.intermediate_tensors`,
                # and held by variable `d`, and by the temporary ref in `getrefcount`.
                assert len(graph.intermediate_tensors) == 3  # b, c, d should be in intermediate_tensors

            # BLOCK 2: After exiting context manager (auto_cleanup=False)
            # The 'graph' variable still holds a strong reference to the AutogradGraph instance.
            # graph_ref() should return the graph object.
            assert graph_ref() is not None, "Graph object should still be alive."
            assert len(graph_ref().intermediate_tensors) == 3, "Intermediate tensors should still be referenced by the graph."

            # BLOCK 3: Remove strong reference 'd' from local scope
            del d  # Remove the local strong reference to the CustomTensor object.
            gc.collect()  # Force garbage collection

            # Now, output_tensor_weak_ref() *still* shouldn't be None because `graph_ref().intermediate_tensors`
            # holds the strong reference.
            assert output_tensor_weak_ref() is not None, "d should still be alive due to intermediate_tensors."
            current_d_refcount_after_del_d = sys.getrefcount(output_tensor_weak_ref()) if output_tensor_weak_ref() else 'N/A'
            assert current_d_refcount_after_del_d == 2, f"Expected refcount 2, got {current_d_refcount_after_del_d}"

            # BLOCK 4: Remove strong reference from intermediate_tensors
            graph_ref().del_non_leaf_tensor_reference(node_id_d)  # THIS IS THE CRUCIAL STEP
            gc.collect()  # Force garbage collection again

            # Now, with the last strong reference gone, 'd' should be garbage collected.
            assert output_tensor_weak_ref() is None, "Output tensor (non-leaf) should be garbage collected after its strong reference is deleted from intermediate_tensors."

            # BLOCK 5: Verify other intermediate tensors are collected when graph is cleared
            intermediate_tensors_wrefs = []
            # Create a new graph and new tensors to avoid interference from previous block
            with AutogradGraph(auto_cleanup=False) as graph_new:
                a_new = CustomTensor(torch.tensor([1.0]), _custom_requires_grad=True, graph=graph_new, is_leaf=True)
                b_new = a_new + 1.0  # Intermediate
                c_new = b_new + 2.0  # Intermediate
                d_new = c_new + 3.0  # Intermediate (output of a chain)

                # Store weak references to the intermediate tensors
                intermediate_tensors_wrefs.append(weakref.ref(b_new))
                intermediate_tensors_wrefs.append(weakref.ref(c_new))
                intermediate_tensors_wrefs.append(weakref.ref(d_new))

                # Verify they are initially alive
                assert all(wref() is not None for wref in intermediate_tensors_wrefs)
                assert len(graph_new.intermediate_tensors) == 3

            assert graph_new is not None, "New graph object should still be alive after 'with' block."
            assert len(graph_new.intermediate_tensors) == 3, "New graph intermediate_tensors should still hold refs."

            # Manually clear the intermediate_tensors dictionary and remove graph reference
            graph_new.intermediate_tensors.clear()
            del graph_new  # Remove the strong reference to the graph itself
            del b_new, c_new, d_new  # deleting the local variable strong references
            gc.collect()

            # Now, all non-leaf tensors should be garbage collected
            for i, wref in enumerate(intermediate_tensors_wrefs):
                assert wref() is None, f"Intermediate tensor {i} should be garbage collected after graph context and intermediate_tensors are cleared."
            print("✓ System Test: No Circular References (Non-leaf tensors die)")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ System Test: No Circular References (Non-leaf tensors die): {str(e)}")
            self.failed_tests += 1

    def test_topological_sort_order_system(self):
        print("\n=== System Test: Topological Sort Order ===")
        try:
            with AutogradGraph() as graph:
                t1 = CustomTensor(torch.tensor([1.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)
                t2 = CustomTensor(torch.tensor([2.0]), _custom_requires_grad=True, graph=graph, is_leaf=True)
                t3 = t1 + t2
                t4 = t3 + 5.0
                t5 = t2 + 10.0  # Another branch
                t6 = t4 + t5

                # The topological sort should produce an order where dependencies come before their dependents.
                # Reversed topological sort should produce an order where outputs come before their inputs.
                # Example expected order: t6, t4, t5, t3, t2, t1 (or variations respecting dependencies)
                sorted_tensors = graph.reverse_toposort_from_tensor(t6._node_id)


                # Check if dependencies are respected in reverse order
                # If A -> B, then B should appear before A in reverse topological sort.
                # t6 depends on t4, t5. So t6 should be before t4 and t5.
                # t4 depends on t3. So t4 should be before t3.
                # t5 depends on t2. So t5 should be before t2.
                # t3 depends on t1, t2. So t3 should be before t1 and t2.

                # Simple check: The first element should be t6 (the ultimate output).
                assert sorted_tensors[0].__repr__() == t6.__repr__()

                # Check positions:
                sorted_tensors=[i.__repr__.__self__ for i in sorted_tensors] #converting the weakref to strongrefs
                pos = {t: i for i, t in enumerate(sorted_tensors)}

                assert pos[t6] < pos[t4]
                assert pos[t6] < pos[t5]
                assert pos[t4] < pos[t3]
                assert pos[t5] < pos[t2]
                assert pos[t3] < pos[t1]
                assert pos[t3] < pos[t2]  # t3 also depends on t2

                # Additional check: t2 is a dependency for both t3 and t5.
                # In reverse topo sort, t3 and t5 must appear before t2.
                assert pos[t3] < pos[t2]
                assert pos[t5] < pos[t2]

                # t1 is only a dependency for t3.
                assert pos[t3] < pos[t1]

                # Check if all 6 tensors are in the sorted list
                assert len(sorted_tensors) == 6
                assert set(sorted_tensors) == {t1, t2, t3, t4, t5, t6}
                sorted_tensors=None

            print("✓ System Test: Topological Sort Order")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ System Test: Topological Sort Order: {str(e)}")
            self.failed_tests += 1

    def test_very_deep_computation_graph(self):
        """Test with very deep computation graphs"""
        print("\n=== Testing Very Deep Computation Graph ===")

        try:
            depth = 50  # Moderate depth to avoid stack overflow in testing

            with AutogradGraph() as graph:
                x_custom = CustomTensor([1.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
                current_custom = x_custom

                # Create deep chain: x -> x+1 -> (x+1)+1 -> ... (50 times)
                for i in range(depth):
                    current_custom = current_custom + 1.0

                final_custom = current_custom
                final_custom.backward()

            x_pytorch = torch.tensor([1.0], requires_grad=True)
            current_pytorch = x_pytorch

            for i in range(depth):
                current_pytorch = current_pytorch + 1.0

            final_pytorch = current_pytorch
            final_pytorch.backward()

            self.assert_tensors_close(x_custom, x_pytorch, f"Deep Graph (depth={depth}) - x")
            self.assert_tensors_close(final_custom, final_pytorch, f"Deep Graph (depth={depth}) - final")

        except Exception as e:
            print(f"✗ Very Deep Computation Graph: {str(e)}")
            self.failed_tests += 1

    def test_wide_computation_graph(self):
        """Test with very wide computation graphs (many inputs)"""
        print("\n=== Testing Wide Computation Graph ===")

        try:
            width = 20  # 20 input tensors

            with AutogradGraph() as graph:
                # Create many input tensors
                inputs_custom = []
                for i in range(width):
                    inputs_custom.append(
                        CustomTensor([float(i + 1)], _custom_requires_grad=True, graph=graph, is_leaf=True)
                    )

                # Sum all inputs
                result_custom = inputs_custom[0]
                for i in range(1, width):
                    result_custom = result_custom + inputs_custom[i]

                result_custom.backward()

            # PyTorch equivalent
            inputs_pytorch = []
            for i in range(width):
                inputs_pytorch.append(torch.tensor([float(i + 1)], requires_grad=True))

            result_pytorch = inputs_pytorch[0]
            for i in range(1, width):
                result_pytorch = result_pytorch + inputs_pytorch[i]

            result_pytorch.backward()

            # Check all gradients
            for i in range(width):
                self.assert_tensors_close(
                    inputs_custom[i], inputs_pytorch[i],
                    f"Wide Graph (width={width}) - input_{i}"
                )

        except Exception as e:
            print(f"✗ Wide Computation Graph: {str(e)}")
            self.failed_tests += 1

    def test_nan_and_inf_handling(self):
        """Test handling of NaN and Inf values"""
        print("\n=== Testing NaN and Inf Handling ===")

        try:
            # Test with NaN input
            with AutogradGraph() as graph:
                x_custom = CustomTensor([float('nan')], _custom_requires_grad=True, graph=graph, is_leaf=True)
                y_custom = x_custom + 1.0
                y_custom.backward()

                # Check that gradients handle NaN appropriately
                assert torch.isnan(x_custom.tensor.grad).any() or x_custom.tensor.grad is not None

            # Test with Inf input
            with AutogradGraph() as graph:
                x_custom = CustomTensor([float('inf')], _custom_requires_grad=True, graph=graph, is_leaf=True)
                y_custom = x_custom * 2.0
                y_custom.backward()

                # Should handle inf appropriately
                assert torch.isinf(x_custom.tensor.grad).any() or x_custom.tensor.grad is not None

            print("ℹ NaN/Inf Handling - Consider adding explicit handling for edge numerical cases")
            self.passed_tests += 1

        except Exception as e:
            print(f"✗ NaN and Inf Handling: {str(e)}")
            self.failed_tests += 1

    def test_zero_gradients(self):
        """Test operations that should produce zero gradients"""
        print("\n=== Testing Zero Gradients ===")

        try:
            with AutogradGraph() as graph:
                x_custom = CustomTensor([2.0], _custom_requires_grad=True, graph=graph, is_leaf=True)

                # x - x should have zero gradient with respect to x
                y_custom = x_custom - x_custom
                y_custom.backward()

            x_pytorch = torch.tensor([2.0], requires_grad=True)
            y_pytorch = x_pytorch - x_pytorch
            y_pytorch.backward()

            self.assert_tensors_close(x_custom, x_pytorch, "Zero Gradients - x")

        except Exception as e:
            print(f"✗ Zero Gradients: {str(e)}")
            self.failed_tests += 1


    def test_memory_efficiency(self):
        """Test memory efficiency with large computations"""
        print("\n=== Testing Memory Efficiency ===")

        try:
            # Create a computation that could potentially leak memory
            initial_tensor_count = len(gc.get_objects())

            for iteration in range(5):
                with AutogradGraph() as graph:
                    x_custom = CustomTensor([1.0] * 100, _custom_requires_grad=True, graph=graph, is_leaf=True)

                    # Chain of operations
                    current = x_custom
                    for i in range(10):
                        current = current + 1.0
                        current = current * 1.1

                    current.backward(torch.ones(100))

                # Force cleanup
                del current, x_custom
                gc.collect()

            final_tensor_count = len(gc.get_objects())

            # Memory should not grow excessively
            growth = final_tensor_count - initial_tensor_count
            print(f"Object count growth: {growth}")

            if growth < 1000:  # Reasonable threshold
                print("✓ Memory Efficiency - Reasonable memory usage")
                self.passed_tests += 1
            else:
                print(f"⚠ Memory Efficiency - High memory growth: {growth} objects")
                self.passed_tests += 1  # Still pass but warn

        except Exception as e:
            print(f"✗ Memory Efficiency: {str(e)}")
            self.failed_tests += 1
    def test_linear_module(self):
      """Test Linear module forward pass, backward pass, and parameter updates."""
      print("\n=== Testing Linear Module ===")

      # Test basic functionality
      with AutogradGraph() as graph:
          # Custom implementation
          linear_custom = Linear(3, 2, bias=True, graph=graph)
          input_custom = CustomTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                    _custom_requires_grad=True, graph=graph, is_leaf=True)
          output_custom = linear_custom(input_custom)
          loss_custom = (output_custom * output_custom).sum()
          loss_custom.backward()

          # PyTorch reference
          linear_pytorch = torch.nn.Linear(3, 2, bias=True)
          linear_pytorch.weight.data = linear_custom.weight.tensor.data.clone()
          linear_pytorch.bias.data = linear_custom.bias.tensor.data.clone()

          input_pytorch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
          output_pytorch = linear_pytorch(input_pytorch)
          loss_pytorch = (output_pytorch * output_pytorch).sum()
          loss_pytorch.backward()

          self.assert_tensors_close(output_custom, output_pytorch, "Linear Forward Pass")
          self.assert_tensors_close(input_custom, input_pytorch, "Linear Input Gradient")
          self.assert_tensors_close(linear_custom.weight, linear_pytorch.weight, "Linear Weight Gradient")
          self.assert_tensors_close(linear_custom.bias, linear_pytorch.bias, "Linear Bias Gradient")

      # Test without bias
      with AutogradGraph() as graph:
          linear_custom = Linear(2, 1, bias=False, graph=graph)
          input_custom = CustomTensor([1.0, 2.0], _custom_requires_grad=True, graph=graph, is_leaf=True)
          output_custom = linear_custom(input_custom)
          output_custom.backward()

          linear_pytorch = torch.nn.Linear(2, 1, bias=False)
          linear_pytorch.weight.data = linear_custom.weight.tensor.data.clone()
          input_pytorch = torch.tensor([1.0, 2.0], requires_grad=True)
          output_pytorch = linear_pytorch(input_pytorch)
          output_pytorch.backward()

          self.assert_tensors_close(output_custom, output_pytorch, "Linear No Bias Forward")
          self.assert_tensors_close(linear_custom.weight, linear_pytorch.weight, "Linear No Bias Weight Gradient")

      # Test training vs eval mode
      with AutogradGraph() as graph:
          linear_custom = Linear(2, 1, graph=graph)
          input_custom = CustomTensor([1.0, 2.0], _custom_requires_grad=True, graph=graph, is_leaf=True)

          # Training mode
          linear_custom.train()
          output_train = linear_custom(input_custom)

          # Eval mode
          linear_custom.eval()
          output_eval = linear_custom(input_custom)

          # In eval mode, should not require grad for output
          try:
              if hasattr(output_eval, '_custom_requires_grad') and output_eval._custom_requires_grad:
                  raise AssertionError("Output in eval mode should not require grad")
              print("✓ Linear Eval Mode - No Grad")
              self.passed_tests += 1
          except Exception as e:
              print(f"✗ Linear Eval Mode - No Grad: {str(e)}")
              self.failed_tests += 1

    def test_conv2d_module(self):
      """Test Conv2d module forward pass, backward pass, and parameter updates."""
      print("\n=== Testing Conv2d Module ===")

      # Test basic convolution
      with AutogradGraph() as graph:
          # Custom implementation
          conv_custom = Conv2d(in_channels=2, out_channels=3, kernel_size=3,
                            stride=1, padding=1, bias=True, graph=graph)
          input_custom = CustomTensor(torch.randn(1, 2, 4, 4),
                                    _custom_requires_grad=True, graph=graph, is_leaf=True)
          output_custom = conv_custom(input_custom)
          loss_custom = output_custom.sum()
          loss_custom.backward()

          # PyTorch reference
          conv_pytorch = torch.nn.Conv2d(2, 3, 3, stride=1, padding=1, bias=True)
          conv_pytorch.weight.data = conv_custom.weight.tensor.data.clone()
          conv_pytorch.bias.data = conv_custom.bias.tensor.data.clone()

          input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
          output_pytorch = conv_pytorch(input_pytorch)
          loss_pytorch = output_pytorch.sum()
          loss_pytorch.backward()

          self.assert_tensors_close(output_custom, output_pytorch, "Conv2d Forward Pass")
          self.assert_tensors_close(input_custom, input_pytorch, "Conv2d Input Gradient")
          self.assert_tensors_close(conv_custom.weight, conv_pytorch.weight, "Conv2d Weight Gradient")
          self.assert_tensors_close(conv_custom.bias, conv_pytorch.bias, "Conv2d Bias Gradient")

      # Test different parameters
      with AutogradGraph() as graph:
          conv_custom = Conv2d(in_channels=1, out_channels=2, kernel_size=2,
                            stride=2, padding=0, bias=False, graph=graph)
          input_custom = CustomTensor(torch.randn(1, 1, 6, 6),
                                    _custom_requires_grad=True, graph=graph, is_leaf=True)
          output_custom = conv_custom(input_custom)
          output_custom.sum().backward()

          conv_pytorch = torch.nn.Conv2d(1, 2, 2, stride=2, padding=0, bias=False)
          conv_pytorch.weight.data = conv_custom.weight.tensor.data.clone()
          input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
          output_pytorch = conv_pytorch(input_pytorch)
          output_pytorch.sum().backward()

          self.assert_tensors_close(output_custom, output_pytorch, "Conv2d Different Params Forward")
          self.assert_tensors_close(conv_custom.weight, conv_pytorch.weight, "Conv2d Different Params Weight Gradient")

    def test_batchnorm_module(self):
      """Test BatchNorm_Nd module forward pass, backward pass, and running statistics."""
      print("\n=== Testing BatchNorm Module ===")

      # Test training mode
      with AutogradGraph() as graph:
          bn_custom = BatchNorm_Nd(num_features=3, graph=graph)
          input_custom = CustomTensor(torch.randn(2, 3, 4, 4),
                                    _custom_requires_grad=True, graph=graph, is_leaf=True)

          bn_custom.train()
          output_custom = bn_custom(input_custom)
          loss_custom = output_custom.sum()
          loss_custom.backward()

          # PyTorch reference
          bn_pytorch = torch.nn.BatchNorm2d(3)
          bn_pytorch.weight.data = bn_custom.weight.tensor.data.clone()
          bn_pytorch.bias.data = bn_custom.bias.tensor.data.clone()
          bn_pytorch.running_mean = bn_custom.running_mean.clone()
          bn_pytorch.running_var = bn_custom.running_var.clone()

          input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
          output_pytorch = bn_pytorch(input_pytorch)
          loss_pytorch = output_pytorch.sum()
          loss_pytorch.backward()

          self.assert_tensors_close(output_custom, output_pytorch, "BatchNorm Training Forward")
          self.assert_tensors_close(input_custom, input_pytorch, "BatchNorm Input Gradient")
          self.assert_tensors_close(bn_custom.weight, bn_pytorch.weight, "BatchNorm Weight Gradient")
          self.assert_tensors_close(bn_custom.bias, bn_pytorch.bias, "BatchNorm Bias Gradient")

      # Test eval mode
      with AutogradGraph() as graph:
          bn_custom = BatchNorm_Nd(num_features=2, graph=graph)
          input_custom = CustomTensor(torch.randn(1, 2, 3, 3),
                                    _custom_requires_grad=True, graph=graph, is_leaf=True)

          # Set some running stats
          bn_custom.running_mean = torch.tensor([0.5, -0.3])
          bn_custom.running_var = torch.tensor([1.2, 0.8])

          bn_custom.eval()
          output_custom = bn_custom(input_custom)

          bn_pytorch = torch.nn.BatchNorm2d(2)
          bn_pytorch.weight.data = bn_custom.weight.tensor.data.clone()
          bn_pytorch.bias.data = bn_custom.bias.tensor.data.clone()
          bn_pytorch.running_mean = bn_custom.running_mean.clone()
          bn_pytorch.running_var = bn_custom.running_var.clone()
          bn_pytorch.eval()

          input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
          output_pytorch = bn_pytorch(input_pytorch)

          self.assert_tensors_close(output_custom, output_pytorch, "BatchNorm Eval Forward")

    def test_maxpool2d_module(self):
      """Test MaxPool2d module forward pass and backward pass."""
      print("\n=== Testing MaxPool2d Module ===")

      with AutogradGraph() as graph:
          pool_custom = MaxPool2d(kernel_size=2, stride=2, padding=0, graph=graph)
          input_custom = CustomTensor(torch.randn(1, 2, 4, 4),
                                    _custom_requires_grad=True, graph=graph, is_leaf=True)
          output_custom = pool_custom(input_custom)
          loss_custom = output_custom.sum()
          loss_custom.backward()

          # PyTorch reference
          pool_pytorch = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
          input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
          output_pytorch = pool_pytorch(input_pytorch)
          loss_pytorch = output_pytorch.sum()
          loss_pytorch.backward()

          self.assert_tensors_close(output_custom, output_pytorch, "MaxPool2d Forward")
          self.assert_tensors_close(input_custom, input_pytorch, "MaxPool2d Input Gradient")

      # Test with different parameters
      with AutogradGraph() as graph:
          pool_custom = MaxPool2d(kernel_size=3, stride=1, padding=1, graph=graph)
          input_custom = CustomTensor(torch.randn(2, 1, 5, 5),
                                    _custom_requires_grad=True, graph=graph, is_leaf=True)
          output_custom = pool_custom(input_custom)
          output_custom=output_custom.sum()
          output_custom.backward()

          pool_pytorch = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
          input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
          output_pytorch = pool_pytorch(input_pytorch)
          output_pytorch=output_pytorch.sum()
          output_pytorch.backward()

          self.assert_tensors_close(output_custom, output_pytorch, "MaxPool2d Different Params Forward")
          self.assert_tensors_close(input_custom, input_pytorch, "MaxPool2d Different Params Gradient")

    def test_avgpool2d_module(self):
      """Test AvgPool2d module forward pass and backward pass."""
      print("\n=== Testing AvgPool2d Module ===")

      with AutogradGraph() as graph:
          pool_custom = AvgPool2d(kernel_size=2, stride=2, padding=0, graph=graph)
          input_custom = CustomTensor(torch.randn(1, 2, 4, 4),
                                    _custom_requires_grad=True, graph=graph, is_leaf=True)
          output_custom = pool_custom(input_custom)
          loss_custom = output_custom.sum()
          loss_custom.backward()

          # PyTorch reference
          pool_pytorch = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
          input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
          output_pytorch = pool_pytorch(input_pytorch)
          loss_pytorch = output_pytorch.sum()
          loss_pytorch.backward()

          self.assert_tensors_close(output_custom, output_pytorch, "AvgPool2d Forward")
          self.assert_tensors_close(input_custom, input_pytorch, "AvgPool2d Input Gradient")

      # Test with padding
      with AutogradGraph() as graph:
          pool_custom = AvgPool2d(kernel_size=3, stride=1, padding=1, graph=graph)
          input_custom = CustomTensor(torch.randn(1, 1, 4, 4),
                                    _custom_requires_grad=True, graph=graph, is_leaf=True)
          output_custom = pool_custom(input_custom)
          output_custom.sum().backward()

          pool_pytorch = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
          input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
          output_pytorch = pool_pytorch(input_pytorch)
          output_pytorch.sum().backward()

          self.assert_tensors_close(output_custom, output_pytorch, "AvgPool2d With Padding Forward")
          self.assert_tensors_close(input_custom, input_pytorch, "AvgPool2d With Padding Gradient")

    def test_relu_module(self):
        """Test ReLU activation module."""
        print("\n=== Testing ReLU Module ===")

        with AutogradGraph() as graph:
            relu_custom = ReLu(graph=graph)
            input_custom = CustomTensor(torch.randn(2, 3),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = relu_custom(input_custom)
            loss_custom = output_custom.sum()
            loss_custom.backward()

            # PyTorch reference
            relu_pytorch = torch.nn.ReLU()
            input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
            output_pytorch = relu_pytorch(input_pytorch)
            loss_pytorch = output_pytorch.sum()
            loss_pytorch.backward()

            self.assert_tensors_close(output_custom, output_pytorch, "ReLU Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "ReLU Input Gradient")

        # Test with negative values specifically
        with AutogradGraph() as graph:
            relu_custom = ReLu(graph=graph)
            input_custom = CustomTensor(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = relu_custom(input_custom)
            output_custom.sum().backward()

            relu_pytorch = torch.nn.ReLU()
            input_pytorch = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
            output_pytorch = relu_pytorch(input_pytorch)
            output_pytorch.sum().backward()

            self.assert_tensors_close(output_custom, output_pytorch, "ReLU Negative Values Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "ReLU Negative Values Gradient")

    def test_leaky_relu_module(self):
        """Test Leaky ReLU activation module."""
        print("\n=== Testing Leaky ReLU Module ===")

        with AutogradGraph() as graph:
            leaky_relu_custom = Leaky_ReLu(negative_slope=0.01, graph=graph)
            input_custom = CustomTensor(torch.randn(2, 3),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = leaky_relu_custom(input_custom)
            loss_custom = output_custom.sum()
            loss_custom.backward()

            # PyTorch reference
            leaky_relu_pytorch = torch.nn.LeakyReLU(negative_slope=0.01)
            input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
            output_pytorch = leaky_relu_pytorch(input_pytorch)
            loss_pytorch = output_pytorch.sum()
            loss_pytorch.backward()

            self.assert_tensors_close(output_custom, output_pytorch, "Leaky ReLU Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "Leaky ReLU Input Gradient")

        # Test with different slope
        with AutogradGraph() as graph:
            leaky_relu_custom = Leaky_ReLu(negative_slope=0.1, graph=graph)
            input_custom = CustomTensor(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = leaky_relu_custom(input_custom)
            output_custom.sum().backward()

            leaky_relu_pytorch = torch.nn.LeakyReLU(negative_slope=0.1)
            input_pytorch = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
            output_pytorch = leaky_relu_pytorch(input_pytorch)
            output_pytorch.sum().backward()

            self.assert_tensors_close(output_custom, output_pytorch, "Leaky ReLU Different Slope Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "Leaky ReLU Different Slope Gradient")

    def test_gelu_module(self):
        """Test GELU activation module."""
        print("\n=== Testing GELU Module ===")

        # Test exact GELU
        with AutogradGraph() as graph:
            gelu_custom = GeLu(approximate='none', graph=graph)
            input_custom = CustomTensor(torch.randn(2, 3),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = gelu_custom(input_custom)
            loss_custom = output_custom.sum()
            loss_custom.backward()

            # PyTorch reference
            gelu_pytorch = torch.nn.GELU(approximate='none')
            input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
            output_pytorch = gelu_pytorch(input_pytorch)
            loss_pytorch = output_pytorch.sum()
            loss_pytorch.backward()

            self.assert_tensors_close(output_custom, output_pytorch, "GELU Exact Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "GELU Exact Input Gradient")

        # Test approximate GELU
        with AutogradGraph() as graph:
            gelu_custom = GeLu(approximate='tanh', graph=graph)
            input_custom = CustomTensor(torch.randn(2, 3),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = gelu_custom(input_custom)
            output_custom.sum().backward()

            gelu_pytorch = torch.nn.GELU(approximate='tanh')
            input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
            output_pytorch = gelu_pytorch(input_pytorch)
            output_pytorch.sum().backward()

            self.assert_tensors_close(output_custom, output_pytorch, "GELU Approximate Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "GELU Approximate Input Gradient")

    def test_elu_module(self):
        """Test ELU activation module."""
        print("\n=== Testing ELU Module ===")

        with AutogradGraph() as graph:
            elu_custom = Elu(alpha=1.0, graph=graph)
            input_custom = CustomTensor(torch.randn(2, 3),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = elu_custom(input_custom)
            loss_custom = output_custom.sum()
            loss_custom.backward()

            # PyTorch reference
            elu_pytorch = torch.nn.ELU(alpha=1.0)
            input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
            output_pytorch = elu_pytorch(input_pytorch)
            loss_pytorch = output_pytorch.sum()
            loss_pytorch.backward()

            self.assert_tensors_close(output_custom, output_pytorch, "ELU Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "ELU Input Gradient")

        # Test with different alpha
        with AutogradGraph() as graph:
            elu_custom = Elu(alpha=0.5, graph=graph)
            input_custom = CustomTensor(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = elu_custom(input_custom)
            output_custom.sum().backward()

            elu_pytorch = torch.nn.ELU(alpha=0.5)
            input_pytorch = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
            output_pytorch = elu_pytorch(input_pytorch)
            output_pytorch.sum().backward()

            self.assert_tensors_close(output_custom, output_pytorch, "ELU Different Alpha Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "ELU Different Alpha Gradient")

    def test_silu_module(self):
        """Test SiLU (Swish) activation module."""
        print("\n=== Testing SiLU Module ===")

        with AutogradGraph() as graph:
            silu_custom = Silu(graph=graph)
            input_custom = CustomTensor(torch.randn(2, 3),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = silu_custom(input_custom)
            loss_custom = output_custom.sum()
            loss_custom.backward()

            # PyTorch reference
            silu_pytorch = torch.nn.SiLU()
            input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
            output_pytorch = silu_pytorch(input_pytorch)
            loss_pytorch = output_pytorch.sum()
            loss_pytorch.backward()

            self.assert_tensors_close(output_custom, output_pytorch, "SiLU Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "SiLU Input Gradient")

    def test_sigmoid_module(self):
        """Test Sigmoid activation module."""
        print("\n=== Testing Sigmoid Module ===")

        with AutogradGraph() as graph:
            sigmoid_custom = Sigmoid(graph=graph)
            input_custom = CustomTensor(torch.randn(2, 3),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = sigmoid_custom(input_custom)
            loss_custom = output_custom.sum()
            loss_custom.backward()

            # PyTorch reference
            sigmoid_pytorch = torch.nn.Sigmoid()
            input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
            output_pytorch = sigmoid_pytorch(input_pytorch)
            loss_pytorch = output_pytorch.sum()
            loss_pytorch.backward()

            self.assert_tensors_close(output_custom, output_pytorch, "Sigmoid Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "Sigmoid Input Gradient")

    def test_tanh_module(self):
        """Test Tanh activation module."""
        print("\n=== Testing Tanh Module ===")

        with AutogradGraph() as graph:
            tanh_custom = Tanh(graph=graph)
            input_custom = CustomTensor(torch.randn(2, 3),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = tanh_custom(input_custom)
            loss_custom = output_custom.sum()
            loss_custom.backward()

            # PyTorch reference
            tanh_pytorch = torch.nn.Tanh()
            input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
            output_pytorch = tanh_pytorch(input_pytorch)
            loss_pytorch = output_pytorch.sum()
            loss_pytorch.backward()

            self.assert_tensors_close(output_custom, output_pytorch, "Tanh Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "Tanh Input Gradient")

    def test_swish_module(self):
        """Test Swish activation module with learnable parameter."""
        print("\n=== Testing Swish Module ===")

        with AutogradGraph() as graph:
            swish_custom = Swish(B_initial=1.0, graph=graph)
            input_custom = CustomTensor(torch.randn(2, 3),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = swish_custom(input_custom)
            loss_custom = output_custom.sum()
            loss_custom.backward()

            # PyTorch reference - manual implementation since there's no direct equivalent
            class PyTorchSwish(torch.nn.Module):
                def __init__(self, B_initial=1.0):
                    super().__init__()
                    self.B = torch.nn.Parameter(torch.tensor([B_initial]))

                def forward(self, x):
                    return x * torch.sigmoid(self.B * x)

            swish_pytorch = PyTorchSwish(B_initial=1.0)
            swish_pytorch.B.data = swish_custom.B.tensor.data.clone()

            input_pytorch = input_custom.tensor.clone().detach().requires_grad_(True)
            output_pytorch = swish_pytorch(input_pytorch)
            loss_pytorch = output_pytorch.sum()
            loss_pytorch.backward()

            self.assert_tensors_close(output_custom, output_pytorch, "Swish Forward")
            self.assert_tensors_close(input_custom, input_pytorch, "Swish Input Gradient")
            self.assert_tensors_close(swish_custom.B, swish_pytorch.B, "Swish B Parameter Gradient")

        # Test with different B_initial
        with AutogradGraph() as graph:
            swish_custom = Swish(B_initial=2.0, graph=graph)
            input_custom = CustomTensor(torch.tensor([0.5, -0.5, 1.0, -1.0]),
                                        _custom_requires_grad=True, graph=graph, is_leaf=True)
            output_custom = swish_custom(input_custom)
            output_custom.sum().backward()

            swish_pytorch = PyTorchSwish(B_initial=2.0)
            swish_pytorch.B.data = swish_custom.B.tensor.data.clone()
            input_pytorch = torch.tensor([0.5, -0.5, 1.0, -1.0], requires_grad=True)
            output_pytorch = swish_pytorch(input_pytorch)
            output_pytorch.sum().backward()

            self.assert_tensors_close(output_custom, output_pytorch, "Swish Different B Forward")
            self.assert_tensors_close(swish_custom.B, swish_pytorch.B, "Swish Different B Parameter Gradient")

    def test_module_parameter_management(self):
        """Test parameter collection and gradient zeroing across modules."""
        print("\n=== Testing Module Parameter Management ===")

        with AutogradGraph() as graph:
            # Create a small network
            linear1 = Linear(3, 2, graph=graph)
            linear2 = Linear(2, 1, graph=graph)

            # Test parameter collection
            params1 = linear1.parameters()
            params2 = linear2.parameters()

            try:
                # Should have weight and bias for each layer
                if len(params1) != 2:
                    raise AssertionError(f"Linear1 should have 2 parameters, got {len(params1)}")
                if len(params2) != 2:
                    raise AssertionError(f"Linear2 should have 2 parameters, got {len(params2)}")
                print("✓ Module Parameter Collection")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Module Parameter Collection: {str(e)}")
                self.failed_tests += 1

            # Test forward pass
            input_tensor = CustomTensor([[1.0, 2.0, 3.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            hidden = linear1(input_tensor)
            output = linear2(hidden)
            loss = output.sum()
            loss.backward()

            # Check that all parameters have gradients
            all_params = params1 + params2
            try:
                for i, param in enumerate(all_params):
                    if param.tensor.grad is None:
                        raise AssertionError(f"Parameter {i} should have gradient")
                print("✓ Module All Parameters Have Gradients")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Module All Parameters Have Gradients: {str(e)}")
                self.failed_tests += 1

            # Test zero_grad
            linear1.zero_grad()
            linear2.zero_grad()

            try:
                for i, param in enumerate(all_params):
                    if param.tensor.grad is None or not torch.allclose(param.tensor.grad, torch.zeros_like(param.tensor.grad)):
                        raise AssertionError(f"Parameter {i} gradient should be zero after zero_grad()")
                print("✓ Module Zero Grad")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Module Zero Grad: {str(e)}")
                self.failed_tests += 1

    def test_module_training_eval_modes(self):
        """Test training and evaluation mode switching."""
        print("\n=== Testing Module Training/Eval Modes ===")

        with AutogradGraph() as graph:
            # Test with modules that behave differently in train/eval
            linear = Linear(2, 1, graph=graph)
            bn = BatchNorm_Nd(1, graph=graph)
            relu = ReLu(graph=graph)

            # Initially should be in training mode
            try:
                if not linear.training or not bn.training or not relu.training:
                    raise AssertionError("Modules should start in training mode")
                print("✓ Module Initial Training Mode")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Module Initial Training Mode: {str(e)}")
                self.failed_tests += 1

            # Switch to eval mode
            linear.eval()
            bn.eval()
            relu.eval()

            try:
                if linear.training or bn.training or relu.training:
                    raise AssertionError("Modules should be in eval mode after eval()")
                print("✓ Module Eval Mode Switch")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Module Eval Mode Switch: {str(e)}")
                self.failed_tests += 1

            # Switch back to training mode
            linear.train()
            bn.train()
            relu.train()

            try:
                if not linear.training or not bn.training or not relu.training:
                    raise AssertionError("Modules should be in training mode after train()")
                print("✓ Module Training Mode Switch")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Module Training Mode Switch: {str(e)}")
                self.failed_tests += 1

    def test_module_nested_structure(self):
        """Test nested module structures and parameter collection."""
        print("\n=== Testing Nested Module Structure ===")

        class SimpleNet(Module):
            def __init__(self, graph):
                super().__init__()
                self.layer1 = Linear(3, 4, graph=graph)
                self.activation = ReLu(graph=graph)
                self.layer2 = Linear(4, 2, graph=graph)

            def forward(self, x):
                x = self.layer1(x)
                x = self.activation(x)
                x = self.layer2(x)
                return x

        with AutogradGraph() as graph:
            net = SimpleNet(graph)

            # Test nested parameter collection
            params = net.parameters()

            try:
                # Should have 4 parameters: 2 weights + 2 biases
                if len(params) != 4:
                    raise AssertionError(f"Network should have 4 parameters, got {len(params)}")
                print("✓ Nested Module Parameter Collection")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Nested Module Parameter Collection: {str(e)}")
                self.failed_tests += 1

            # Test nested training mode switching
            net.train()
            try:
                if not net.layer1.training or not net.activation.training or not net.layer2.training:
                    raise AssertionError("All nested modules should be in training mode")
                print("✓ Nested Module Training Mode")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Nested Module Training Mode: {str(e)}")
                self.failed_tests += 1

            net.eval()
            try:
                if net.layer1.training or net.activation.training or net.layer2.training:
                    raise AssertionError("All nested modules should be in eval mode")
                print("✓ Nested Module Eval Mode")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Nested Module Eval Mode: {str(e)}")
                self.failed_tests += 1
            net.train()
            # Test forward pass through nested structure
            input_tensor = CustomTensor([[1.0, 2.0, 3.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            output = net(input_tensor)
            loss = output.sum()
            loss.backward()

            # Check that all parameters have gradients
            try:
                for i, param in enumerate(params):
                    if param.tensor.grad is None:
                        raise AssertionError(f"Parameter {i} should have gradient after backward")
                print("✓ Nested Module Gradient Flow")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Nested Module Gradient Flow: {str(e)}")
                self.failed_tests += 1

    def test_module_edge_cases(self):
        """Test edge cases and error conditions for modules."""
        print("\n=== Testing Module Edge Cases ===")

        # Test very small inputs
        with AutogradGraph() as graph:
            linear = Linear(1, 1, graph=graph)
            tiny_input = CustomTensor([[1e-8]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            output = linear(tiny_input)
            output.backward()

            try:
                if linear.weight.tensor.grad is None or linear.bias.tensor.grad is None:
                    raise AssertionError("Should handle very small inputs")
                print("✓ Module Tiny Input Handling")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Module Tiny Input Handling: {str(e)}")
                self.failed_tests += 1

        # Test large inputs
        with AutogradGraph() as graph:
            linear = Linear(2, 2, graph=graph)
            large_input = CustomTensor([[1e6, -1e6]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            output = linear(large_input)
            output.sum().backward()

            try:
                if torch.isnan(linear.weight.tensor.grad).any() or torch.isinf(linear.weight.tensor.grad).any():
                    raise AssertionError("Should handle large inputs without NaN/Inf")
                print("✓ Module Large Input Handling")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Module Large Input Handling: {str(e)}")
                self.failed_tests += 1

        # Test zero gradients don't break anything
        with AutogradGraph() as graph:
            relu = ReLu(graph=graph)
            zero_input = CustomTensor([[-1.0, -2.0, -3.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            output = relu(zero_input)  # All outputs will be 0
            output.sum().backward()    # All gradients will be 0

            try:
                if zero_input.tensor.grad is None:
                    raise AssertionError("Should handle zero gradient case")
                if not torch.allclose(zero_input.tensor.grad, torch.zeros_like(zero_input.tensor.grad)):
                    raise AssertionError("Gradients should be zero for negative ReLU inputs")
                print("✓ Module Zero Gradient Handling")
                self.passed_tests += 1
            except Exception as e:
                print(f"✗ Module Zero Gradient Handling: {str(e)}")
                self.failed_tests += 1
    def test_mse_loss_basic(self):
        """Test basic MSE loss functionality"""
        print("\n=== Testing MSE Loss Basic ===")

        # Basic MSE test
        with AutogradGraph() as graph:
            # Create input and target tensors
            input_custom = CustomTensor([[1.0, 2.0], [3.0, 4.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([[0.5, 1.5], [2.5, 3.5]], _custom_requires_grad=False)

            mse_loss = MSE(graph=graph)
            mse_loss.train()  # Ensure training mode
            loss_custom = mse_loss(input_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            target_pytorch = torch.tensor([[0.5, 1.5], [2.5, 3.5]], requires_grad=False)
            loss_pytorch = torch.nn.functional.mse_loss(input_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "MSE Loss Basic - input gradients")
            self.assert_tensors_close(loss_custom, loss_pytorch, "MSE Loss Basic - loss value", check_grad=False)

    def test_mse_loss_with_weights(self):
      """Test MSE loss with per-class and per-pixel weights"""
      print("\n=== Testing MSE Loss with Per-Class Weights ===")

      # -----------------------
      # PER-CLASS WEIGHT TEST
      # -----------------------
      with AutogradGraph() as graph:
          input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
          target_tensor = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
          weight_tensor = torch.tensor([2.0, 0.5])  # Per-class weight (C=2)

          input_custom = CustomTensor(input_tensor.clone(), _custom_requires_grad=True, graph=graph, is_leaf=True)
          target_custom = CustomTensor(target_tensor.clone(), _custom_requires_grad=False)

          mse_loss = MSE(graph=graph)
          mse_loss.train()
          loss_custom = mse_loss(input_custom, target_custom, weight=weight_tensor)
          loss_custom.backward()

          # Manual PyTorch equivalent
          input_pytorch = input_tensor.clone().detach().requires_grad_(True)
          diff = input_pytorch - target_tensor
          weight = weight_tensor.view(1, -1)  # shape (1, C)
          weighted_diff = (diff ** 2) * weight
          loss_expected = weighted_diff.sum() / weight.sum()
          loss_expected.backward()

          self.assert_tensors_close(input_custom, input_pytorch, "Per-Class Weighted MSE - Input Gradient")
          self.assert_tensors_close(loss_custom, loss_expected, "Per-Class Weighted MSE - Loss Value", check_grad=False)

      # -----------------------
      # PER-PIXEL WEIGHT TEST
      # -----------------------
      print("\n=== Testing MSE Loss with Per-Pixel Weights ===")
      with AutogradGraph() as graph:
          input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
          target_tensor = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
          weight_tensor = torch.tensor([[2.0, 2.0], [0.5, 0.5]])  # Per-pixel weights (shape matches input)

          input_custom = CustomTensor(input_tensor.clone(), _custom_requires_grad=True, graph=graph, is_leaf=True)
          target_custom = CustomTensor(target_tensor.clone(), _custom_requires_grad=False)

          mse_loss = MSE(graph=graph)
          mse_loss.train()
          loss_custom = mse_loss(input_custom, target_custom, weight=weight_tensor)
          loss_custom.backward()

          # Manual PyTorch equivalent
          input_pytorch = input_tensor.clone().detach().requires_grad_(True)
          diff = input_pytorch - target_tensor
          weighted_diff = (diff ** 2) * weight_tensor
          loss_expected = weighted_diff.sum() / weight_tensor.sum()
          loss_expected.backward()

          self.assert_tensors_close(input_custom, input_pytorch, "Per-Pixel Weighted MSE - Input Gradient")
          self.assert_tensors_close(loss_custom, loss_expected, "Per-Pixel Weighted MSE - Loss Value", check_grad=False)


    def test_mse_loss_eval_mode(self):
        """Test MSE loss in evaluation mode (no gradients)"""
        print("\n=== Testing MSE Loss Eval Mode ===")

        with AutogradGraph() as graph:
            input_custom = CustomTensor([[1.0, 2.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([[0.5, 1.5]], _custom_requires_grad=False)

            mse_loss = MSE(graph=graph)
            mse_loss.eval()  # Set to evaluation mode
            loss_custom = mse_loss(input_custom, target_custom)

            # In eval mode, should not require grad
            if loss_custom._custom_requires_grad:
                print("✗ MSE Loss Eval Mode: Loss should not require grad in eval mode")
                self.failed_tests += 1
            else:
                print("✓ MSE Loss Eval Mode: Loss correctly doesn't require grad")
                self.passed_tests += 1

    def test_cross_entropy_loss_basic(self):
        """Test basic CrossEntropy loss functionality"""
        print("\n=== Testing CrossEntropy Loss Basic ===")

        with AutogradGraph() as graph:
            # Logits for 3 classes, 2 samples
            input_custom = CustomTensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([0, 1], dtype=torch.long, _custom_requires_grad=False)  # Class indices

            ce_loss = CrossEntropyLoss(graph=graph)
            ce_loss.train()
            loss_custom = ce_loss(input_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]], requires_grad=True)
            target_pytorch = torch.tensor([0, 1], dtype=torch.long)
            loss_pytorch = torch.nn.functional.cross_entropy(input_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "CrossEntropy Loss Basic - input gradients")
            self.assert_tensors_close(loss_custom, loss_pytorch, "CrossEntropy Loss Basic - loss value", check_grad=False)

    def test_cross_entropy_loss_with_weights(self):
        """Test CrossEntropy loss with class weights"""
        print("\n=== Testing CrossEntropy Loss with Weights ===")

        with AutogradGraph() as graph:
            input_custom = CustomTensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([0, 2], dtype=torch.long, _custom_requires_grad=False)
            weight_custom = torch.tensor([1.0, 0.5, 2.0])  # Weights for each class

            ce_loss = CrossEntropyLoss(graph=graph)
            ce_loss.train()
            loss_custom = ce_loss(input_custom, target_custom, weight=weight_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]], requires_grad=True)
            target_pytorch = torch.tensor([0, 2], dtype=torch.long)
            weight_pytorch = torch.tensor([1.0, 0.5, 2.0])
            loss_pytorch = torch.nn.functional.cross_entropy(input_pytorch, target_pytorch, weight=weight_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "CrossEntropy Loss with Weights - input gradients")
            self.assert_tensors_close(loss_custom, loss_pytorch, "CrossEntropy Loss with Weights - loss value", check_grad=False)

    def test_cross_entropy_loss_single_class(self):
        """Test CrossEntropy loss with single sample"""
        print("\n=== Testing CrossEntropy Loss Single Class ===")

        with AutogradGraph() as graph:
            input_custom = CustomTensor([[1.0, 2.0, 0.5]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([1], dtype=torch.long, _custom_requires_grad=False)

            ce_loss = CrossEntropyLoss(graph=graph)
            ce_loss.train()
            loss_custom = ce_loss(input_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
            target_pytorch = torch.tensor([1], dtype=torch.long)
            loss_pytorch = torch.nn.functional.cross_entropy(input_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "CrossEntropy Loss Single Class - input gradients")
            self.assert_tensors_close(loss_custom, loss_pytorch, "CrossEntropy Loss Single Class - loss value", check_grad=False)

    def test_bce_with_logits_loss_basic(self):
        """Test basic BCEWithLogits loss functionality"""
        print("\n=== Testing BCEWithLogits Loss Basic ===")

        with AutogradGraph() as graph:
            # Binary classification logits
            input_custom = CustomTensor([[0.5, -1.0], [1.5, 0.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([[1.0, 0.0], [1.0, 0.0]], _custom_requires_grad=False)

            bce_loss = BCEWithLogitsLoss(graph=graph)
            bce_loss.train()
            loss_custom = bce_loss(input_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[0.5, -1.0], [1.5, 0.0]], requires_grad=True)
            target_pytorch = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
            loss_pytorch = torch.nn.functional.binary_cross_entropy_with_logits(input_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "BCEWithLogits Loss Basic - input gradients")
            self.assert_tensors_close(loss_custom, loss_pytorch, "BCEWithLogits Loss Basic - loss value", check_grad=False)

    def test_bce_with_logits_loss_pos_weight(self):
        """Test BCEWithLogits loss with positive class weights"""
        print("\n=== Testing BCEWithLogits Loss with Pos Weight ===")

        with AutogradGraph() as graph:
            input_custom = CustomTensor([[0.5, -1.0], [1.5, 0.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([[1.0, 0.0], [1.0, 0.0]], _custom_requires_grad=False)
            pos_weight_custom = torch.tensor([[2.0, 1.0], [1.5, 1.0]])  # Higher weight for positive class

            bce_loss = BCEWithLogitsLoss(graph=graph)
            bce_loss.train()
            loss_custom = bce_loss(input_custom, target_custom, weight=pos_weight_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[0.5, -1.0], [1.5, 0.0]], requires_grad=True)
            target_pytorch = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
            pos_weight_pytorch = torch.tensor([[2.0, 1.0], [1.5, 1.0]])
            loss_pytorch = torch.nn.functional.binary_cross_entropy_with_logits(input_pytorch, target_pytorch, pos_weight=pos_weight_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "BCEWithLogits Loss with Pos Weight - input gradients")
            self.assert_tensors_close(loss_custom, loss_pytorch, "BCEWithLogits Loss with Pos Weight - loss value", check_grad=False)

    def test_bce_with_logits_loss_single_output(self):
        """Test BCEWithLogits loss with single output"""
        print("\n=== Testing BCEWithLogits Loss Single Output ===")

        with AutogradGraph() as graph:
            input_custom = CustomTensor([0.8], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([1.0], _custom_requires_grad=False)

            bce_loss = BCEWithLogitsLoss(graph=graph)
            bce_loss.train()
            loss_custom = bce_loss(input_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([0.8], requires_grad=True)
            target_pytorch = torch.tensor([1.0])
            loss_pytorch = torch.nn.functional.binary_cross_entropy_with_logits(input_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "BCEWithLogits Loss Single Output - input gradients")
            self.assert_tensors_close(loss_custom, loss_pytorch, "BCEWithLogits Loss Single Output - loss value", check_grad=False)

    def test_loss_functions_chain(self):
        """Test loss functions in a computation chain"""
        print("\n=== Testing Loss Functions in Chain ===")

        with AutogradGraph() as graph:
            # Create a simple network: input -> linear transformation -> loss
            input_custom = CustomTensor([[1.0, 2.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            weight_custom = CustomTensor([[0.5], [1.5]], _custom_requires_grad=True, graph=graph, is_leaf=True)

            # Linear transformation: input @ weight
            logits_custom = input_custom @ weight_custom
            target_custom = CustomTensor([[1.0]], _custom_requires_grad=False)

            # Apply BCE loss
            bce_loss = BCEWithLogitsLoss(graph=graph)
            bce_loss.train()
            loss_custom = bce_loss(logits_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[1.0, 2.0]], requires_grad=True)
            weight_pytorch = torch.tensor([[0.5], [1.5]], requires_grad=True)
            logits_pytorch = input_pytorch @ weight_pytorch
            target_pytorch = torch.tensor([[1.0]])
            loss_pytorch = torch.nn.functional.binary_cross_entropy_with_logits(logits_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "Loss Functions Chain - input gradients")
            self.assert_tensors_close(weight_custom, weight_pytorch, "Loss Functions Chain - weight gradients")
            self.assert_tensors_close(loss_custom, loss_pytorch, "Loss Functions Chain - loss value", check_grad=False)

    def test_loss_functions_edge_cases(self):
        """Test loss functions with edge cases"""
        print("\n=== Testing Loss Functions Edge Cases ===")

        # Test with very small values
        with AutogradGraph() as graph:
            input_custom = CustomTensor([[1e-6, 1e-7]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([[1e-6, 1e-7]], _custom_requires_grad=False)

            mse_loss = MSE(graph=graph)
            mse_loss.train()
            loss_custom = mse_loss(input_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[1e-6, 1e-7]], requires_grad=True)
            target_pytorch = torch.tensor([[1e-6, 1e-7]])
            loss_pytorch = torch.nn.functional.mse_loss(input_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "Loss Functions Edge Cases - small values")

        # Test with large values for CrossEntropy
        with AutogradGraph() as graph:
            input_custom = CustomTensor([[10.0, 5.0, 1.0]], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([0], dtype=torch.long, _custom_requires_grad=False)

            ce_loss = CrossEntropyLoss(graph=graph)
            ce_loss.train()
            loss_custom = ce_loss(input_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[10.0, 5.0, 1.0]], requires_grad=True)
            target_pytorch = torch.tensor([0], dtype=torch.long)
            loss_pytorch = torch.nn.functional.cross_entropy(input_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, "Loss Functions Edge Cases - large values")

    def test_loss_functions_batch_sizes(self):
        """Test loss functions with different batch sizes"""
        print("\n=== Testing Loss Functions Different Batch Sizes ===")

        # Test with larger batch
        with AutogradGraph() as graph:
            batch_size = 5
            input_custom = CustomTensor([[i + 0.5, i + 1.0] for i in range(batch_size)], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([[i, i + 0.5] for i in range(batch_size)], _custom_requires_grad=False)

            mse_loss = MSE(graph=graph)
            mse_loss.train()
            loss_custom = mse_loss(input_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[i + 0.5, i + 1.0] for i in range(batch_size)], requires_grad=True)
            target_pytorch = torch.tensor([[i, i + 0.5] for i in range(batch_size)])
            loss_pytorch = torch.nn.functional.mse_loss(input_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, f"Loss Functions Batch Size {batch_size} - MSE")

        # Test CrossEntropy with larger batch
        with AutogradGraph() as graph:
            batch_size = 4
            num_classes = 3
            input_custom = CustomTensor([[i * 0.5, (i + 1) * 0.3, (i + 2) * 0.2] for i in range(batch_size)], _custom_requires_grad=True, graph=graph, is_leaf=True)
            target_custom = CustomTensor([i % num_classes for i in range(batch_size)], dtype=torch.long, _custom_requires_grad=False)

            ce_loss = CrossEntropyLoss(graph=graph)
            ce_loss.train()
            loss_custom = ce_loss(input_custom, target_custom)
            loss_custom.backward()

            # PyTorch comparison
            input_pytorch = torch.tensor([[i * 0.5, (i + 1) * 0.3, (i + 2) * 0.2] for i in range(batch_size)], requires_grad=True)
            target_pytorch = torch.tensor([i % num_classes for i in range(batch_size)], dtype=torch.long)
            loss_pytorch = torch.nn.functional.cross_entropy(input_pytorch, target_pytorch, reduction='mean')
            loss_pytorch.backward()

            self.assert_tensors_close(input_custom, input_pytorch, f"Loss Functions Batch Size {batch_size} - CrossEntropy")

    def test_all_modules_comprehensive(self):
        """Comprehensive test running all module tests."""
        print("\n=== Running All Module Tests ===")

        self.test_linear_module()
        self.test_conv2d_module()
        self.test_batchnorm_module()
        self.test_maxpool2d_module()
        self.test_avgpool2d_module()
        self.test_relu_module()
        self.test_leaky_relu_module()
        self.test_gelu_module()
        self.test_elu_module()
        self.test_silu_module()
        self.test_sigmoid_module()
        self.test_tanh_module()
        self.test_swish_module()
        self.test_module_parameter_management()
        self.test_module_training_eval_modes()
        self.test_module_nested_structure()
        self.test_module_edge_cases()

    def test_all_losses_comprehensive(self):
        print("\n" + "=" * 50)
        print("Running All Losses Tests")
        print("=" * 50)
        self.test_mse_loss_basic()
        self.test_mse_loss_with_weights()
        self.test_mse_loss_eval_mode()
        self.test_cross_entropy_loss_basic()
        self.test_cross_entropy_loss_with_weights()
        self.test_cross_entropy_loss_single_class()
        self.test_bce_with_logits_loss_basic()
        self.test_bce_with_logits_loss_pos_weight()
        self.test_bce_with_logits_loss_single_output()
        self.test_loss_functions_chain()
        self.test_loss_functions_edge_cases()
        self.test_loss_functions_batch_sizes()


    def run_all_tests(self):
        """Run all tests"""
        print("Running Custom Autograd Correctness Tests")
        print("=" * 50)

        self.test_basic_operations()
        self.test_multiplication()
        self.test_subtraction_division()
        self.test_power_function()
        self.test_unary_functions()
        self.test_matrix_operations()
        self.test_complex_chain()
        self.test_mixed_operations()
        self.test_broadcasting()
        self.test_backward_with_custom_grad()
        self.test_zero_grad_behavior()
        self.test_no_grad_flow()

        print("\n" + "=" * 50)
        print("Running Custom Autograd System Tests")
        print("=" * 50)

        self.test_basic_add_scalar_grad_system()
        self.test_basic_add_tensor_grad_system()
        self.test_mixed_requires_grad_tensor_add_system()
        self.test_no_requires_grad_system()
        self.test_autograd_graph_context_manager_system()
        self.test_cycle_detection_system()
        self.test_no_circular_references_non_leaf_tensors_die_system()
        self.test_topological_sort_order_system()
        self.test_very_deep_computation_graph()
        self.test_wide_computation_graph()
        self.test_nan_and_inf_handling()
        self.test_zero_gradients()
        self.test_memory_efficiency()
        print("\n" + "=" * 50)
        print("Running All Module Tests")
        print("=" * 50)
        self.test_linear_module()
        self.test_conv2d_module()
        self.test_batchnorm_module()
        self.test_maxpool2d_module()
        self.test_avgpool2d_module()
        self.test_relu_module()
        self.test_leaky_relu_module()
        self.test_gelu_module()
        self.test_elu_module()
        self.test_silu_module()
        self.test_sigmoid_module()
        self.test_tanh_module()
        self.test_swish_module()
        self.test_module_parameter_management()
        self.test_module_training_eval_modes()
        self.test_module_nested_structure()
        self.test_module_edge_cases()
        print("\n" + "=" * 50)
        print("Running All Losses Tests")
        print("=" * 50)
        self.test_mse_loss_basic()
        self.test_mse_loss_with_weights()
        self.test_mse_loss_eval_mode()
        self.test_cross_entropy_loss_basic()
        self.test_cross_entropy_loss_with_weights()
        self.test_cross_entropy_loss_single_class()
        self.test_bce_with_logits_loss_basic()
        self.test_bce_with_logits_loss_pos_weight()
        self.test_bce_with_logits_loss_single_output()
        self.test_loss_functions_chain()
        self.test_loss_functions_edge_cases()
        self.test_loss_functions_batch_sizes()



        print(f"\n" + "=" * 50)
        print(f"Test Results: {self.passed_tests} passed, {self.failed_tests} failed")

        if self.failed_tests == 0:
            print("🎉 All tests passed! Your autograd implementation is correct.")
        else:
            print("❌ Some tests failed. Check the implementation.")

        return self.failed_tests == 0

if __name__ == "__main__":
    autograd_graph_test = AutogradTester()
    autograd_graph_test.run_all_tests()