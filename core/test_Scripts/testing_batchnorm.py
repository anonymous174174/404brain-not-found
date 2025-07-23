import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def manual_batchnorm_forward_backward(input_tensor: torch.Tensor, 
                                    weight: torch.Tensor, 
                                    bias: torch.Tensor, 
                                    eps: float = 1e-5,
                                    grad_output:torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor]:
    """
    Manual implementation of BatchNorm forward and backward pass
    Returns: output, grad_input, grad_weight, grad_bias
    """
    # Forward pass
    channel_axis = 1
    reduce_dims = tuple(i for i in range(input_tensor.dim()) if i != channel_axis)
    
    # Compute statistics
    mean = input_tensor.mean(dim=reduce_dims, keepdim=False)
    var = input_tensor.var(dim=reduce_dims, keepdim=False, unbiased=False)
    total_elements = input_tensor.numel() // input_tensor.shape[channel_axis]
    
    # Reshape for broadcasting
    input_shape = input_tensor.shape
    broadcast_shape = (1,) + (input_shape[1],) + (1,) * (len(input_shape) - 2)
    
    mean_shaped = mean.view(broadcast_shape)
    var_shaped = var.view(broadcast_shape)
    weight_shaped = weight.view(broadcast_shape)
    bias_shaped = bias.view(broadcast_shape)
    
    # Forward computation
    inv_std = torch.rsqrt(var_shaped + eps)
    input_minus_mean = input_tensor - mean_shaped
    normalized = input_minus_mean * inv_std
    output = normalized * weight_shaped + bias_shaped
    
    # # Backward pass - assume gradient w.r.t output is all ones for simplicity
    # grad_output = torch.ones_like(output)
    
    # Gradient w.r.t bias
    grad_bias = grad_output.sum(dim=reduce_dims)
    
    # Gradient w.r.t weight
    grad_weight = (grad_output * normalized).sum(dim=reduce_dims)
    
    # Gradient w.r.t input (using your formula)
    outer_term = weight_shaped * inv_std
    term_1 = grad_output
    term_2 = (-1/total_elements) * grad_output.sum(dim=reduce_dims, keepdim=True)
    term3_sum_component = (input_minus_mean * grad_output).sum(dim=reduce_dims, keepdim=True)
    term3 = inv_std**2 * (-1/total_elements) * input_minus_mean * term3_sum_component
    grad_input = outer_term * (term_1 + term_2 + term3)
    
    return output, grad_input, grad_weight, grad_bias

def test_batchnorm_gradients():
    """Test BatchNorm gradients against PyTorch's implementation"""
    
    print("Testing BatchNorm Gradient Verification")
    print("=" * 50)
    
    # Test parameters
    batch_size = 4
    channels = 3
    height = 8
    width = 8
    eps = 1e-5
    
    # Create test data
    input_data = torch.randn(batch_size, channels, height, width, requires_grad=True)
    weight_data = torch.randn(channels, requires_grad=True)
    bias_data = torch.randn(channels, requires_grad=True)
    grad_output = torch.randn_like(input_data)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Weight shape: {weight_data.shape}")
    print(f"Bias shape: {bias_data.shape}")
    print()
    
    # Manual implementation
    print("Running manual implementation...")
    manual_output, manual_grad_input, manual_grad_weight, manual_grad_bias = manual_batchnorm_forward_backward(
        input_data.detach().clone(), 
        weight_data.detach().clone(), 
        bias_data.detach().clone(), 
        eps,
        grad_output#..clone()
    )
    
    # PyTorch implementation
    print("Running PyTorch implementation...")
    input_torch = input_data.detach().clone().requires_grad_(True)
    weight_torch = weight_data.detach().clone().requires_grad_(True)
    bias_torch = bias_data.detach().clone().requires_grad_(True)
    
    # Use functional batch norm for exact control
    torch_output = torch.nn.functional.batch_norm(
        input_torch, 
        running_mean=None, 
        running_var=None, 
        weight=weight_torch, 
        bias=bias_torch, 
        training=True, 
        momentum=0.1, 
        eps=eps
    )
    
    # Backward pass with same gradient
    
    grad_output_torch = grad_output #torch.ones_like(torch_output)
    torch_output.backward(grad_output_torch)
    
    torch_grad_input = input_torch.grad
    torch_grad_weight = weight_torch.grad
    torch_grad_bias = bias_torch.grad
    
    print()
    print("Comparing Results:")
    print("-" * 30)
    
    # Compare outputs
    output_diff = torch.abs(manual_output - torch_output).max().item()
    print(f"Max output difference: {output_diff:.2e}")
    
    # Compare gradients
    grad_input_diff = torch.abs(manual_grad_input - torch_grad_input).max().item()
    grad_weight_diff = torch.abs(manual_grad_weight - torch_grad_weight).max().item()
    grad_bias_diff = torch.abs(manual_grad_bias - torch_grad_bias).max().item()
    
    print(f"Max grad_input difference: {grad_input_diff:.2e}")
    print(f"Max grad_weight difference: {grad_weight_diff:.2e}")
    print(f"Max grad_bias difference: {grad_bias_diff:.2e}")
    
    print()
    print("Relative Errors:")
    print("-" * 20)
    
    # Relative errors
    output_rel_error = (output_diff / torch.abs(torch_output).max().item()) * 100
    grad_input_rel_error = (grad_input_diff / torch.abs(torch_grad_input).max().item()) * 100
    grad_weight_rel_error = (grad_weight_diff / torch.abs(torch_grad_weight).max().item()) * 100
    grad_bias_rel_error = (grad_bias_diff / torch.abs(torch_grad_bias).max().item()) * 100
    
    print(f"Output relative error: {output_rel_error:.4f}%")
    print(f"Grad_input relative error: {grad_input_rel_error:.4f}%")
    print(f"Grad_weight relative error: {grad_weight_rel_error:.4f}%")
    print(f"Grad_bias relative error: {grad_bias_rel_error:.4f}%")
    
    print()
    print("Verification Results:")
    print("-" * 25)
    
    tolerance = 1e-5
    
    tests_passed = 0
    total_tests = 4
    
    if output_diff < tolerance:
        print("✓ Forward pass: PASSED")
        tests_passed += 1
    else:
        print("✗ Forward pass: FAILED")
    
    if grad_input_diff < tolerance:
        print("✓ Input gradient: PASSED")
        tests_passed += 1
    else:
        print("✗ Input gradient: FAILED")
    
    if grad_weight_diff < tolerance:
        print("✓ Weight gradient: PASSED")
        tests_passed += 1
    else:
        print("✗ Weight gradient: FAILED")
    
    if grad_bias_diff < tolerance:
        print("✓ Bias gradient: PASSED")
        tests_passed += 1
    else:
        print("✗ Bias gradient: FAILED")
    
    print()
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All gradient calculations are correct!")
    else:
        print("⚠️  Some gradients need adjustment.")
    
    return tests_passed == total_tests

def test_different_shapes():
    """Test with different input shapes"""
    print("\n" + "=" * 50)
    print("Testing Different Input Shapes")
    print("=" * 50)
    
    test_shapes = [
        (2, 4, 10, 10),      # Standard 2D
        (1, 8, 16),          # 1D case
        (3, 6, 5, 5, 5),     # 3D case
    ]
    
    all_passed = True
    
    for i, shape in enumerate(test_shapes):
        print(f"\nTest {i+1}: Shape {shape}")
        print("-" * 30)
        
        channels = shape[1]
        input_data = torch.randn(shape, requires_grad=True)
        weight_data = torch.randn(channels, requires_grad=True)
        bias_data = torch.randn(channels, requires_grad=True)
        grad_output = torch.randn_like(input_data,requires_grad=False)
        try:
            manual_output, manual_grad_input, manual_grad_weight, manual_grad_bias = manual_batchnorm_forward_backward(
                input_data.detach().clone(), 
                weight_data.detach().clone(), 
                bias_data.detach().clone(),
                grad_output =grad_output

            )
            
            # PyTorch implementation
            input_torch = input_data.detach().clone().requires_grad_(True)
            weight_torch = weight_data.detach().clone().requires_grad_(True)
            bias_torch = bias_data.detach().clone().requires_grad_(True)
            
            torch_output = torch.nn.functional.batch_norm(
                input_torch, None, None, weight_torch, bias_torch, 
                training=True, momentum=0.1, eps=1e-5
            )
            
            torch_output.backward(grad_output)
            
            # Check differences
            output_diff = torch.abs(manual_output - torch_output).max().item()
            grad_input_diff = torch.abs(manual_grad_input - input_torch.grad).max().item()
            
            print(f"Max output difference: {output_diff:.2e}")
            print(f"Max grad_input difference: {grad_input_diff:.2e}")
            
            if output_diff < 1e-5 and grad_input_diff < 1e-5:
                print("✓ PASSED")
            else:
                print("✗ FAILED")
                all_passed = False
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    # Run main test
    main_test_passed = test_batchnorm_gradients()
    
    # Run shape tests
    shape_tests_passed = test_different_shapes()
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    if main_test_passed and shape_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your BatchNorm gradient implementation is correct!")
    else:
        print("⚠️  Some tests failed. Check the implementation.")