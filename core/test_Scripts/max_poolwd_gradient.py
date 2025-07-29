import torch
import torch.nn.functional as F

# Manual backprop for MaxPool2d using returned indices

def maxpool2d_forward(x, kernel_size, stride=None, padding=0, dilation=1):

    y, indices = F.max_pool2d(x,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              return_indices=True)
    return y, indices


def maxpool2d_backward(grad_output, indices, input_shape):

    # grad_output: (N, C, H_out, W_out)
    # indices:     (N, C, H_out, W_out)
    N, C, H_out, W_out = grad_output.shape
    # Initialize grad_input
    grad_input = torch.zeros(input_shape, dtype=grad_output.dtype, device=grad_output.device)
    # Flatten spatial dims
    grad_output_flat = grad_output.view(N, C, -1)
    indices_flat = indices.view(N, C, -1)
    grad_input_flat = grad_input.view(N, C, -1)
    # Scatter gradients into appropriate positions
    grad_input_flat.scatter_add_(2, indices_flat, grad_output_flat)
    # Reshape back to input shape
    grad_input = grad_input_flat.view(input_shape)
    return grad_input


if __name__ == "__main__":

    torch.manual_seed(0)

    x = torch.randn(2, 3, 8, 8, requires_grad=True)

    kernel_size = (2, 2)
    stride = (2, 2)
    padding = 0
    dilation = 1

    y, indices = maxpool2d_forward(x, kernel_size, stride, padding, dilation)

    loss = y.sum()

    loss.backward()
    grad_autograd = x.grad.clone()

    x.grad.zero_()

    grad_output = torch.ones_like(y)
    grad_manual = maxpool2d_backward(grad_output, indices, x.shape)
    max_diff = (grad_autograd - grad_manual).abs().max()
    print(f"Max absolute difference between autograd and manual: {max_diff.item():.6f}")
    assert torch.allclose(grad_autograd, grad_manual, atol=1e-6), \
        "Manual backward does not match autograd!"
    print("Manual backprop matches PyTorch autograd!")
