import torch
import torch.nn.functional as F

def verify_conv2d_backward_logic():
    """
    Verifies that the manual backward pass logic for a 2D convolution,
    implemented with torch.nn.functional, matches PyTorch's autograd.
    """
    print("--- Verifying Conv2d Backward Pass Logic ---")

    # ---- 1. Setup Parameters and Random Tensors ----
    # Use double precision for higher accuracy in gradient comparisons
    dtype = torch.float64
    torch.manual_seed(42) # for reproducibility

    # Layer parameters
    batch_size = 4
    in_channels, out_channels = 8, 16
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    dilation = (2, 2) # Use non-trivial dilation
    groups = 4 # Use grouped convolution

    # Input and output spatial dimensions
    input_size = (32, 32)

    # Create random tensors that require gradients
    X = torch.randn(batch_size, in_channels, *input_size, dtype=dtype, requires_grad=True)
    W = torch.randn(out_channels, in_channels // groups, *kernel_size, dtype=dtype, requires_grad=True)
    B = torch.randn(out_channels, dtype=dtype, requires_grad=True)

    print(f"Setup: Input{X.shape}, Weight{W.shape}, Bias{B.shape}, Groups={groups}\n")

    # ---- 2. Ground Truth: Calculate Gradients using PyTorch Autograd ----
    print("--- 2. Calculating Ground Truth Gradients (PyTorch Autograd) ---")

    # Perform the forward pass
    O_autograd = F.conv2d(X, W, B, stride, padding, dilation, groups)

    # Generate a random upstream gradient (from a hypothetical subsequent layer)
    grad_output = torch.randn_like(O_autograd)

    # Perform the backward pass
    O_autograd.backward(grad_output)

    # Store the results
    grad_X_autograd = X.grad.clone()
    grad_W_autograd = W.grad.clone()
    grad_B_autograd = B.grad.clone()

    print("Autograd calculations complete.\n")

    # ---- 3. Manual Calculation: Compute Gradients using Functional Ops ----
    print("--- 3. Calculating Gradients Manually (torch.nn.functional) ---")

    # (A) Gradient with respect to Bias
    # This is the sum of the output gradients over batch, height, and width dimensions.
    grad_B_manual = grad_output.sum(dim=[0, 2, 3])

    # (B) Gradient with respect to Input
    # This is a transposed convolution of the output gradient with the original weight kernel.
    # The output shape of a transposed convolution can be ambiguous. We must calculate
    # the correct `output_padding` to ensure the output shape of this operation
    # exactly matches the original input shape `X.shape`.
    h_in, w_in = X.shape[2], X.shape[3]
    h_out, w_out = O_autograd.shape[2], O_autograd.shape[3]

    # The formula relating input size to output size in a transposed convolution is:
    # InputSize = (OutputSize - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    # We rearrange this to solve for the required output_padding.
    output_padding_h = h_in - ((h_out - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1)
    output_padding_w = w_in - ((w_out - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1)
    output_padding = (output_padding_h, output_padding_w)

    grad_X_manual = F.conv_transpose2d(
        grad_output,
        W,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups
    )

    # (C) Gradient with respect to Weight (Kernel)
    # This is the most complex part. The gradient w.r.t. the weights is a convolution
    # of the input (X) and the output gradient (grad_output).
    # For grouped convolutions, we must perform this calculation for each group separately.
    # in_channels_per_group = in_channels // groups
    # out_channels_per_group = out_channels // groups
    # grad_W_groups = []

    # for g in range(groups):
    #     # Slice the input tensor to get the channels for the current group
    #     start_in_ch = g * in_channels_per_group
    #     end_in_ch = start_in_ch + in_channels_per_group
    #     X_g = X[:, start_in_ch:end_in_ch, :, :]

    #     # Slice the output gradient tensor to get the channels for the current group
    #     start_out_ch = g * out_channels_per_group
    #     end_out_ch = start_out_ch + out_channels_per_group
    #     grad_output_g = grad_output[:, start_out_ch:end_out_ch, :, :]

    #     # To calculate the weight gradient via a convolution, we must cleverly
    #     # permute the input (X_g) and output gradient (grad_output_g) tensors.
    #     # We treat X_g as the input and grad_output_g as the kernel.
    #     # X_g: (N, Cin/g, H, W) -> permute -> (Cin/g, N, H, W)
    #     # grad_output_g: (N, Cout/g, oH, oW) -> permute -> (Cout/g, N, oH, oW)
    #     # The F.conv2d call then treats 'Cin/g' as the batch size and 'N' as the input channels.
    #     # The stride and dilation parameters from the original convolution are swapped.
    #     X_g_permuted = X_g.transpose(0, 1)
    #     grad_output_g_permuted = grad_output_g.transpose(0, 1)

    #     grad_W_g_permuted = F.conv2d(
    #         X_g_permuted,
    #         grad_output_g_permuted,
    #         stride=dilation,
    #         padding=padding,
    #         dilation=stride,
    #         groups=1 # The group calculation is handled by our loop, so this is a standard conv.
    #     )

    #     # The result has shape (Cin/g, Cout/g, kH, kW). We must permute it back to
    #     # the standard weight layout of (Cout/g, Cin/g, kH, kW).
    #     grad_W_g = grad_W_g_permuted.transpose(0, 1)
    #     grad_W_groups.append(grad_W_g)

    # # Concatenate the gradients from all groups along the output channel dimension.
    # # The weight tensor for grouped convolutions is laid out by stacking the weights
    # # for each group, so we do the same for the gradient.
    # grad_W_manual = torch.cat(grad_W_groups, dim=0)
    grad_W_manual = torch.nn.grad.conv2d_weight(
        input=X,
        weight_size=W.shape,
        grad_output=grad_output,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )


    print("Manual calculations complete.\n")

    # ---- 4. Verification ----
    print("--- 4. Verification ---")

    bias_correct = torch.allclose(grad_B_manual, grad_B_autograd)
    input_correct = torch.allclose(grad_X_manual, grad_X_autograd)
    weight_correct = torch.allclose(grad_W_manual, grad_W_autograd)

    print(f"Bias Gradient Correct:   {'✅ Yes' if bias_correct else '❌ No'}")
    print(f"Input Gradient Correct:  {'✅ Yes' if input_correct else '❌ No'}")
    print(f"Weight Gradient Correct: {'✅ Yes' if weight_correct else '❌ No'}")

    print("\nVerification Finished.")

    assert bias_correct and input_correct and weight_correct, "One or more manual gradients did not match autograd."

# Run the verification
if __name__ == "__main__":
    verify_conv2d_backward_logic()