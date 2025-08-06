# import torch
# from warnings import warn
# import logging

# # disabling AutoGrad for the entire module
# logging.basicConfig(
#     filename='core.log',
#     level=logging.DEBUG,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# torch.autograd.set_grad_enabled(False)
# # datatype of tensors
# dtype = torch.float32 
# # Detect hardware availability
# RUN_ON_GPU = torch.cuda.is_available()
# RUN_ON_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built() if not RUN_ON_GPU else False

# # Conditionally check for TPU only if other accelerators are not available
# RUN_ON_TPU = False
# if not RUN_ON_GPU and not RUN_ON_MPS:
#     try:
#         import torch_xla.core.xla_model as xm
#         RUN_ON_TPU = xm.xla_device() is not None # Check if an XLA device is actually available
#     except ImportError:
#         # torch_xla is not installed, so RUN_ON_TPU remains False
#         logging.info("torch_xla is not installed. Skipping TPU device check.")
#     except Exception as e:
#         # Catch other potential errors during XLA device initialization
#         logging.warning(f"Error checking for XLA device: {e}. Skipping TPU device check.")
#         RUN_ON_TPU = False

# RUN_ON_CPU = not RUN_ON_GPU and not RUN_ON_MPS and not RUN_ON_TPU


# # Determine active device
# if RUN_ON_GPU:
#     device = torch.device("cuda")
# elif RUN_ON_MPS:
#     device = torch.device("mps")
# elif RUN_ON_TPU: # This block will only be reached if torch_xla was successfully imported and a TPU is available
#     device = xm.xla_device()
# else:
#     device = torch.device("cpu")

# device_summary = f"Running on: {'GPU' if RUN_ON_GPU else 'MPS' if RUN_ON_MPS else 'TPU' if RUN_ON_TPU else 'CPU'}"
# logging.info(device_summary)

# __all__ = []
# from . import (
#     custom_tensor,
#     autograd_graph,
#     module,
#     optimizers,
#     losses
# )

# if __name__ == "__main__":
#     warn("This module is not intended to be run directly. Please import it in your application.")