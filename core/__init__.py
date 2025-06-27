import torch 
from warnings import warn
import logging

# disabling AutoGrad for the entire module
logging.basicConfig(
    filename='core.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
torch.autograd.set_grad_enabled(False)




# Detect hardware availability
RUN_ON_GPU = torch.cuda.is_available()
RUN_ON_TPU = getattr(torch, "xla", None) and torch.xla.is_available() if not RUN_ON_GPU else False
RUN_ON_CPU = not RUN_ON_GPU and not RUN_ON_TPU

# Determine active device
if RUN_ON_GPU:
    device = torch.device("cuda")
elif RUN_ON_TPU:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
else:
    device = torch.device("cpu")

device_summary = f"Running on: {'GPU' if RUN_ON_GPU else 'TPU' if RUN_ON_TPU else 'CPU'}"
logging.info(device_summary)

__all__ = []
from . import (
    tensor
)

if __name__ == "__main__":
    warn("This module is not intended to be run directly. Please import it in your application.")