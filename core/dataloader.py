## this was a failure because of custom class not being able to be pickled properly due to weakref proxies and god knows what for multiprocessing for torch dataloader must convert the tensors to custom tensors after receiving the torch tensors
# import torch
# import time
# from typing import Callable, Optional, Union, Any
# from torch.utils.data import DataLoader, Dataset
# from custom_tensor import CustomTensor

# from torch.utils.data._utils.collate import default_collate
# import torch
# from collections.abc import Mapping
# from typing import Any


# class CustomDataLoader:

#     @staticmethod
#     def _wrap_nested(data: Any) -> Any:
#         """
#         Recursively wraps torch.Tensors inside nested data structures with CustomTensor.
#         """
#         if isinstance(data, torch.Tensor):
#             return CustomTensor(data, _custom_requires_grad = False, due_to_operation=True)
#         elif isinstance(data, (list, tuple)):
#             # Handle lists and tuples
#             return type(data)(CustomDataLoader._wrap_nested(x) for x in data)
#         elif isinstance(data, Mapping):
#             # Handle dictionaries and other mappings
#             return type(data)({key: CustomDataLoader._wrap_nested(value) for key, value in data.items()})
#         else:
#             return data

#     @staticmethod
#     def custom_collate_fn(batch: list) -> Any:
#         """
#         Collates a batch and wraps any resulting torch.Tensor objects
#         with CustomTensor, including those in nested structures.
#         """
#         collated = default_collate(batch)
#         return CustomDataLoader._wrap_nested(collated)

#     @staticmethod
#     def custom_data_loader(
#         dataset: torch.utils.data.Dataset,
#         batch_size: int = 32,
#         shuffle: bool = True,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         drop_last: bool = False,
#         **kwargs,
#     ) -> torch.utils.data.DataLoader:
#         return torch.utils.data.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             collate_fn=CustomDataLoader.custom_collate_fn,
#             drop_last=drop_last,
#             **kwargs,
#         )