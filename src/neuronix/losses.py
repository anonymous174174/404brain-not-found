from .module import Module
import weakref
import torch
import torch.nn.functional as F
from .custom_tensor import CustomTensor
from .config import device, dtype
# TODO Lone MSE , MSE with softmax, MSE with sigmoid, cross entropy with softmax, binary cross entropy with sigmoid
class MSE(Module):
    __slots__ = ('graph','__weakref__')
    def __init__(self, *, graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self, input_tensor, target_tensor, weight=None):
        input_t = input_tensor.tensor
        target_t = target_tensor.tensor

        if weight is None:
            loss = F.mse_loss(input_t, target_t, reduction='mean')
        else:
            weight_t = weight
            squared_error = (input_t - target_t) ** 2

            if weight_t.shape == input_t.shape:
                # Per-pixel weight
                weighted_error = weight_t * squared_error
                loss = weighted_error.sum() / weight_t.sum()

            elif weight_t.ndim == 1 and weight_t.shape[0] == input_t.shape[1]:
                # Per-class weight
                dims_to_add = [1] * (input_t.ndim - 2)
                weight_t = weight_t.view(1, -1, *dims_to_add)
                weighted_error = weight_t * squared_error
                loss = weighted_error.sum() / weight_t.sum()

            else:
                raise ValueError(f"Unsupported weight shape: {weight_t.shape}")

        if not self.training:
            return CustomTensor(loss, due_to_operation=True)

        result = CustomTensor(
            loss,
            _custom_requires_grad=True,
            graph=self.graph,
            due_to_operation=True,
            is_leaf=False
        )

        if self.graph is not None:
            self.graph.add_edge(input_tensor._node_id, result._node_id)
            result._backward = self._create_backward(input_tensor, target_tensor, weight)

        return result

    def _create_backward(self, input_tensor, target_tensor, weight):
        input_ref = weakref.proxy(input_tensor)
        target_ref = weakref.proxy(target_tensor)
        weight_ref = weight if weight is not None else None

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()

            grad_input = MSE._calculate_input_grad(
                input_ref.tensor,
                target_ref.tensor,
                weight_ref
            )
            input_ref.tensor.grad.add_(grad_input)

        return _backward
    @staticmethod
    #@torch.compile
    def _calculate_input_grad(input_t, target_t, weight):
        diff = input_t - target_t
        if weight is None:
            return (2 * diff) / input_t.numel()

        if weight.shape == input_t.shape:
            return (2 * weight * diff) / weight.sum()

        elif weight.ndim == 1 and weight.shape[0] == input_t.shape[1]:
            dims_to_add = [1] * (input_t.ndim - 2)
            weight = weight.view(1, -1, *dims_to_add)
            return (2 * weight * diff) / weight.sum()

        else:
            raise ValueError(f"Unsupported weight shape in backward: {weight.shape}")

class CrossEntropyLoss(Module):
    __slots__ = ('graph','__weakref__')
    def __init__(self, *, graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self, input_tensor, target_tensor, weight= None):

        output_tensor = F.cross_entropy(
            input_tensor.tensor,
            target_tensor.tensor,
            reduction='mean',
            weight=weight
        )

        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)

        result = CustomTensor(
            output_tensor,
            _custom_requires_grad=True,
            graph=self.graph,
            due_to_operation=True,
            is_leaf=False
        )

        self.graph.add_edge(input_tensor._node_id, result._node_id)
        result._backward = self._create_backward(input_tensor, target_tensor, weight)
        return result



    def _create_backward(self, input_tensor, target_tensor,
                        weight):
        input_ref = weakref.proxy(input_tensor)
        target_ref = weakref.proxy(target_tensor)
        weight_ref = weight

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()

            grad_input = CrossEntropyLoss._calculate_input_grad(
                input_ref.tensor,
                target_ref.tensor,
                weight_ref
            )
            input_ref.tensor.grad.add_(grad_input)

        return _backward
    @staticmethod
    #@torch.compile
    def _calculate_input_grad(input_tensor, target_tensor,
                             weight):
        batch_size = input_tensor.size(0)
        num_classes = input_tensor.size(1)

        target_one_hot = F.one_hot(target_tensor, num_classes=num_classes).to(input_tensor.dtype)

        softmax_probs = F.softmax(input_tensor, dim=1)

        grad = softmax_probs - target_one_hot

        if weight is not None:
            sample_weights = weight[target_tensor].view(-1, 1)
            grad = grad * sample_weights
            normalizer = sample_weights.sum()
        else:
            normalizer = batch_size
        grad = grad / normalizer
        return grad

class BCEWithLogitsLoss(Module):
    __slots__ = ('graph','__weakref__')
    def __init__(self, *, graph=None):

        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self, input_tensor, target_tensor, weight= None):
        output_tensor = F.binary_cross_entropy_with_logits(
            input_tensor.tensor,
            target_tensor.tensor,
            reduction='mean',
            pos_weight=weight
        )

        if not self.training:
            return CustomTensor(output_tensor, due_to_operation=True)


        result = CustomTensor(
            output_tensor,
            _custom_requires_grad=True,
            graph=self.graph,
            due_to_operation=True,
            is_leaf=False
        )

        if self.graph is not None:
            self.graph.add_edge(input_tensor._node_id, result._node_id)
            result._backward = self._create_backward(input_tensor, target_tensor, weight)

        return result

    def _create_backward(self, input_tensor, target_tensor, weight):

        input_ref = weakref.proxy(input_tensor)
        target_ref = weakref.proxy(target_tensor)
        weight_ref = weight

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()

            grad_input = BCEWithLogitsLoss._calculate_input_grad(
                input_ref.tensor,
                target_ref.tensor,
                weight_ref
            )


            input_ref.tensor.grad.add_(grad_input)

        return _backward
    @staticmethod
    #@torch.compile
    def _calculate_input_grad(input_tensor, target_tensor, weight):
        sigmoid_input = torch.sigmoid(input_tensor)

        grad = (sigmoid_input - target_tensor) / input_tensor.numel()

        if weight is not None:
            # pos_weight affects the positive class term (where target == 1)
            # The gradient becomes: (sigmoid - target) * weight / num_elements for positive targets
            # For negative targets, it remains: sigmoid / num_elements
            # This matches PyTorch's implementation of pos_weight in BCEWithLogitsLoss
            weight_factor = torch.where(target_tensor == 1, weight, 1.0)
            grad = grad * weight_factor

        return grad
    
