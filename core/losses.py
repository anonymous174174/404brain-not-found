from module import Module
import weakref
import torch
import torch.nn.functional as F
from custom_tensor import CustomTensor
from .__init__ import device, dtype
# TODO Lone MSE , MSE with softmax, MSE with sigmoid, cross entropy with softmax, binary cross entropy with sigmoid


class MSE(Module):
    def __init__(self, *, graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None
    
    def forward(self, input_tensor, target_tensor, 
                weight = None):

        output_tensor = F.mse_loss(
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
        
        if self.graph is not None:
            self.graph.add_edge(input_tensor._node_id, result._node_id)
            result._backward = self._create_backward(input_tensor, target_tensor, weight)
            
        return result

    def _create_backward(self, input_tensor, target_tensor, 
                        weight):
        input_ref = weakref.proxy(input_tensor)
        target_ref = weakref.proxy(target_tensor)
        weight_ref = weakref.proxy(weight) if weight is not None else None
        
        def _backward():      
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            
            grad_input = self._calculate_input_grad(
                input_ref.tensor, 
                target_ref.tensor, 
                weight_ref
            )
            
            input_ref.tensor.grad.add_(grad_input)

        return _backward
    
    @torch.compile
    def _calculate_input_grad(self, input_tensor, target_tensor, 
                             weight):
        if weight is None:
            return 2 * (input_tensor - target_tensor) / input_tensor.numel()
        else:
            return (2 * weight * (input_tensor - target_tensor)) / weight.sum()
        

class CrossEntropyLoss(Module):
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
        weight_ref = weakref.proxy(weight) if weight is not None else None
        
        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            
            grad_input = self._calculate_input_grad(
                input_ref.tensor, 
                target_ref.tensor, 
                weight_ref
            )
            input_ref.tensor.grad.add_(grad_input)
                    
        return _backward
    
    @torch.compile
    def _calculate_input_grad(self, input_tensor, target_tensor, 
                             weight):
        batch_size = input_tensor.size(0)
        num_classes = input_tensor.size(1)
        
        target_one_hot = F.one_hot(target_tensor, num_classes=num_classes).to(input_tensor.dtype)
        
        softmax_probs = F.softmax(input_tensor, dim=1)

        grad = (softmax_probs - target_one_hot) / batch_size
        
        if weight is not None:
            weight_expanded = weight.view(1, -1).expand_as(input_tensor)
            grad = grad * weight_expanded
        
        return grad
        
class BCEWithLogitsLoss(Module):
    def __init__(self, *, graph=None):
        super().__init__()
        self.graph = weakref.proxy(graph) if graph is not None else None

    def forward(self, input_tensor, target_tensor, weight=None):

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
        weight_ref = weakref.proxy(weight) if weight is not None else None

        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()

            grad_input = self._calculate_input_grad(
                input_ref.tensor,
                target_ref.tensor,
                weight_ref
            )
            input_ref.tensor.grad.add_(grad_input)

        return _backward

    @torch.compile
    def _calculate_input_grad(self, input_tensor, target_tensor, weight):
        sigmoid_input = torch.sigmoid(input_tensor)
        grad = (sigmoid_input - target_tensor) / input_tensor.numel()
        if weight is not None:
            grad = grad * weight
        return grad
    
class BCEWithLogitsLoss(Module):
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
        weight_ref = weakref.proxy(weight) if weight is not None else None
        
        def _backward():
            if input_ref.tensor.grad is None:
                input_ref._zero_grad()
            
            grad_input = self._calculate_input_grad(
                input_ref.tensor,
                target_ref.tensor,
                weight_ref
            )
            

            input_ref.tensor.grad.add_(grad_input)

        return _backward
    
    @torch.compile
    def _calculate_input_grad(self, input_tensor, target_tensor, weight):
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

    


















