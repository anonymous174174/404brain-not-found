# don't subclass torch tensor new approach https://chatgpt.com/s/t_6867b46df1288191ad5e0dbf19b3c93f
# https://chatgpt.com/share/6867b73b-d830-800e-9c06-953fcbba142b
import torch
import weakref
from autograd_graph import AutogradGraph

class CustomTensor(torch.Tensor):
    __slots__=("node_id","_custom_requires_grad","_is_leaf","_backward","graph")
    def __new__(cls,data=None,dtype=None,requires_grad=False,is_leaf=False,due_to_operation=False,device=None,graph=None):
        
        if data is None:
            raise ValueError("Data cannot be None")
        if device is None:
            raise ValueError("Device cannot be None")
        
        if graph is None and requires_grad:
            raise ValueError("Graph instance must be provided for tensors which require gradients")
        dtype = dtype if dtype is not None else torch.float16
        
        if due_to_operation and requires_grad:#if isinstance(data,CustomTensor) and due_to_operation:

            instance = data
            instance._custom_requires_grad = True # requires_grad # Must set True when due to operation is set to True and while making this type of tensor
            instance._is_leaf = False # is_leaf # must set False while making this type of tensor as it is an intermediate tensor
            instance._backward = lambda : None # must set this correctly in __operation__ methods like __add__ __mul__ etc
            instance.node_id = None
            graph.add_tensor_graph(instance) # the graph class will initialize node id inside the function
            graph.add_non_leaf_tensor_references(instance) # stores a strong reference to the tensor in a python dict 
            instance.graph = weakref.ref(graph) 
            return instance # returning a strong reference here but __operation__ methods like __add__ before returning the new tensor must enclose it with weakref.ref(instance) for non leaf gradient tensors
        elif due_to_operation and not requires_grad:
            instance = data
            instance._is_leaf = False
            instance._custom_requires_grad = False # It can only be False
            instance.node_id = None
            instance._backward = lambda : None
            instance.graph = None
            return instance
        elif not isinstance(data,CustomTensor):
            data = torch.as_tensor(data,dtype=dtype,device=device)
            instance = torch.Tensor._make_subclass(cls,data)
            instance.requires_grad_(False)
        else:
            return data # if the tensor is already created return it as it is
        


        if not requires_grad:
            instance._is_leaf = False
            instance._custom_requires_grad = False # It can only be False
            instance.node_id = None
            instance._backward = lambda : None
            instance.graph = None
            return instance # strong reference being retuned for a non leaf non gradient tensor
        else:
            # this part will always initiate leaf tensors
            instance._is_leaf = is_leaf # should always be True
            instance._custom_requires_grad = True # It can only be True
            instance.node_id = None
            instance._backward = lambda : None
            graph.add_tensor_graph(instance) # the graph class will initialize node id inside the function
            # graph.add_non_leaf_tensor_references(instance) # leaf tensors don't need strong references 
            instance.graph = weakref.ref(graph)
            return instance
    
    def zero_(self):
        return CustomTensor(data=torch.zeros(self.shape),requires_grad=False,dtype=self.dtype,device=self.device)
    def add_grad(self):
        pass #how the hell do i add grad now?

    def __add__(self,other):
        """
        There are 9 cases possible here:
        1. scaler and non leaf tensor (without grad)
        2. non leaf and non leaf (without gradients for both)
        3. scaler and non leaf tensor (with grad)
        4. non leaf and non leaf (both with gradients)
        5. leaf and scaler (with grad)
        6. leaf and a leaf (both with grad)
        7. non leaf and a leaf (both with grad)
        8. non leaf non grad and non leaf with grad 
        9. non leaf non grad and leaf grad

        Scalar and Leaf (with grad) tick

        Scalar and Non-leaf (no grad) tick

        Scalar and Non-leaf (with grad) tick

        Leaf (with grad) and Leaf (with grad) tick

        Leaf (with grad) and Non-leaf (with grad) tick

        Leaf (with grad) and Non-leaf (no grad) tick

        Non-leaf (no grad) and Non-leaf (no grad) tick

        Non-leaf (no grad) and Non-leaf (with grad) tick

        Non-leaf (with grad) and Non-leaf (with grad) tick

        Let the Madness Begin
        """
        # determine type of the operands
        self_is_weak_ref = isinstance(self,weakref.ref) # if True cannot be a leaf tensor
        other_is_weak_ref = isinstance(other,weakref.ref)# if True cannot be a leaf tensor
        if self_is_weak_ref:
            self = self()
        if other_is_weak_ref:
            other = other()

        self_is_custom_tensor = isinstance(self,CustomTensor)
        other_is_custom_tensor = isinstance(other,CustomTensor)
        if not (self_is_custom_tensor or other_is_custom_tensor):
            raise ValueError("At least one of the operand must be a Custom Tensor")
        
        
        if self_is_custom_tensor and other_is_custom_tensor:
            if self._custom_requires_grad or other_is_custom_tensor:
                def _backward():
                    if self._custom_requires_grad:
                        if self.grad is None:
                            self.grad = torch.zeros(self.shape,dtype=self.dtype,device=self.device,requires_grad=False)#CustomTensor(data=torch.zeros(self.shape),requires_grad=False,dtype=self.dtype,device=self.device)
                        self.grad+=output.grad
                    if other._custom_requires_grad:
                        if other.grad is None:
                            other.grad = torch.zeros(other.shape,dtype=other.dtype,device=other.device,requires_grad=False)
                        other.grad+=output.grad
            if self._custom_requires_grad and other._custom_requires_grad: # if even one of the tensors requires grad then intermediate tensor also requires a grad
                output = super().add__(other)
                # pytorch returns the output as a custom Tensor but without any of the custom attributes 
                graph=self.graph()
                output = CustomTensor(data=output,dtype=self.dtype,requires_grad=True,is_leaf=False,due_to_operation=True,device=self.device,graph=graph) #self.graph() to pass a strong reference of the graph
                graph.add_edge(self.node_id,output.node_id)
                graph.add_edge(other.node_id,output.node_id)
                output._backward=_backward
                return weakref.ref(output) #returning a weak reference because this is a non leaf tensor with grad 
            elif self._custom_requires_grad and not other._custom_requires_grad:
                output = super().add__(other)
                # pytorch returns the output as a custom Tensor but without any of the custom attributes 
                # we put attributes of self because self is the tensor with requires Grad = True
                graph=self.graph()
                output = CustomTensor(data = output,dtype=self.dtype,requires_grad=True,is_leaf=False,due_to_operation=True,device=self.device,graph=graph) #self.graph() to pass a strong reference of the graph)
                graph.add_edge(self.node_id,output.node_id)
                output._backward=_backward
                return weakref.ref(output)
            elif not self._custom_requires_grad and other._custom_requires_grad:
                output = super().add__(other)
                graph=other.graph()
                output = CustomTensor(data = output,dtype=self.dtype,requires_grad=True,is_leaf=False,due_to_operation=True,device=self.device,graph=graph) #self.graph() to pass a strong reference of the graph)
                graph.add_edge(other.node_id,output.node_id)
                output._backward=_backward
                return weakref.ref(output)
            else:
                output = super().add__(other)
                output = CustomTensor(data = output,dtype=self.dtype,requires_grad=False,is_leaf=False,due_to_operation=False,device=self.device,graph=None)
                return output
        elif self_is_custom_tensor and not other_is_custom_tensor:

            if self._custom_requires_grad:
                output = super().add__(other)
                graph=self.graph()
                output = CustomTensor(data = output,dtype=self.dtype,requires_grad=True,is_leaf=False,due_to_operation=True,device=self.device,graph=graph) #self.graph() to pass a strong reference of the graph)
                graph.add_edge(self.node_id,output.node_id)
                return weakref.ref(output)
            else:
                output = super().add__(other)
                output = CustomTensor(data = output,dtype=self.dtype,requires_grad=False,is_leaf=False,due_to_operation=False,device=self.device,graph=None)
                return output
        elif not self_is_custom_tensor and other_is_custom_tensor:
            if other._custom_requires_grad:
                output = super().add__(other)
                graph=other.graph()
                output = CustomTensor(data = output,dtype=self.dtype,requires_grad=True,is_leaf=False,due_to_operation=True,device=self.device,graph=graph) #self.graph() to pass a strong reference of the graph)
                graph.add_edge(other.node_id,output.node_id)
                return weakref.ref(output)
            else:
                output = super().add__(other)
                output = CustomTensor(data = output,dtype=self.dtype,requires_grad=False,is_leaf=False,due_to_operation=False,device=self.device,graph=None)
                return output
        
            




