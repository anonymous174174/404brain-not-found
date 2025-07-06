import weakref
import torch
import rustworkx as rx
from torch.utils._python_dispatch import TorchDispatchMode

class TrackingTensor:
    __slots__ = (
        "value",       # the real torch.Tensor
        "requires_grad",
        "is_leaf",
        "dtype",
        "device",
        "_node",
        "graph"
    )

    def __init__(
        self,
        data,
        *,
        requires_grad: bool = False,
        graph: rx.PyDiGraph,
        is_leaf: bool = True,
    ):
        # wrap / coerce into a torch.Tensor
        t = (
            data
            if isinstance(data, torch.Tensor)
            else torch.tensor(data, dtype=getattr(data, "dtype", None))
        )
        # ensure correct dtype/device
        self.value = t.to(device=getattr(t, "device", None))
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.dtype = self.value.dtype
        self.device = self.value.device

        # side‑car DAG
        self.graph = graph
        # create a node for this tensor
        # label leaf nodes differently
        label = f"{'leaf' if is_leaf else 'node'}:{self.dtype}@{self.device}"
        self._node = self.graph.add_node(label)

    @classmethod
    def _wrap(
        cls,
        raw_tensor: torch.Tensor,
        graph: rx.PyDiGraph,
        parents: list["TrackingTensor"],
        func_name: str,
    ) -> "TrackingTensor":
        # create a new wrapper without calling __init__
        t = cls.__new__(cls)
        t.value = raw_tensor
        t.requires_grad = any(p.requires_grad for p in parents)
        t.is_leaf = False
        t.dtype = raw_tensor.dtype
        t.device = raw_tensor.device

        t.graph = graph
        # create node with op name & dtype/device
        label = f"{func_name}:{t.dtype}@{t.device}"
        t._node = graph.add_node(label)

        # add edges from each parent
        for p in parents:
            graph.add_edge(p._node, t._node, func_name)

        return t

    def __repr__(self):
        return (
            f"<TrackingTensor node={self._node} op={'leaf' if self.is_leaf else ''} "
            f"requires_grad={self.requires_grad} shape={tuple(self.value.shape)} "
            f"dtype={self.dtype} device={self.device}>"
        )


class GraphCaptureMode(TorchDispatchMode):
    def __init__(self):
        self.graph = rx.PyDiGraph()
        # optional: weak map node_id -> TrackingTensor
        self._live = weakref.WeakValueDictionary()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *args):
        return super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 1) unwrap TrackingTensor → raw torch.Tensor
        raw_args = [
            a.value if isinstance(a, TrackingTensor) else a
            for a in args
        ]
        raw_kwargs = {
            k: (v.value if isinstance(v, TrackingTensor) else v)
            for k, v in (kwargs or {}).items()
        }

        # 2) call the real op at full C/CUDA speed
        result = func(*raw_args, **raw_kwargs)

        # 3) record into Rustworkx
        parents = [a for a in args if isinstance(a, TrackingTensor)]
        wrapped = TrackingTensor._wrap(result, self.graph, parents, func.__name__)

        # 4) keep it alive if it's a non‑leaf requiring gradient
        if wrapped.requires_grad and not wrapped.is_leaf:
            self._live[wrapped._node] = wrapped

        return wrapped
mode = GraphCaptureMode()
with mode:
    a = TrackingTensor([1,2,3], requires_grad=True, graph=mode.graph)
    b = TrackingTensor(torch.randn(3), requires_grad=True, graph=mode.graph)


with mode:
    c = a + b        # __torch_dispatch__ intercepts torch.add
    d = torch.relu(c)
    e = d * 5.0
