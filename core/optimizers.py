import torch

class Optimizer:
    def __init__(self, params, defaults):
        self.param_groups = []
        self.state = {}#weakref.WeakKeyDictionary()
        param_list = list(params)

        if not param_list:
            raise ValueError("Optimizer got an empty parameter list.")

        param_group = {'params': param_list, **defaults}
        self.param_groups.append(param_group)

    def step(self):
        raise NotImplementedError

    def clear(self):
        self.param_group = []
        self.state.clear()

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.tensor.grad is not None:
                    p.tensor.zero_()

class SGD(Optimizer):
    def __new__(cls, params, lr, weight_decay=None):
        assert lr > 0
        assert weight_decay is None or weight_decay > 0
        return super().__new__(cls)

    def __init__(self, params, lr, weight_decay=None):
        defaults = {'lr': lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                t = p.tensor
                grad = t.grad
                if grad is None:
                    continue

                if weight_decay:
                    grad.add_(t, alpha=weight_decay)

                t.add_(grad, alpha=-lr)


class Momentum(Optimizer):
    def __new__(cls, params, lr, momentum=0.0, weight_decay=None):
        assert lr > 0
        assert momentum > 0
        assert weight_decay is None or weight_decay > 0
        return super().__new__(cls)

    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        defaults = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def step(self):
        state = self.state
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            for p in group['params']:
                t=p.tensor
                grad = t.grad
                if grad is None:
                    continue
                if weight_decay:
                    grad.add_(t, alpha=weight_decay)

                if p not in state:
                    buf = torch.clone(grad)
                    state[p] = {'momentum_buffer': buf }
                else:
                    buf = state[p]['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                grad = buf
                t.add_(grad,alpha=-lr)

class Nesterov(Optimizer):
    #this is not a True nesterov but a reformulated nesterov
    def __new__(cls, params, lr, momentum=0.0, weight_decay=None):
        assert lr > 0
        assert momentum > 0
        assert weight_decay is None or weight_decay > 0
        return super().__new__(cls)

    def __init__(self, params, lr, momentum=0.0, weight_decay=None):
        defaults = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def step(self):
        state = self.state
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            for p in group['params']:
                t = p.tensor
                grad = t.grad
                if grad is None:
                    continue
                if weight_decay:
                    grad.add_(t, alpha=weight_decay)
                if p not in state:
                    buf = grad*(1-momentum)
                    state[p] = {'momentum_buffer': buf}
                else:
                    buf = state[p]['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1-momentum)
                grad = buf
                t.add_(grad, alpha=-lr)


class AdamW(Optimizer):
    def __new__(cls, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=None):
        assert lr >= 0.0
        assert 0.0 <= betas[0] < 1.0
        assert 0.0 <= betas[1] < 1.0
        assert eps >= 0.0
        assert weight_decay is None or weight_decay > 0.0
        return super().__new__(cls)

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=None):
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, (beta1, beta2), eps, weight_decay = (
                group['lr'], group['betas'], group['eps'], group['weight_decay']
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if p not in self.state:
                    self.state[p] = {
                        'time_step': 0,
                        'm': torch.zeros_like(p.tensor),
                        'v': torch.zeros_like(p.tensor)
                    }

                state = self.state[p]
                m, v = state['m'], state['v']

                state['time_step'] += 1
                t = state['time_step']

                if weight_decay:
                    p.tensor.mul_(1 - lr * weight_decay)


                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_corrected = m / (1 - beta1 ** t)
                v_corrected = v / (1 - beta2 ** t)

                update = m_corrected / (v_corrected.sqrt() + eps)
                p.tensor.add_(update, alpha=-lr)


class Lion(Optimizer):
    """Implements the Lion optimizer.
    
    Based on the paper "Symbolic Discovery of Optimization Algorithms"
    and reference implementation: https://github.com/lucidrains/lion-pytorch
    """
    def __init__(
        self,
        params,
        lr = 1e-4,
        betas = (0.9, 0.99),
        weight_decay = None
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay is None or weight_decay > 0
        self._init_lr = lr
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, wd, (beta1, beta2) = group['lr'], group['weight_decay'], group['betas']
            state = self.state

            for p_obj in group['params']:
                p = p_obj.tensor
                if p.grad is None:
                    continue

                grad = p.grad

                if p_obj not in state:
                    state[p_obj] = {"exp_avg": torch.zeros_like(p)}

                exp_avg = state[p_obj]['exp_avg']

                if wd:
                    p.data.mul_(1. - lr * wd)

                update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1. - beta1).sign_()
                p.add_(update, alpha = -lr)
                exp_avg.mul_(beta2).add_(grad, alpha = 1. - beta2)
