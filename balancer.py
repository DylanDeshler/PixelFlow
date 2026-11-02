"""
EnCodec's loss balancer (Defossez et al. 2022, "High Fidelity Neural Audio Compression"), from the Jazz repo
(neural/fad.py, GradientBalancer, used by MultiTaskFAD), unchanged apart from comments.

At a point where shared features branch into several tasks, each branch passes the features through balancer(features,
task). The forward pass is the identity; the backward pass rescales the gradient flowing from that task into the shared trunk
so its running-average norm becomes (task weight / total weight) x total_norm. Each task then contributes a fixed share of
the trunk's gradient, whatever its loss scale, and the share replaces the loss weight as the knob. Layers on the task side of
the branch still see their own loss's gradient unscaled.

Changes from Jazz: an optional reference task, whose running gradient norm replaces the constant total_norm (see
GradientBalancer). Here (train_jit.py --disp_balance > 0): the branch point is the output of the block the Dispersive Loss reads. The rest of
the network, ending in the diffusion loss, is one task ('diff'); the dispersive loss is the other ('disp'). The blocks up to
and including that block get both gradients, normalized to their shares; the blocks after it see the diffusion gradient
as usual.
"""

import torch
import torch.nn as nn


class _GradientBalancerFunction(torch.autograd.Function):
    """
    Intercepts the gradient flowing from one task's branch down into the shared trunk and scales it on the fly.
    """
    @staticmethod
    def forward(ctx, features, total_buffer, fix_buffer, task_weight, total_weight, ema_decay, total_norm, epsilon,
                ref_total=None, ref_fix=None):
        ctx.ref_total, ctx.ref_fix = ref_total, ref_fix # reference task's EMA buffers (total_norm = its running norm)
        ctx.total_buffer = total_buffer
        ctx.fix_buffer = fix_buffer
        ctx.task_weight = task_weight
        ctx.total_weight = total_weight
        ctx.ema_decay = ema_decay
        ctx.total_norm = total_norm
        ctx.epsilon = epsilon
        return features.clone() # the identity in the forward pass

    @staticmethod
    def backward(ctx, grad_output):
        # per-batch-item gradient norm, averaged (EnCodec), into in-place EMA buffers (tensor ops only: no graph breaks)
        dims = tuple(range(1, grad_output.dim()))
        norm = grad_output.norm(dim=dims).mean()
        batch_size = grad_output.shape[0]
        ctx.total_buffer.mul_(ctx.ema_decay).add_(norm * batch_size)
        ctx.fix_buffer.mul_(ctx.ema_decay).add_(batch_size)
        avg_norm = ctx.total_buffer / ctx.fix_buffer
        # scale so this task's gradient norm becomes its share of total_norm (or of the reference task's running norm)
        total_norm = ctx.total_norm
        if ctx.ref_total is not None: # before the reference has a norm (first step), fall back to this task's own
            ref_norm = ctx.ref_total / ctx.ref_fix.clamp_min(1e-12)
            total_norm = torch.where(ctx.ref_fix > 0, ref_norm, avg_norm)
        ratio = ctx.task_weight / ctx.total_weight
        scale = ratio * total_norm / (ctx.epsilon + avg_norm)
        return grad_output * scale, None, None, None, None, None, None, None, None, None


class GradientBalancer(nn.Module):
    """
    weights: {task: weight}; each task's share of the trunk gradient is weight / sum(weights), at total norm total_norm.
    The EMA trackers are buffers, so they live on the device and are saved in the module's state_dict.
    """
    def __init__(self, weights: dict, ema_decay=0.999, total_norm=1.0, epsilon=1e-12, reference=None):
        super().__init__()
        # reference: a task name; total_norm is then that task's running gradient norm instead of a constant, so the trunk's
        # gradient keeps its natural scale (EnCodec's constant 1.0 is far above a mean-over-batch loss's per-item gradient,
        # which would put gradient clipping on every step)
        self.reference = reference
        self.weights = weights
        self.ema_decay = ema_decay
        self.total_norm = total_norm
        self.epsilon = epsilon
        self.total_weight = sum(weights.values())
        for task in weights.keys():
            self.register_buffer(f'total_{task}', torch.tensor(0.0))
            self.register_buffer(f'fix_{task}', torch.tensor(0.0))

    def forward(self, features, task_name):
        # pass the trunk features through this before sending them to a task's branch
        ref = (getattr(self, f'total_{self.reference}'), getattr(self, f'fix_{self.reference}')) if self.reference else (None, None)
        return _GradientBalancerFunction.apply(features, getattr(self, f'total_{task_name}'), getattr(self, f'fix_{task_name}'),
                                               self.weights[task_name], self.total_weight, self.ema_decay, self.total_norm,
                                               self.epsilon, *ref)

    def avg_norms(self):
        # for logging: each task's running-average gradient norm at the branch point (before scaling)
        return {t: (getattr(self, f'total_{t}') / getattr(self, f'fix_{t}').clamp_min(1e-12)).item() for t in self.weights}
