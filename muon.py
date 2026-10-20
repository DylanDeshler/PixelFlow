"""
Muon, single device: momentum SGD whose update for each 2D weight matrix is orthogonalized with a Newton-Schulz iteration.
Meant for the hidden matrices of the transformer blocks only; embeddings, the output layer, gains and biases stay on
AdamW (see train_jit.py's optimizer setup).

Two published update scalings (adjust_lr):
    'original'          Keller Jordan's reference (https://github.com/KellerJordan/Muon): update * sqrt(max(1, A/B)),
                        used with its own lr (default 0.02) and weight decay (default 0)
    'match_rms_adamw'   Moonshot, "Muon is Scalable for LLM Training" (2025): update * 0.2 * sqrt(max(A, B)), which gives
                        the update an RMS of ~0.2 like AdamW's, so Muon uses the same lr and weight decay as AdamW.
                        Weight decay is decoupled and uses the unscaled lr: W <- W - lr * (0.2 sqrt(max(A, B)) O + wd W).
Both match torch.optim.Muon's adjust_lr_fn of the same names.

split: a fused weight (e.g. qkv) is orthogonalized as `split` separate row blocks, each its own matrix (as modded-nanogpt
does for Q, K, V), with the scaling computed from the block shape.
"""

import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Approximately orthogonalizes G (replaces its singular values by ~1) with a quintic Newton-Schulz iteration in bfloat16,
    using the reference implementation's coefficients.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transpose = G.size(-2) > G.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5, adjust_lr='original', split=1):
        assert adjust_lr in ('original', 'match_rms_adamw'), f'unknown adjust_lr {adjust_lr}'
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                                      ns_steps=ns_steps, adjust_lr=adjust_lr, split=split))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                g = p.grad
                buf.lerp_(g, 1 - group['momentum'])
                update = g.lerp(buf, group['momentum']) if group['nesterov'] else buf
                update = update.reshape(len(update), -1) # conv-shaped weights as matrices
                k = group['split']
                assert update.size(0) % k == 0, f'cannot split {tuple(update.shape)} into {k} row blocks'
                A, B = update.size(0) // k, update.size(1) # shape of each orthogonalized block
                update = zeropower_via_newtonschulz5(update.view(k, A, B), group['ns_steps']).reshape(p.size(0), -1)
                if group['adjust_lr'] == 'original':
                    scale = max(1, A / B) ** 0.5
                else:
                    scale = 0.2 * max(A, B) ** 0.5
                if group['weight_decay']:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update.reshape(p.shape).to(p.dtype), alpha=-group['lr'] * scale)

class CombinedOptimizer:
    """
    Several optimizers behind one interface (step / zero_grad / param_groups / state_dict), e.g. Muon + AdamW.
    """
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for opt in self.optimizers for g in opt.param_groups]

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {'combined': [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state):
        assert 'combined' in state and len(state['combined']) == len(self.optimizers), 'checkpoint optimizer type does not match --optimizer'
        for opt, s in zip(self.optimizers, state['combined']):
            opt.load_state_dict(s)
