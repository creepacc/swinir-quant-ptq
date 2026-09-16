from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


def _clone_arg(a):
    if torch.is_tensor(a):
        return a.detach()
    return a


class FeatureHook:
    """Collect per-module forward inputs / outputs (and optional retain_grad)."""

    def __init__(self, modules: Dict[str, nn.Module], retain_grad: bool = False):
        self.modules = modules
        self.retain_grad = retain_grad
        self.records: Dict[str, List[dict]] = {k: [] for k in modules}
        self._handles = []

    def _make(self, name: str):
        def fn(mod, inp, out):
            rec = {
                "input": tuple(_clone_arg(x) for x in inp),
                "output": out.detach() if torch.is_tensor(out) else out,
            }
            if self.retain_grad and torch.is_tensor(out):
                out.retain_grad()
                rec["out_ref"] = out
            self.records[name].append(rec)

        return fn

    def register(self):
        self.close()
        for name, m in self.modules.items():
            self._handles.append(m.register_forward_hook(self._make(name)))
        return self

    def clear(self):
        for k in self.records:
            self.records[k].clear()

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self.register()

    def __exit__(self, *exc):
        self.close()


class GradHook:
    def __init__(self, modules: Dict[str, nn.Module]):
        self.modules = modules
        self.grads: Dict[str, Optional[torch.Tensor]] = {k: None for k in modules}
        self._handles = []

    def _make(self, name: str):
        def fn(mod, gin, gout):
            g = gout[0] if isinstance(gout, tuple) else gout
            self.grads[name] = g.detach() if torch.is_tensor(g) else g

        return fn

    def register(self):
        self.close()
        for name, m in self.modules.items():
            self._handles.append(m.register_full_backward_hook(self._make(name)))
        return self

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self.register()

    def __exit__(self, *exc):
        self.close()
