# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import contextlib
from deepspeed.accelerator import get_accelerator


def is_compile_supported():
    return hasattr(torch, "compiler") and hasattr(torch.nn.Module, "compile")


def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func


@contextlib.contextmanager
def compiled_autograd(enabled, kwargs):
    try:
        if enabled:
            with torch._dynamo.compiled_autograd.enable(
                    torch.compile(backend=get_accelerator().get_compile_backend(), **kwargs)):
                yield
        else:
            yield
    finally:
        pass
