import torch
import tabulate
import triton

from tritonPlayground.kernels.official.dropout import (
    dropout_kernel,
    seeded_dropout_kernel,
)


def dropout_op(x: torch.Tensor, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    dropout_kernel[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


def seeded_dropout_op(x: torch.Tensor, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    seeded_dropout_kernel[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output
