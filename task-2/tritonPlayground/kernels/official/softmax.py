# In this tutorial, you will write a fused softmax operation that is significantly faster than PyTorch’s native op for a particular class of matrices: those whose rows can fit in the GPU’s SRAM.
# In doing so, you will learn about:
#     The benefits of kernel fusion for bandwidth-bound operations.
#     Reduction operators in Triton.

import torch
import triton
import triton.language as tl

from tritonPlayground.utils.device import *


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Suppose n_rows=10, row_step=4, and there are 4 programs (threads) in total.
        # Then:
        #   Thread 0: row_start=0, loop over rows 0, 4, 8
        #   Thread 1: row_start=1, loop over rows 1, 5, 9
        #   Thread 2: row_start=2, loop over rows 2, 6
        #   Thread 3: row_start=3, loop over rows 3, 7
        # num_stages -> pipeline

        # The stride represents how much we need to increase the pointer to advance 1 row
        # This marks the actual length of the row in memory, which may contain paddings
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
