import triton
import triton.language as tl

@triton.jit
def rmsnorm_fw_kernel(
    X_ptr, # ptr to input
    Y_ptr, # ptr to output
    W_ptr, # ptr to weight parameter Gamma
    rstd_ptr, # ptr to output reciprocal std dev (backward pass)
    stride_row, # distance between memory rows
    N_cols, # cols per row
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0) # map instance to row

    row_start_ptr = X_ptr + row_idx * stride_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_cols # computing memory offsets

    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32) # load x into SRAM
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32) # load w into SRAM

    x_sq = x*x
    mean_sq = tl.sum(x_sq, axis=0) / N_cols
    rstd = tl.math.rsqrt(mean_sq + eps) # computed RMS

    tl.store (rstd_ptr + row_idx, rstd) # write reciprocal std dev for backward pass

    y = x * rstd * w # normalize and scale

    y_row_start_ptr = Y_ptr + row_idx * stride_row
    tl.store(y_row_start_ptr + col_offsets, y, mask=mask) # store back to HBM
