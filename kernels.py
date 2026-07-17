import triton
import triton.language as tl

def early_config_prune(configs, named_args, **kwargs): # makes sure block size selected by autotuner is valid
    N_cols = named_args['N_cols']
    pruned = [cfg for cfg in configs if cfg.kwargs['BLOCK_SIZE'] >= N_cols]
    if len(pruned) == 0:
        max_cfg = max(configs, key=lambda c: c.kwargs['BLOCK_SIZE'])
        return [max_cfg]
    return pruned

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['N_cols'],
    prune_configs_by={
            'early_config_prune': early_config_prune
        }
) # triton autotuner for block size
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

@triton.jit
def rmsnorm_bw_kernel(
    dY_ptr,         # ptr to output gradients
    X_ptr,          # ptr to input features
    W_ptr,          # ptr to weights Gamma
    rstd_ptr,       # ptr to saved reciprocal std dev
    dX_ptr,         # ptr to output gradient
    dW_row_ptr,     # temp row-wise gradient buffer
    stride_row,     # distance between memory rows
    N_cols,         # cols per row
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_offset = row_idx * stride_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_cols

    # load from HBM into SRAM
    dy = tl.load(dY_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)

    x_hat = x * rstd

    dw_row = dy * x_hat
    tl.store(dW_row_ptr + row_offset + col_offsets, dw_row, mask=mask)

    dy_w_x = dy * w * x
    sum_dy_w_x = tl.sum(dy_w_x, axis=0)


    term1 = dy * w * rstd
    term2 = x * (rstd * rstd * rstd) * sum_dy_w_x / N_cols
    dx = term1 - term2

    tl.store(dX_ptr + row_offset + col_offsets, dx, mask=mask)
