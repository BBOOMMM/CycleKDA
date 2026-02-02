import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp, exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_kernel_o(
    q,
    k,  # 新加 k
    v,
    v_org,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    initial_t,  # 新增起始偏移
    T_cycle: tl.constexpr, # 周期 T
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    
    b_qk = tl.zeros([BT], dtype=tl.float32)      # 用于累加 qk 内积 [BT]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))   # 新增 k 的指针
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))  # 加载 k
        
        b_qk += tl.sum(b_q * b_k, 1) # 累加 qk 内积
        
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        if USE_EXP2:
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        else:
            b_qg = (b_q * exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))    # inter
            
    b_o *= scale
    b_qk *= scale   # qk 内积也需要应用 scale
    
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_v_org = tl.make_block_ptr(v_org + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))   # 新加
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_v_org = tl.load(p_v_org, boundary_check=(0, 1))    # 新加
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)                            # intra
    
    # 新加内部修正项
    offsets = tl.arange(0, BT)
    curr_t = initial_t + i_t * BT + offsets
    b_coeff = (curr_t % T_cycle).to(tl.float32) / T_cycle
    b_extra = (b_coeff * b_qk)[:, None] * b_v_org  # [BT] * [BT, BV] -> [BT, BV]
    b_o += b_extra
    
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    k: torch.Tensor,  # 新加 k
    v: torch.Tensor,
    v_org: torch.Tensor,  # 记录原始 v
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    initial_t: int = 0, # 新增
    T_cycle: int = 8,   # 新增
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_gla_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        v_org=v_org,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        initial_t=initial_t, # 传入起始偏移
        T_cycle=T_cycle,     # 传入周期
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
    )
    return o