import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, autotune_cuda_graph_kwargs

from einops import rearrange


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    initial_t: int = 0,
    T_cycle: int = 8,
    chunk_indices: torch.LongTensor | None = None,
    use_triton: bool = True,
    **kwargs,
):
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    assert q.shape == k.shape == g.shape == beta.shape, "q, k, g, beta must have the same shape."
    assert k.shape[-1] <= 256, "Currently we only support key headdim <=256 for KDA :-("
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_triton:
        o, final_state = ChunkKDAFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            use_gate_in_kernel,
            cu_seqlens,
            initial_t,
            T_cycle,
            chunk_indices,
        )
    else:
        o, final_state = ChunkKDAFunction_pytorch()(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            use_gate_in_kernel,
            cu_seqlens,
            initial_t,
            T_cycle,
            chunk_indices,
        )
    return o, final_state



class ChunkKDAFunction_pytorch(torch.nn.Module):
    @input_guard
    @autocast_custom_fwd
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        initial_t: int = 0,
        T_cycle: int = 8,
        chunk_indices: torch.LongTensor | None = None,
    ):
        g_org = None
        if use_gate_in_kernel:
            g_org = g
            g = kda_gate_fwd(g=g_org, A_log=A_log, dt_bias=dt_bias)

        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        chunk_size = 64
        C = chunk_size

        B, T, H, K = k.shape
        V = v.shape[-1]
        N = (T + C - 1) // C

        o = torch.empty((B, T, H, V), device=v.device, dtype=v.dtype)

        S = torch.zeros((B, H, K, V), device=k.device, dtype=torch.float32)
        if initial_state is not None:
            S = S + initial_state.to(torch.float32)

        q = q * scale

        for i in range(N):
            start = i * C
            end = min((i + 1) * C, T)
            L = end - start
            if L <= 0:
                continue

            q_i = q[:, start:end].permute(0, 2, 1, 3).to(torch.float32)      # [B,H,L,K]
            k_i = k[:, start:end].permute(0, 2, 1, 3).to(torch.float32)      # [B,H,L,K]
            v_i = v[:, start:end].permute(0, 2, 1, 3).to(torch.float32)      # [B,H,L,V]
            g_i = g[:, start:end].permute(0, 2, 1, 3).to(torch.float32)      # [B,H,L,K]
            beta_i = beta[:, start:end].permute(0, 2, 1, 3).to(torch.float32)   # [B,H,L,K]

            cur_t_start = initial_t + start
            t_math = cur_t_start + torch.arange(L, device=q.device)
            mod_t = t_math % T_cycle

            rho = torch.where(mod_t == 0, 1.0, mod_t.to(torch.float32) / T_cycle)  # [L]
            rho = rho.view(1, 1, L, 1)  # [1,1,L,1]
            beta_i = beta_i * rho

            alpha_i = torch.exp(g_i)

            update_indices = torch.nonzero(mod_t == 0, as_tuple=True)[0].tolist()

            S_curr = S
            last_update_j = -1

            def compute_o_segment(start_idx: int, end_idx: int):
                if start_idx > end_idx:
                    return
                q_seg = q_i[:, :, start_idx : end_idx + 1]   # [B,H,S,K]
                k_seg = k_i[:, :, start_idx : end_idx + 1]   # [B,H,S,K]
                v_seg = v_i[:, :, start_idx : end_idx + 1]   # [B,H,S,V]
                a_seg = alpha_i[:, :, start_idx : end_idx + 1] # [B,H,S,K]
                b_seg = beta_i[:, :, start_idx : end_idx + 1]  # [B,H,S,K]

                
                c_seg = (torch.einsum("bhlk,bhlk->bhl", q_seg, k_seg*b_seg)).unsqueeze(-1)  # [B,H,S,1]

                aq_seg = a_seg * q_seg  # [B,H,S,K]
                term1 = torch.einsum("bhkv,bhlk->bhlv", S_curr, aq_seg)

                ak_seg = a_seg * k_seg
                S_ak = torch.einsum("bhkv,bhlk->bhlv", S_curr, ak_seg)
                term2 = c_seg * S_ak

                term3 = c_seg * v_seg
                out_seg = term1 - term2 + term3  # [B,H,S,V]

                o[:, start + start_idx : start + end_idx + 1] = out_seg.permute(0, 2, 1, 3).to(v.dtype)

            for j in update_indices:
                compute_o_segment(last_update_j + 1, j)
                last_update_j = j

                if cur_t_start + j == 0:
                    continue

                a_j = alpha_i[:, :, j, :].unsqueeze(-1)   # [B,H,K,1]
                k_j = k_i[:, :, j, :]                     # [B,H,K]
                v_j = v_i[:, :, j, :]                     # [B,H,V]
                b_j = beta_i[:, :, j, :]                  # [B,H,K]

                a_S = a_j * S_curr
                k_a_S = torch.einsum("bhk,bhkv->bhv", k_j, a_S)
                b_k = b_j * k_j
                S_curr = a_S - torch.einsum("bhk,bhv->bhkv", b_k, k_a_S) + torch.einsum("bhk,bhv->bhkv", b_k, v_j)

            compute_o_segment(last_update_j + 1, L - 1)
            S = S_curr

        if output_final_state:
            return o, S
        return o, None


class ChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        initial_t: int = 0,
        T_cycle: int = 8,
        chunk_indices: torch.LongTensor | None = None,
    ):
        g_org = None
        if use_gate_in_kernel:
            g_org = g
            g = kda_gate_fwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
            )
        
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        
        chunk_size = 64
                
        o, Aqk, final_state, chunk_states, intra_states = chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            initial_t=initial_t,
            T_cycle=T_cycle,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )
        
        
        if use_gate_in_kernel:
            g = None
        
        
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, g_org, beta, A_log, dt_bias, Aqk, initial_state, cu_seqlens, chunk_indices, chunk_states, intra_states,
        )
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.initial_t = initial_t
        ctx.T_cycle = T_cycle
        return o.to(q.dtype), final_state
    
    
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor = None):
        (q, q_rstd, k, k_rstd, v, g, g_org, beta, A_log, dt_bias, 
         Aqk, initial_state, cu_seqlens, chunk_indices, chunk_states, intra_states, 
        ) = ctx.saved_tensors
        
        if ctx.use_gate_in_kernel:
            g = kda_gate_fwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
            )
            g = chunk_local_cumsum(
                g=g,
                chunk_size=ctx.chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices
            )
        
        dq, dk, dv, dg, dbeta, d_initial_state = chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            do,
            Aqk,
            initial_state,
            chunk_states,
            intra_states,
            dht,
            scale=ctx.scale,
            initial_t=ctx.initial_t,
            T_cycle=ctx.T_cycle, 
            chunk_size=ctx.chunk_size
        )
        
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        dA_log, ddt_bias = None, None
        if ctx.use_gate_in_kernel:
            dg, dA, dbias = kda_gate_bwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                dyg=dg,
            )
            dA = dA.to(A_log)
            if dt_bias is not None:
                dbias = dbias.to(dt_bias)

        return (
            dq, dk, dv, dg, dbeta, dA_log, ddt_bias, None, d_initial_state,
            None, None, None, None, None, None, None
        )



def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    initial_t: int = 0,
    T_cycle: int = 8,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    N = (T + C - 1) // C
    M = C // T_cycle + 1  # 每个 chunk 内最多的状态跳变次数

    o = torch.empty_like(v)
    Aqk = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
    
    chunk_states = torch.empty((B, H, N, K, V), device=q.device, dtype=torch.float32)
    intra_states = torch.empty((B, H, N, M, K, V), device=q.device, dtype=torch.float32)
    
    final_state = None
    if output_final_state:
        final_state = torch.empty((B, H, K, V), device=q.device, dtype=torch.float32)

    BK = triton.next_power_of_2(K)
    grid_state = lambda META: (B * H, triton.cdiv(V, META['BV']))

    if T_cycle == 1:
        chunk_kda_fwd_state_kernel_t1[grid_state](
            k,
            v,
            g,
            beta,
            chunk_states,
            initial_state,
            final_state,
            initial_t,
            N,
            C,
            B,
            T,
            H,
            K,
            V,
            BK,
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
        )
    else:
        chunk_kda_fwd_state_kernel[grid_state](
            k,
            v,
            g,
            beta,
            chunk_states,
            initial_state,
            final_state,
            initial_t,
            T_cycle,
            N,
            C,
            B,
            T,
            H,
            K,
            V,
            BK,
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
        )

    grid_output = lambda META: (N, B * H, triton.cdiv(V, META['BV']))
    chunk_kda_fwd_output_kernel[grid_output](
        q,
        k,
        v,
        g,
        beta,
        chunk_states,
        o,
        Aqk,
        intra_states,
        scale,
        initial_t,
        T_cycle,
        N,
        C,
        B,
        T,
        H,
        K,
        V,
        BK,
        M,
    )

    return o, Aqk, final_state, chunk_states, intra_states


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2]
    ],
    key=['H', 'K', 'V', 'C'],
    **autotune_cache_kwargs,
    **autotune_cuda_graph_kwargs,
)
@triton.jit
def chunk_kda_fwd_state_kernel(
    k,              # [B, T, H, K]
    v,              # [B, T, H, V]
    g,              # [B, T, H, K]
    beta,           # [B, T, H, K]
    chunk_states,   # [B, H, N, K, V]
    initial_state,  # [B, H, K, V]
    final_state,
    initial_t,
    T_cycle: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BV: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    b_S = tl.zeros([BK, BV], dtype=tl.float32)
    mask_h = (o_k < K)[:, None] & (o_v < V)[None, :]
    
    if USE_INITIAL_STATE:
        p_h0 = initial_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        b_S += tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for i_n in range(N):
        p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_cs, b_S, mask=(o_k < K)[:, None] & (o_v < V)[None, :])

        start_t = initial_t + i_n * C
        rem = start_t % T_cycle
        j = (T_cycle - rem) if rem != 0 else 0
        if start_t == 0 and j == 0:
            j += T_cycle
        
        while j < C:
            token_idx = i_n * C + j
            offset = i_b * T * H + token_idx * H + i_h

            p_k = k + offset * K + o_k
            p_v = v + offset * V + o_v
            p_g = g + offset * K + o_k
            p_b = beta + offset * K + o_k

            mask_token = token_idx < T
            mask_k = o_k < K
            mask_v = o_v < V

            # token 越界时：u_k/u_v/u_beta=0，u_alpha=1 (通过 g 的 other=0 达成)
            u_k = tl.load(p_k, mask=mask_k & mask_token, other=0.0).to(tl.float32)
            u_v = tl.load(p_v, mask=mask_v & mask_token, other=0.0).to(tl.float32)
            u_alpha = tl.exp(tl.load(p_g, mask=mask_k & mask_token, other=0).to(tl.float32))
            u_beta = tl.load(p_b, mask=mask_k & mask_token, other=0.0).to(tl.float32)

            a_S = u_alpha[:, None] * b_S
            k_a_S = tl.sum(u_k[:, None] * a_S, axis=0)
            b_k = u_beta * u_k
            b_S = a_S - b_k[:, None] * k_a_S[None, :] + b_k[:, None] * u_v[None, :]

            j += T_cycle

    if STORE_FINAL_STATE:
        p_ht = final_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_S, mask=mask_h)


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2]
    ],
    key=['H', 'K', 'V', 'C'],
    **autotune_cache_kwargs,
    **autotune_cuda_graph_kwargs,
)
@triton.jit
def chunk_kda_fwd_state_kernel_t1(
    k,              # [B, T, H, K]
    v,              # [B, T, H, V]
    g,              # [B, T, H, K]
    beta,           # [B, T, H, K]
    chunk_states,   # [B, H, N, K, V]
    initial_state,  # [B, H, K, V]
    final_state,
    initial_t,
    N: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BV: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    b_S = tl.zeros([BK, BV], dtype=tl.float32)
    mask_h = (o_k < K)[:, None] & (o_v < V)[None, :]

    if USE_INITIAL_STATE:
        p_h0 = initial_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        b_S += tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for i_n in range(N):
        p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_cs, b_S, mask=mask_h)

        start_t = initial_t + i_n * C
        if start_t == 0:
            for j in range(1, C):
                token_idx = i_n * C + j
                offset = i_b * T * H + token_idx * H + i_h

                p_k = k + offset * K + o_k
                p_v = v + offset * V + o_v
                p_g = g + offset * K + o_k
                p_b = beta + offset * K + o_k

                mask_token = token_idx < T
                mask_k = o_k < K
                mask_v = o_v < V

                u_k = tl.load(p_k, mask=mask_k & mask_token, other=0.0).to(tl.float32)
                u_v = tl.load(p_v, mask=mask_v & mask_token, other=0.0).to(tl.float32)
                u_alpha = tl.exp(tl.load(p_g, mask=mask_k & mask_token, other=0).to(tl.float32))
                u_beta = tl.load(p_b, mask=mask_k & mask_token, other=0.0).to(tl.float32)

                a_S = u_alpha[:, None] * b_S
                k_a_S = tl.sum(u_k[:, None] * a_S, axis=0)
                b_k = u_beta * u_k
                b_S = a_S - b_k[:, None] * k_a_S[None, :] + b_k[:, None] * u_v[None, :]
        else:
            for j in range(C):
                token_idx = i_n * C + j
                offset = i_b * T * H + token_idx * H + i_h

                p_k = k + offset * K + o_k
                p_v = v + offset * V + o_v
                p_g = g + offset * K + o_k
                p_b = beta + offset * K + o_k

                mask_token = token_idx < T
                mask_k = o_k < K
                mask_v = o_v < V

                u_k = tl.load(p_k, mask=mask_k & mask_token, other=0.0).to(tl.float32)
                u_v = tl.load(p_v, mask=mask_v & mask_token, other=0.0).to(tl.float32)
                u_alpha = tl.exp(tl.load(p_g, mask=mask_k & mask_token, other=0).to(tl.float32))
                u_beta = tl.load(p_b, mask=mask_k & mask_token, other=0.0).to(tl.float32)

                a_S = u_alpha[:, None] * b_S
                k_a_S = tl.sum(u_k[:, None] * a_S, axis=0)
                b_k = u_beta * u_k
                b_S = a_S - b_k[:, None] * k_a_S[None, :] + b_k[:, None] * u_v[None, :]

    if STORE_FINAL_STATE:
        p_ht = final_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_S, mask=mask_h)


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in[1, 2]
    ],
    key=['H', 'K', 'V', 'C'],
    **autotune_cache_kwargs,
    **autotune_cuda_graph_kwargs,
)
@triton.jit
def chunk_kda_fwd_output_kernel(
    q, k, v, g, beta,
    chunk_states,
    o,
    Aqk,
    intra_states,
    scale,
    initial_t,
    T_cycle: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    M: tl.constexpr,
    BV: tl.constexpr
):
    i_n = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_v = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, C)

    p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
    b_S = tl.load(p_cs, mask=(o_k < K)[:, None] & (o_v < V)[None, :], other=0.0).to(tl.float32)

    t = i_n * C + o_c
    mask_t = t < T

    base_bth = (i_b * T * H)
    base_tokens = base_bth + t[:, None] * H + i_h

    p_q = q + base_tokens * K + o_k[None, :]
    p_k = k + base_tokens * K + o_k[None, :]
    p_v = v + base_tokens * V + o_v[None, :]
    p_g = g + base_tokens * K + o_k[None, :]
    p_b = beta + base_tokens * K + o_k[None, :]
    
    mask_tk = mask_t[:, None] & (o_k < K)[None, :]
    mask_tv = mask_t[:, None] & (o_v < V)[None, :]

    b_q = tl.load(p_q, mask=mask_tk, other=0.0).to(tl.float32) * scale    # [C,BK]
    b_k = tl.load(p_k, mask=mask_tk, other=0.0).to(tl.float32)            # [C,BK]
    b_v = tl.load(p_v, mask=mask_tv, other=0.0).to(tl.float32)            # [C,BV]
    b_g = tl.load(p_g, mask=mask_tk, other=-float("inf")).to(tl.float32)  # [C,BK]
    b_beta = tl.load(p_b, mask=mask_tk, other=0.0).to(tl.float32)         # [C,BK]

    b_alpha = tl.exp(b_g)

    cur_t_global = initial_t + i_n * C + o_c
    mod_t = cur_t_global % T_cycle
    rho = tl.where(mod_t == 0, 1.0, mod_t.to(tl.float32) / T_cycle)  # [C]
    b_beta_tilde = b_beta * rho[:, None]   # [C,BK]

    b_o = tl.zeros([C, BV], dtype=tl.float32)

    start_t = initial_t + i_n * C
    rem = start_t % T_cycle
    j = (T_cycle - rem) if rem != 0 else 0
    if start_t == 0 and j == 0:
        j += T_cycle

    last_update_j = -1
    step = 0

    while j < C:        
        mask_seg = (o_c > last_update_j) & (o_c <= j) & mask_t

        q_seg = tl.where(mask_seg[:, None], b_q, 0.0)
        k_seg = tl.where(mask_seg[:, None], b_k, 0.0)
        v_seg = tl.where(mask_seg[:, None], b_v, 0.0)
        a_seg = tl.where(mask_seg[:, None], b_alpha, 0.0)
        b_seg = tl.where(mask_seg[:, None], b_beta_tilde, 0.0)

        aq_seg = a_seg * q_seg
        ak_seg = a_seg * k_seg
        c_seg_val = tl.sum(q_seg * k_seg * b_seg, axis=1)   # [C]

        # 所有的 BV block 算出来的 Aqk 都是一致的(仅与q, k, beta有关)
        # 所以只需要 i_v == 0 的 block 执行写回一次即可，防止冗余的写入开销
        if i_v == 0:
            p_aqk = Aqk + (base_bth + t * H + i_h)
            tl.store(p_aqk, c_seg_val, mask=mask_seg)

        term1 = tl.dot(aq_seg, b_S)
        S_ak = tl.dot(ak_seg, b_S)
        b_o += term1 - (c_seg_val[:, None] * S_ak) + (c_seg_val[:, None] * v_seg)

        # 状态跳变更新
        token_idx = i_n * C + j
        mask_u_t = token_idx < T

        offset = base_bth + token_idx * H + i_h

        u_k = tl.load(k + offset * K + o_k, mask=(o_k < K) & mask_u_t, other=0.0).to(tl.float32)
        u_v = tl.load(v + offset * V + o_v, mask=(o_v < V) & mask_u_t, other=0.0).to(tl.float32)
        u_alpha = tl.exp(tl.load(g + offset * K + o_k, mask=(o_k < K) & mask_u_t, other=0).to(tl.float32))
        u_beta = tl.load(beta + offset * K + o_k, mask=(o_k < K) & mask_u_t, other=0.0).to(tl.float32)

        a_S = u_alpha[:, None] * b_S
        k_a_S = tl.sum(u_k[:, None] * a_S, axis=0)
        b_k_u = u_beta * u_k

        b_S = a_S - b_k_u[:, None] * k_a_S[None, :] + b_k_u[:, None] * u_v[None, :]
        
        # intra_states [B, H, N, M, K, V] 存储每个 chunk 内每次跳变后的状态
        p_intra = intra_states + (i_bh * N + i_n) * M * K * V + step * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_intra, b_S, mask=(o_k < K)[:, None] & (o_v < V)[None, :])

        last_update_j = j
        j += T_cycle
        step += 1

    if last_update_j < C - 1:
        mask_seg = (o_c > last_update_j) & mask_t

        q_seg = tl.where(mask_seg[:, None], b_q, 0.0)
        k_seg = tl.where(mask_seg[:, None], b_k, 0.0)
        v_seg = tl.where(mask_seg[:, None], b_v, 0.0)
        a_seg = tl.where(mask_seg[:, None], b_alpha, 0.0)
        b_seg = tl.where(mask_seg[:, None], b_beta_tilde, 0.0)

        aq_seg = a_seg * q_seg
        ak_seg = a_seg * k_seg
        c_seg_val = tl.sum(q_seg * k_seg * b_seg, axis=1)

        # 尾部同理，只用 i_v == 0 的 block 保存
        if i_v == 0:
            p_aqk = Aqk + (base_bth + t * H + i_h)
            tl.store(p_aqk, c_seg_val, mask=mask_seg)

        term1 = tl.dot(aq_seg, b_S)
        S_ak = tl.dot(ak_seg, b_S)
        b_o += term1 - (c_seg_val[:, None] * S_ak) + (c_seg_val[:, None] * v_seg)

    p_o = o + (base_bth + t[:, None] * H + i_h) * V + o_v[None, :]
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_t[:, None] & (o_v < V)[None, :])




def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    Aqk: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_states: torch.Tensor,
    intra_states: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    initial_t: int,
    T_cycle: int,
    chunk_size: int,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    N = (T + C - 1) // C
    M = C // T_cycle + 1

    # 初始化为全零，用于使用 atomic_add
    dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dg = torch.zeros(g.shape, device=g.device, dtype=torch.float32)
    dbeta = torch.zeros(beta.shape, device=beta.device, dtype=torch.float32)
    
    # dv 和 d_initial_state 都是各 block 独立写入的，可以用 empty 
    dv = torch.empty(v.shape, device=v.device, dtype=torch.float32)
    d_initial_state = torch.empty((B, H, K, V), device=q.device, dtype=torch.float32) if initial_state is not None else None

    BK = triton.next_power_of_2(K)
    grid = lambda META: (B * H, triton.cdiv(V, META['BV']))
    if T_cycle == 1:
        chunk_kda_bwd_kernel_t1[grid](
            q, k, v, g, beta, do, Aqk,
            chunk_states, intra_states,
            dq, dk, dv, dg, dbeta, dht, d_initial_state,
            scale, initial_t, N, C, B, T, H, K, V, BK, M,
            LOAD_FINAL_STATE=(dht is not None),
            HAS_INITIAL_STATE=(initial_state is not None),
        )
    else:
        chunk_kda_bwd_kernel[grid](
            q, k, v, g, beta, do, Aqk,
            chunk_states, intra_states,
            dq, dk, dv, dg, dbeta, dht, d_initial_state,
            scale, initial_t, T_cycle, N, C, B, T, H, K, V, BK, M,
            LOAD_FINAL_STATE=(dht is not None),
            HAS_INITIAL_STATE=(initial_state is not None),
        )
    
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g.dtype), dbeta.to(beta.dtype), d_initial_state



@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in[16, 32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2]
    ],
    key=['H', 'K', 'V', 'C'],
    reset_to_zero=['dq', 'dk', 'dg', 'dbeta'],   # 在测试每个配置前，将 output_ptr 指向的内存清零
    **autotune_cache_kwargs,
    **autotune_cuda_graph_kwargs,
)
@triton.jit
def chunk_kda_bwd_kernel(
    q, k, v, g, beta, do, Aqk,
    chunk_states, intra_states,
    dq, dk, dv, dg, dbeta, dht, d_initial_state,
    scale, initial_t, 
    T_cycle: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr, 
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    M: tl.constexpr,
    LOAD_FINAL_STATE: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    BV: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, C)
    
    mask_kv = (o_k < K)[:, None] & (o_v < V)[None, :]

    d_b_S = tl.zeros([BK, BV], dtype=tl.float32)
    if LOAD_FINAL_STATE:
        p_dht = dht + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        d_b_S = tl.load(p_dht, mask=mask_kv, other=0.0).to(tl.float32)

    for i_n in range(N - 1, -1, -1):
        t = i_n * C + o_c
        mask_t = t < T
        base_bth = i_b * T * H
        base_tokens = base_bth + t[:, None] * H + i_h
        
        p_q = q + base_tokens * K + o_k[None, :]
        p_k = k + base_tokens * K + o_k[None, :]
        p_v = v + base_tokens * V + o_v[None, :]
        p_g = g + base_tokens * K + o_k[None, :]
        p_b = beta + base_tokens * K + o_k[None, :]
        p_do = do + base_tokens * V + o_v[None, :]
        p_Aqk = Aqk + (base_bth + t * H + i_h)
        
        mask_tk = mask_t[:, None] & (o_k < K)[None, :]
        mask_tv = mask_t[:, None] & (o_v < V)[None, :]

        b_q = tl.load(p_q, mask=mask_tk, other=0.0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_tk, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_tv, other=0.0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask_tk, other=-float("inf")).to(tl.float32) # 恢复为-inf对齐前向
        b_beta = tl.load(p_b, mask=mask_tk, other=0.0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_tv, other=0.0).to(tl.float32)
        b_Aqk = tl.load(p_Aqk, mask=mask_t, other=0.0).to(tl.float32)
        b_alpha = tl.exp(b_g)
        
        cur_t_global = initial_t + i_n * C + o_c
        mod_t = cur_t_global % T_cycle
        rho = tl.where(mod_t == 0, 1.0, mod_t.to(tl.float32) / T_cycle)
        b_beta_tilde = b_beta * rho[:, None]

        b_dq = tl.zeros([C, BK], dtype=tl.float32)
        b_dk = tl.zeros([C, BK], dtype=tl.float32)
        b_dv = tl.zeros([C, BV], dtype=tl.float32)
        b_dg = tl.zeros([C, BK], dtype=tl.float32)
        b_dbeta = tl.zeros([C, BK], dtype=tl.float32)

        start_t = initial_t + i_n * C
        rem = start_t % T_cycle
        first_j = (T_cycle - rem) if rem != 0 else 0
        if start_t == 0 and first_j == 0:
            first_j += T_cycle
            
        num_updates = 0
        j = first_j
        last_update_j = -1
        while j < C:
            last_update_j = j
            num_updates += 1
            j += T_cycle

        step = num_updates

        if last_update_j < C - 1:
            if step == 0:
                p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
                b_S = tl.load(p_cs, mask=mask_kv, other=0.0).to(tl.float32)
            else:
                p_intra = intra_states + (i_bh * N + i_n) * M * K * V + (step - 1) * K * V + o_k[:, None] * V + o_v[None, :]
                b_S = tl.load(p_intra, mask=mask_kv, other=0.0).to(tl.float32)

            mask_seg = (o_c > last_update_j) & mask_t

            q_seg = tl.where(mask_seg[:, None], b_q, 0.0)
            k_seg = tl.where(mask_seg[:, None], b_k, 0.0)
            v_seg = tl.where(mask_seg[:, None], b_v, 0.0)
            a_seg = tl.where(mask_seg[:, None], b_alpha, 0.0)
            b_seg = tl.where(mask_seg[:, None], b_beta_tilde, 0.0)     
            do_seg = tl.where(mask_seg[:, None], b_do, 0.0)
            c_seg_val = tl.where(mask_seg, b_Aqk, 0.0)

            aq_seg = a_seg * q_seg
            ak_seg = a_seg * k_seg
            bk_seg = b_seg * k_seg
            S_ak = tl.dot(ak_seg, b_S)

            d_c_seg_val = tl.sum(do_seg * (v_seg - S_ak), axis=1)
            d_S_ak = -c_seg_val[:, None] * do_seg

            d_aq_seg = tl.dot(do_seg, tl.trans(b_S))
            d_b_S_1 = tl.dot(tl.trans(aq_seg), do_seg)
            d_ak_seg = tl.dot(d_S_ak, tl.trans(b_S))
            d_b_S_2 = tl.dot(tl.trans(ak_seg), d_S_ak)
            
            b_dq += (d_c_seg_val[:, None] * bk_seg + d_aq_seg * a_seg) * scale
            b_dk += (d_c_seg_val[:, None] * q_seg * b_seg + d_ak_seg * a_seg)
            b_dv += c_seg_val[:, None] * do_seg
            b_dg += (d_aq_seg * q_seg + d_ak_seg * k_seg) * a_seg
            b_dbeta += d_c_seg_val[:, None] * q_seg * k_seg * rho[:, None]

            d_b_S += d_b_S_1 + d_b_S_2

        j = last_update_j
        step = num_updates - 1
        
        while step >= 0:
            prev_j = j - T_cycle
            if step == 0:
                prev_j = -1
            
            if step == 0:
                p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
                b_S = tl.load(p_cs, mask=mask_kv, other=0.0).to(tl.float32)
            else:
                p_intra = intra_states + (i_bh * N + i_n) * M * K * V + (step - 1) * K * V + o_k[:, None] * V + o_v[None, :]
                b_S = tl.load(p_intra, mask=mask_kv, other=0.0).to(tl.float32)

            mask_j = (o_c == j) & mask_t
            token_idx = i_n * C + j

            if token_idx < T:
                offset = base_bth + token_idx * H + i_h
                u_k = tl.load(k + offset * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32)
                u_v = tl.load(v + offset * V + o_v, mask=(o_v < V), other=0.0).to(tl.float32)
                u_alpha = tl.exp(tl.load(g + offset * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32)) # 越界为0保持状态不丢失
                u_beta = tl.load(beta + offset * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32)

                a_S = u_alpha[:, None] * b_S
                k_a_S = tl.sum(u_k[:, None] * a_S, axis=0)
                b_k_u = u_beta * u_k

                d_a_S = d_b_S + 0.0
                d_b_k_u = tl.sum(d_b_S * (u_v[None, :] - k_a_S[None, :]), axis=1)
                d_k_a_S = tl.sum(d_b_S * (-b_k_u[:, None]), axis=0)
                d_u_v = tl.sum(d_b_S * b_k_u[:, None], axis=0)

                d_u_k_1 = tl.sum(d_k_a_S[None, :] * a_S, axis=1)
                d_a_S += d_k_a_S[None, :] * u_k[:, None]

                d_u_beta = d_b_k_u * u_k
                d_u_k_2 = d_b_k_u * u_beta

                d_u_alpha = tl.sum(d_a_S * b_S, axis=1)
                d_b_S = d_a_S * u_alpha[:, None]

                d_u_k = d_u_k_1 + d_u_k_2
                d_u_g = d_u_alpha * u_alpha
                
                b_dk += tl.where(mask_j[:, None], d_u_k[None, :], 0.0)
                b_dv += tl.where(mask_j[:, None], d_u_v[None, :], 0.0)
                b_dg += tl.where(mask_j[:, None], d_u_g[None, :], 0.0)
                b_dbeta += tl.where(mask_j[:, None], d_u_beta[None, :], 0.0)

            mask_seg = (o_c > prev_j) & (o_c <= j) & mask_t

            q_seg = tl.where(mask_seg[:, None], b_q, 0.0)
            k_seg = tl.where(mask_seg[:, None], b_k, 0.0)
            v_seg = tl.where(mask_seg[:, None], b_v, 0.0)
            a_seg = tl.where(mask_seg[:, None], b_alpha, 0.0)
            b_seg = tl.where(mask_seg[:, None], b_beta_tilde, 0.0)
            do_seg = tl.where(mask_seg[:, None], b_do, 0.0)
            c_seg_val = tl.where(mask_seg, b_Aqk, 0.0)

            aq_seg = a_seg * q_seg
            ak_seg = a_seg * k_seg
            bk_seg = b_seg * k_seg
            S_ak = tl.dot(ak_seg, b_S)

            d_c_seg_val = tl.sum(do_seg * (v_seg - S_ak), axis=1)
            d_S_ak = -c_seg_val[:, None] * do_seg

            d_aq_seg = tl.dot(do_seg, tl.trans(b_S))
            d_b_S_1 = tl.dot(tl.trans(aq_seg), do_seg)
            d_ak_seg = tl.dot(d_S_ak, tl.trans(b_S))
            d_b_S_2 = tl.dot(tl.trans(ak_seg), d_S_ak)
            
            b_dq += (d_c_seg_val[:, None] * bk_seg + d_aq_seg * a_seg) * scale
            b_dk += (d_c_seg_val[:, None] * q_seg * b_seg + d_ak_seg * a_seg)
            b_dv += c_seg_val[:, None] * do_seg
            b_dg += (d_aq_seg * q_seg + d_ak_seg * k_seg) * a_seg
            b_dbeta += d_c_seg_val[:, None] * q_seg * k_seg * rho[:, None]
            
            d_b_S += d_b_S_1 + d_b_S_2

            j -= T_cycle
            step -= 1

        p_dq = dq + base_tokens * K + o_k[None, :]
        p_dk = dk + base_tokens * K + o_k[None, :]
        p_dv = dv + base_tokens * V + o_v[None, :]
        p_dg = dg + base_tokens * K + o_k[None, :]
        p_dbeta = dbeta + base_tokens * K + o_k[None, :]
        
        # 共享张量用 atomic_add, 读取目标地址当前的旧值, 加上累加的增量值, 再写回去
        tl.atomic_add(p_dq, b_dq, mask=mask_tk)
        tl.atomic_add(p_dk, b_dk, mask=mask_tk)
        tl.store(p_dv, b_dv, mask=mask_tv)  # 各管各的 V 切片, 用 store
        tl.atomic_add(p_dg, b_dg, mask=mask_tk)
        tl.atomic_add(p_dbeta, b_dbeta, mask=mask_tk)

    if HAS_INITIAL_STATE:
        p_d_init = d_initial_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_d_init, d_b_S, mask=mask_kv)


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in[16, 32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2]
    ],
    key=['H', 'K', 'V', 'C'],
    reset_to_zero=['dq', 'dk', 'dg', 'dbeta'],
    **autotune_cache_kwargs,
    **autotune_cuda_graph_kwargs,
)
@triton.jit
def chunk_kda_bwd_kernel_t1(
    q, k, v, g, beta, do, Aqk,
    chunk_states, intra_states,
    dq, dk, dv, dg, dbeta, dht, d_initial_state,
    scale, initial_t,
    N: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    M: tl.constexpr,
    LOAD_FINAL_STATE: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    BV: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, C)

    mask_kv = (o_k < K)[:, None] & (o_v < V)[None, :]

    d_b_S = tl.zeros([BK, BV], dtype=tl.float32)
    if LOAD_FINAL_STATE:
        p_dht = dht + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        d_b_S = tl.load(p_dht, mask=mask_kv, other=0.0).to(tl.float32)

    for i_n in range(N - 1, -1, -1):
        t = i_n * C + o_c
        mask_t = t < T
        base_bth = i_b * T * H
        base_tokens = base_bth + t[:, None] * H + i_h

        p_q = q + base_tokens * K + o_k[None, :]
        p_k = k + base_tokens * K + o_k[None, :]
        p_v = v + base_tokens * V + o_v[None, :]
        p_g = g + base_tokens * K + o_k[None, :]
        p_b = beta + base_tokens * K + o_k[None, :]
        p_do = do + base_tokens * V + o_v[None, :]
        p_Aqk = Aqk + (base_bth + t * H + i_h)

        mask_tk = mask_t[:, None] & (o_k < K)[None, :]
        mask_tv = mask_t[:, None] & (o_v < V)[None, :]

        b_q = tl.load(p_q, mask=mask_tk, other=0.0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_tk, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_tv, other=0.0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask_tk, other=-float("inf")).to(tl.float32)
        b_beta = tl.load(p_b, mask=mask_tk, other=0.0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_tv, other=0.0).to(tl.float32)
        b_Aqk = tl.load(p_Aqk, mask=mask_t, other=0.0).to(tl.float32)
        b_alpha = tl.exp(b_g)

        b_dq = tl.zeros([C, BK], dtype=tl.float32)
        b_dk = tl.zeros([C, BK], dtype=tl.float32)
        b_dv = tl.zeros([C, BV], dtype=tl.float32)
        b_dg = tl.zeros([C, BK], dtype=tl.float32)
        b_dbeta = tl.zeros([C, BK], dtype=tl.float32)

        start_t = initial_t + i_n * C

        if start_t == 0:
            if C == 1:
                p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
                b_S = tl.load(p_cs, mask=mask_kv, other=0.0).to(tl.float32)

                mask_seg = mask_t
                q_seg = tl.where(mask_seg[:, None], b_q, 0.0)
                k_seg = tl.where(mask_seg[:, None], b_k, 0.0)
                v_seg = tl.where(mask_seg[:, None], b_v, 0.0)
                a_seg = tl.where(mask_seg[:, None], b_alpha, 0.0)
                b_seg = tl.where(mask_seg[:, None], b_beta, 0.0)
                do_seg = tl.where(mask_seg[:, None], b_do, 0.0)
                c_seg_val = tl.where(mask_seg, b_Aqk, 0.0)

                aq_seg = a_seg * q_seg
                ak_seg = a_seg * k_seg
                bk_seg = b_seg * k_seg
                S_ak = tl.dot(ak_seg, b_S)

                d_c_seg_val = tl.sum(do_seg * (v_seg - S_ak), axis=1)
                d_S_ak = -c_seg_val[:, None] * do_seg

                d_aq_seg = tl.dot(do_seg, tl.trans(b_S))
                d_b_S_1 = tl.dot(tl.trans(aq_seg), do_seg)
                d_ak_seg = tl.dot(d_S_ak, tl.trans(b_S))
                d_b_S_2 = tl.dot(tl.trans(ak_seg), d_S_ak)

                b_dq += (d_c_seg_val[:, None] * bk_seg + d_aq_seg * a_seg) * scale
                b_dk += (d_c_seg_val[:, None] * q_seg * b_seg + d_ak_seg * a_seg)
                b_dv += c_seg_val[:, None] * do_seg
                b_dg += (d_aq_seg * q_seg + d_ak_seg * k_seg) * a_seg
                b_dbeta += d_c_seg_val[:, None] * q_seg * k_seg

                d_b_S += d_b_S_1 + d_b_S_2
            else:
                for j in range(C - 1, 0, -1):
                    if j == 1:
                        p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
                        b_S = tl.load(p_cs, mask=mask_kv, other=0.0).to(tl.float32)
                        prev_j = -1
                    else:
                        p_intra = intra_states + (i_bh * N + i_n) * M * K * V + (j - 2) * K * V + o_k[:, None] * V + o_v[None, :]
                        b_S = tl.load(p_intra, mask=mask_kv, other=0.0).to(tl.float32)
                        prev_j = j - 1

                    mask_j = (o_c == j) & mask_t
                    token_idx = i_n * C + j

                    if token_idx < T:
                        offset = base_bth + token_idx * H + i_h
                        u_k = tl.load(k + offset * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32)
                        u_v = tl.load(v + offset * V + o_v, mask=(o_v < V), other=0.0).to(tl.float32)
                        u_alpha = tl.exp(tl.load(g + offset * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32))
                        u_beta = tl.load(beta + offset * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32)

                        a_S = u_alpha[:, None] * b_S
                        k_a_S = tl.sum(u_k[:, None] * a_S, axis=0)
                        b_k_u = u_beta * u_k

                        d_a_S = d_b_S + 0.0
                        d_b_k_u = tl.sum(d_b_S * (u_v[None, :] - k_a_S[None, :]), axis=1)
                        d_k_a_S = tl.sum(d_b_S * (-b_k_u[:, None]), axis=0)
                        d_u_v = tl.sum(d_b_S * b_k_u[:, None], axis=0)

                        d_u_k_1 = tl.sum(d_k_a_S[None, :] * a_S, axis=1)
                        d_a_S += d_k_a_S[None, :] * u_k[:, None]

                        d_u_beta = d_b_k_u * u_k
                        d_u_k_2 = d_b_k_u * u_beta

                        d_u_alpha = tl.sum(d_a_S * b_S, axis=1)
                        d_b_S = d_a_S * u_alpha[:, None]

                        d_u_k = d_u_k_1 + d_u_k_2
                        d_u_g = d_u_alpha * u_alpha

                        b_dk += tl.where(mask_j[:, None], d_u_k[None, :], 0.0)
                        b_dv += tl.where(mask_j[:, None], d_u_v[None, :], 0.0)
                        b_dg += tl.where(mask_j[:, None], d_u_g[None, :], 0.0)
                        b_dbeta += tl.where(mask_j[:, None], d_u_beta[None, :], 0.0)

                    mask_seg = (o_c > prev_j) & (o_c <= j) & mask_t
                    q_seg = tl.where(mask_seg[:, None], b_q, 0.0)
                    k_seg = tl.where(mask_seg[:, None], b_k, 0.0)
                    v_seg = tl.where(mask_seg[:, None], b_v, 0.0)
                    a_seg = tl.where(mask_seg[:, None], b_alpha, 0.0)
                    b_seg = tl.where(mask_seg[:, None], b_beta, 0.0)
                    do_seg = tl.where(mask_seg[:, None], b_do, 0.0)
                    c_seg_val = tl.where(mask_seg, b_Aqk, 0.0)

                    aq_seg = a_seg * q_seg
                    ak_seg = a_seg * k_seg
                    bk_seg = b_seg * k_seg
                    S_ak = tl.dot(ak_seg, b_S)

                    d_c_seg_val = tl.sum(do_seg * (v_seg - S_ak), axis=1)
                    d_S_ak = -c_seg_val[:, None] * do_seg

                    d_aq_seg = tl.dot(do_seg, tl.trans(b_S))
                    d_b_S_1 = tl.dot(tl.trans(aq_seg), do_seg)
                    d_ak_seg = tl.dot(d_S_ak, tl.trans(b_S))
                    d_b_S_2 = tl.dot(tl.trans(ak_seg), d_S_ak)

                    b_dq += (d_c_seg_val[:, None] * bk_seg + d_aq_seg * a_seg) * scale
                    b_dk += (d_c_seg_val[:, None] * q_seg * b_seg + d_ak_seg * a_seg)
                    b_dv += c_seg_val[:, None] * do_seg
                    b_dg += (d_aq_seg * q_seg + d_ak_seg * k_seg) * a_seg
                    b_dbeta += d_c_seg_val[:, None] * q_seg * k_seg

                    d_b_S += d_b_S_1 + d_b_S_2
        else:
            for j in range(C - 1, -1, -1):
                if j == 0:
                    p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
                    b_S = tl.load(p_cs, mask=mask_kv, other=0.0).to(tl.float32)
                    prev_j = -1
                else:
                    p_intra = intra_states + (i_bh * N + i_n) * M * K * V + (j - 1) * K * V + o_k[:, None] * V + o_v[None, :]
                    b_S = tl.load(p_intra, mask=mask_kv, other=0.0).to(tl.float32)
                    prev_j = j - 1

                mask_j = (o_c == j) & mask_t
                token_idx = i_n * C + j

                if token_idx < T:
                    offset = base_bth + token_idx * H + i_h
                    u_k = tl.load(k + offset * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32)
                    u_v = tl.load(v + offset * V + o_v, mask=(o_v < V), other=0.0).to(tl.float32)
                    u_alpha = tl.exp(tl.load(g + offset * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32))
                    u_beta = tl.load(beta + offset * K + o_k, mask=(o_k < K), other=0.0).to(tl.float32)

                    a_S = u_alpha[:, None] * b_S
                    k_a_S = tl.sum(u_k[:, None] * a_S, axis=0)
                    b_k_u = u_beta * u_k

                    d_a_S = d_b_S + 0.0
                    d_b_k_u = tl.sum(d_b_S * (u_v[None, :] - k_a_S[None, :]), axis=1)
                    d_k_a_S = tl.sum(d_b_S * (-b_k_u[:, None]), axis=0)
                    d_u_v = tl.sum(d_b_S * b_k_u[:, None], axis=0)

                    d_u_k_1 = tl.sum(d_k_a_S[None, :] * a_S, axis=1)
                    d_a_S += d_k_a_S[None, :] * u_k[:, None]

                    d_u_beta = d_b_k_u * u_k
                    d_u_k_2 = d_b_k_u * u_beta

                    d_u_alpha = tl.sum(d_a_S * b_S, axis=1)
                    d_b_S = d_a_S * u_alpha[:, None]

                    d_u_k = d_u_k_1 + d_u_k_2
                    d_u_g = d_u_alpha * u_alpha

                    b_dk += tl.where(mask_j[:, None], d_u_k[None, :], 0.0)
                    b_dv += tl.where(mask_j[:, None], d_u_v[None, :], 0.0)
                    b_dg += tl.where(mask_j[:, None], d_u_g[None, :], 0.0)
                    b_dbeta += tl.where(mask_j[:, None], d_u_beta[None, :], 0.0)

                mask_seg = (o_c > prev_j) & (o_c <= j) & mask_t
                q_seg = tl.where(mask_seg[:, None], b_q, 0.0)
                k_seg = tl.where(mask_seg[:, None], b_k, 0.0)
                v_seg = tl.where(mask_seg[:, None], b_v, 0.0)
                a_seg = tl.where(mask_seg[:, None], b_alpha, 0.0)
                b_seg = tl.where(mask_seg[:, None], b_beta, 0.0)
                do_seg = tl.where(mask_seg[:, None], b_do, 0.0)
                c_seg_val = tl.where(mask_seg, b_Aqk, 0.0)

                aq_seg = a_seg * q_seg
                ak_seg = a_seg * k_seg
                bk_seg = b_seg * k_seg
                S_ak = tl.dot(ak_seg, b_S)

                d_c_seg_val = tl.sum(do_seg * (v_seg - S_ak), axis=1)
                d_S_ak = -c_seg_val[:, None] * do_seg

                d_aq_seg = tl.dot(do_seg, tl.trans(b_S))
                d_b_S_1 = tl.dot(tl.trans(aq_seg), do_seg)
                d_ak_seg = tl.dot(d_S_ak, tl.trans(b_S))
                d_b_S_2 = tl.dot(tl.trans(ak_seg), d_S_ak)

                b_dq += (d_c_seg_val[:, None] * bk_seg + d_aq_seg * a_seg) * scale
                b_dk += (d_c_seg_val[:, None] * q_seg * b_seg + d_ak_seg * a_seg)
                b_dv += c_seg_val[:, None] * do_seg
                b_dg += (d_aq_seg * q_seg + d_ak_seg * k_seg) * a_seg
                b_dbeta += d_c_seg_val[:, None] * q_seg * k_seg

                d_b_S += d_b_S_1 + d_b_S_2

        p_dq = dq + base_tokens * K + o_k[None, :]
        p_dk = dk + base_tokens * K + o_k[None, :]
        p_dv = dv + base_tokens * V + o_v[None, :]
        p_dg = dg + base_tokens * K + o_k[None, :]
        p_dbeta = dbeta + base_tokens * K + o_k[None, :]

        tl.atomic_add(p_dq, b_dq, mask=mask_tk)
        tl.atomic_add(p_dk, b_dk, mask=mask_tk)
        tl.store(p_dv, b_dv, mask=mask_tv)
        tl.atomic_add(p_dg, b_dg, mask=mask_tk)
        tl.atomic_add(p_dbeta, b_dbeta, mask=mask_tk)

    if HAS_INITIAL_STATE:
        p_d_init = d_initial_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_d_init, d_b_S, mask=mask_kv)