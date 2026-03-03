import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
# from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from myfla.chunk_fwd import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
from myfla.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra, chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_fwd
from fla.ops.kda.wy_fast import recompute_w_u_fwd

from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from fla.utils import IS_NVIDIA_HOPPER, USE_CUDA_GRAPH, autotune_cache_kwargs

from einops import rearrange


@torch.compiler.disable
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
    use_triton: bool = False,
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

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert k.shape[-1] <= 256, "Currently we only support key headdim <=256 for KDA :-("
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
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

        B, T, H, K, V, C = *k.shape, v.shape[-1], chunk_size
        N = (T + C - 1) // C

        q, k, v, g, beta = map(
            lambda x: rearrange(x, 'b (n c) h ... -> b h n c ...', c=C), [q, k, v, g, beta]
        )
        q = q * scale

        S = torch.zeros((B, H, K, V), device=k.device, dtype=torch.float32)
        if initial_state is not None:
            S = S + initial_state.to(torch.float32)

        o = torch.zeros_like(v)

        for i in range(0, N):
            q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
            beta_i = beta[:, :, i]
            q_i, k_i, v_i, g_i = q_i.to(torch.float32), k_i.to(torch.float32), v_i.to(torch.float32), g_i.to(torch.float32)

            cur_t_start = initial_t + i * C
            t_math = cur_t_start + torch.arange(0, C, device=q.device)
            mod_t = t_math % T_cycle

            rho = torch.where(mod_t == 0, 1.0, mod_t / T_cycle)
            rho = rho.view(1, 1, C).to(torch.float32)

            beta_i = beta_i * rho

            alpha_i = torch.exp(g_i)

            update_indices = torch.nonzero(mod_t == 0, as_tuple=True)[0].tolist()

            S_curr = S  # float32
            last_update_j = -1

            def compute_o_segment(start_idx, end_idx):
                if start_idx > end_idx:
                    return
                q_seg = q_i[:, :, start_idx:end_idx+1]
                k_seg = k_i[:, :, start_idx:end_idx+1]
                v_seg = v_i[:, :, start_idx:end_idx+1]
                a_seg = alpha_i[:, :, start_idx:end_idx+1]
                b_seg = beta_i[:, :, start_idx:end_idx+1]

                c_seg = (b_seg * torch.einsum('bhlk, bhlk -> bhl', q_seg, k_seg)).unsqueeze(-1)

                # 用 float32 的 S 做累积更稳
                aq_seg = a_seg * q_seg
                term1 = torch.einsum('bhkv, bhlk -> bhlv', S_curr, aq_seg)

                ak_seg = a_seg * k_seg
                S_ak = torch.einsum('bhkv, bhlk -> bhlv', S_curr, ak_seg)
                term2 = c_seg * S_ak

                term3 = c_seg * v_seg
                o[:, :, i, start_idx:end_idx+1] = term1 - term2 + term3

            for j in update_indices:
                compute_o_segment(last_update_j + 1, j)
                
                last_update_j = j
                
                if cur_t_start + j==0:
                    continue

                a_j = alpha_i[:, :, j, :].unsqueeze(-1)  # [B,H,K,1]
                k_j = k_i[:, :, j, :]                    # [B,H,K]
                v_j = v_i[:, :, j, :]                    # [B,H,V]
                b_j = beta_i[:, :, j].unsqueeze(-1)      # [B,H,1]

                a_S = a_j * S_curr
                k_a_S = torch.einsum('bhk, bhkv -> bhv', k_j, a_S)
                b_k = b_j * k_j
                S_curr = a_S - torch.einsum('bhk, bhv -> bhkv', b_k, k_a_S) + torch.einsum('bhk, bhv -> bhkv', b_k, v_j)

            compute_o_segment(last_update_j + 1, C - 1)
            S = S_curr

        o = rearrange(o, 'b h n c v -> b (n c) h v').to(v.dtype)

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
        initial_t: int = 0, # 新增
        T_cycle: int = 8,   # 新增
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
        if use_qk_l2norm_in_kernel:   # 做 RMSNorm 归一化
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        
        chunk_size = 64
        
        
        o, Aqk, Akk, final_state = chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            initial_t=initial_t, # 新增
            T_cycle=T_cycle,   # 新增
            chunk_indices=chunk_indices,
        )
        
        
        if use_gate_in_kernel:
            g = None
        
        
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, g_org, beta, A_log, dt_bias, Aqk, Akk, initial_state, cu_seqlens, chunk_indices
        )
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.initial_t = initial_t     # 新增：用于 bwd
        ctx.T_cycle = T_cycle         # 新增：用于 bwd
        return o.to(q.dtype), final_state
    
    
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        raise NotImplementedError("The backward function is not implemented yet. Please use the PyTorch version for now.")
    
    


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
    initial_t: int = 0, # 新增
    T_cycle: int = 8,   # 新增
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    # 输入 shape 约定: q, k, v, g, beta 都已经被 flatten 成[B, T, H, D]
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    N = (T + C - 1) // C

    o = torch.empty_like(v)
    Aqk = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
    
    # Kernel 1 和 Kernel 2 之间的桥梁：存储每块起点的状态
    chunk_states = torch.empty((B, H, N, K, V), device=q.device, dtype=torch.float32)
    
    final_state = None
    if output_final_state:
        final_state = torch.empty((B, H, K, V), device=q.device, dtype=torch.float32)

    # ----- 执行 Kernel 1: 扫描状态 -----
    grid_state = (B * H, )
    fused_chunk_kda_state_prep_kernel[grid_state](
        k, v, g, beta, chunk_states, initial_state, final_state,
        scale, initial_t, T_cycle, N, C, B, H, K, V,
        USE_INITIAL_STATE=(initial_state is not None),
        STORE_FINAL_STATE=output_final_state
    )

    # ----- 执行 Kernel 2: 极限并行计算输出 -----
    grid_output = (N, B * H)
    fused_chunk_kda_parallel_output_kernel[grid_output](
        q, k, v, g, beta, chunk_states, o, Aqk,
        scale, initial_t, T_cycle, N, C, B, H, K, V
    )

    Akk = None # 本算法无需 Akk
    return o, Aqk, Akk, final_state



# @triton.autotune(
#     configs=[
#         # triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4]
#         for num_stages in [2, 3, 4]
#         # for BV in [32, 64]
#     ],
#     key=['H', 'K', 'V', 'C'],
#     use_cuda_graph=USE_CUDA_GRAPH,
#     **autotune_cache_kwargs,
# )
@triton.jit
def fused_chunk_kda_state_prep_kernel(
    k,
    v,
    g,
    beta,
    chunk_states,
    initial_state,
    final_state,
    scale,
    initial_t, 
    T_cycle: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    b_S = tl.zeros([K, V], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = initial_state + i_bh * K * V + tl.arange(0, K)[:, None] * V + tl.arange(0, V)[None, :]
        b_S += tl.load(p_h0).to(tl.float32)

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)

    for i_n in range(N):
        # 1. 记录本 Chunk 初始的底座状态
        p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_cs, b_S)

        # 2. 数学计算：找出本 Chunk 内的第一个跳变点 j
        start_t = initial_t + i_n * C
        rem = start_t % T_cycle
        j = (T_cycle - rem) if rem != 0 else 0
        
        # 处理 t=0 不更新的特殊情况
        if start_t == 0 and j == 0:
            j += T_cycle

        # 3. 跨越式扫描（抛弃 for j in range(C)，性能起飞！）
        while j < C:
            token_idx = i_n * C + j
            offset = i_b * (N * C) * H + token_idx * H + i_h
            
            p_k = k + offset * K + o_k
            p_v = v + offset * V + o_v
            p_g = g + offset * K + o_k
            p_b = beta + offset
            
            u_k = tl.load(p_k).to(tl.float32)
            u_v = tl.load(p_v).to(tl.float32)
            u_alpha = tl.exp(tl.load(p_g).to(tl.float32))
            u_beta = tl.load(p_b).to(tl.float32)
            
            a_S = u_alpha[:, None] * b_S 
            k_a_S = tl.sum(u_k[:, None] * a_S, axis=0)
            b_k = u_beta * u_k
            
            b_S = a_S - b_k[:, None] * k_a_S[None, :] + b_k[:, None] * u_v[None, :]
            
            # 直接跳到下一个更新点
            j += T_cycle

    if STORE_FINAL_STATE:
        p_ht = final_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_S)
        
        

@triton.jit
def fused_chunk_kda_parallel_output_kernel(
    q, k, v, g, beta, chunk_states, o, Aqk,
    scale, initial_t, T_cycle: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr
):
    i_n = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    o_c = tl.arange(0, C) 

    # 1. 提取当前块的初始状态
    p_cs = chunk_states + (i_bh * N + i_n) * K * V + o_k[:, None] * V + o_v[None, :]
    b_S = tl.load(p_cs)

    # 2. 一次性加载整个 Chunk
    base_offset = i_b * N * C * H + i_n * C * H + i_h * K
    p_q = q + (i_b * N * C * H + (i_n * C + o_c)[:, None] * H + i_h) * K + o_k[None, :]
    p_k = k + (i_b * N * C * H + (i_n * C + o_c)[:, None] * H + i_h) * K + o_k[None, :]
    p_v = v + (i_b * N * C * H + (i_n * C + o_c)[:, None] * H + i_h) * V + o_v[None, :]
    p_g = g + (i_b * N * C * H + (i_n * C + o_c)[:, None] * H + i_h) * K + o_k[None, :]
    p_b = beta + (i_b * N * C * H + (i_n * C + o_c) * H + i_h)

    b_q = tl.load(p_q).to(tl.float32) * scale
    b_k = tl.load(p_k).to(tl.float32)
    b_v = tl.load(p_v).to(tl.float32)
    b_g = tl.load(p_g).to(tl.float32)
    b_beta = tl.load(p_b).to(tl.float32)

    b_alpha = tl.exp(b_g)
    cur_t_global = initial_t + i_n * C + o_c
    mod_t = cur_t_global % T_cycle
    rho = tl.where(mod_t == 0, 1.0, mod_t.to(tl.float32) / T_cycle)
    b_beta_tilde = b_beta * rho

    b_o = tl.zeros([C, V], dtype=tl.float32)
    
    # 3. 数学寻找起始更新点
    start_t = initial_t + i_n * C
    rem = start_t % T_cycle
    j = (T_cycle - rem) if rem != 0 else 0
    if start_t == 0 and j == 0:
        j += T_cycle

    last_update_j = -1

    # 4. 跨越式分段执行 (替换掉原来的 for j in range(C))
    while j < C:
        # ==== 阶段 A：计算 Segment (last_update_j, j] 的 O ====
        mask_seg = (o_c > last_update_j) & (o_c <= j)
        
        q_seg = tl.where(mask_seg[:, None], b_q, 0.0)
        k_seg = tl.where(mask_seg[:, None], b_k, 0.0)
        v_seg = tl.where(mask_seg[:, None], b_v, 0.0)
        a_seg = tl.where(mask_seg[:, None], b_alpha, 0.0)
        b_seg = tl.where(mask_seg, b_beta_tilde, 0.0)

        aq_seg = a_seg * q_seg
        ak_seg = a_seg * k_seg
        dot_qk = tl.sum(q_seg * k_seg, axis=1) # [C]
        c_seg_val = b_seg * dot_qk
        
        p_aqk = Aqk + (i_b * N * C * H + (i_n * C + o_c) * H + i_h)
        tl.store(p_aqk, c_seg_val, mask=mask_seg)

        # 这里的 tl.dot 现在只在这个段落发生时调用一次！
        term1 = tl.dot(aq_seg, b_S)          
        S_ak = tl.dot(ak_seg, b_S)
        b_o += term1 - (c_seg_val[:, None] * S_ak) + (c_seg_val[:, None] * v_seg)

        # ==== 阶段 B：状态跳变更新 ====
        token_idx = i_n * C + j
        offset = i_b * (N * C) * H + token_idx * H + i_h
        
        u_k = tl.load(k + offset * K + o_k).to(tl.float32)
        u_v = tl.load(v + offset * V + o_v).to(tl.float32)
        u_alpha = tl.exp(tl.load(g + offset * K + o_k).to(tl.float32))
        u_beta = tl.load(beta + offset).to(tl.float32)
        
        a_S = u_alpha[:, None] * b_S
        k_a_S = tl.sum(u_k[:, None] * a_S, axis=0)
        b_k_u = u_beta * u_k
        
        b_S = a_S - b_k_u[:, None] * k_a_S[None, :] + b_k_u[:, None] * u_v[None, :]

        last_update_j = j
        
        # 飞跃到下一个更新点！
        j += T_cycle

    # 5. 处理本 Chunk 的尾部段落 (Tail Segment)
    if last_update_j < C - 1:
        mask_seg = o_c > last_update_j
        
        q_seg = tl.where(mask_seg[:, None], b_q, 0.0)
        k_seg = tl.where(mask_seg[:, None], b_k, 0.0)
        v_seg = tl.where(mask_seg[:, None], b_v, 0.0)
        a_seg = tl.where(mask_seg[:, None], b_alpha, 0.0)
        b_seg = tl.where(mask_seg, b_beta_tilde, 0.0)

        aq_seg = a_seg * q_seg
        ak_seg = a_seg * k_seg
        dot_qk = tl.sum(q_seg * k_seg, axis=1)
        c_seg_val = b_seg * dot_qk
        
        p_aqk = Aqk + (i_b * N * C * H + (i_n * C + o_c) * H + i_h)
        tl.store(p_aqk, c_seg_val, mask=mask_seg)

        term1 = tl.dot(aq_seg, b_S)
        S_ak = tl.dot(ak_seg, b_S)
        b_o += term1 - (c_seg_val[:, None] * S_ak) + (c_seg_val[:, None] * v_seg)

    # 6. 统一写回全局显存
    p_o = o + (i_b * N * C * H + (i_n * C + o_c)[:, None] * H + i_h) * V + o_v[None, :]
    tl.store(p_o, b_o.to(o.dtype.element_ty))
    
    
    
    