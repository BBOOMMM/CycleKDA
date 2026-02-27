import torch

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
    initial_t: int = 0, # 新增
    T_cycle: int = 8,   # 新增
    chunk_indices: torch.LongTensor | None = None,
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
        scale = k.shape[-1] ** -0.5    # 乘 scale 就是 attention 中的 除以根号 d_k
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