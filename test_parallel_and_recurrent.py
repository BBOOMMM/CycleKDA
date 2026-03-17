import torch

# from myfla import chunk_kda, fused_recurrent_kda
from myfla.chunk_vecbeta import chunk_kda
from myfla.fuesd_recurrent_vecbeta import fused_recurrent_kda


def test_chunk_kda():
    torch.manual_seed(42)
    
    B, T, H, K, V = 2, 120, 4, 64, 64
    q = torch.randn(B, T, H, K).cuda()
    k = torch.randn(B, T, H, K).cuda()
    v = torch.randn(B, T, H, V).cuda()
    g = torch.randn(B, T, H, K).cuda()
    g = g - g.max()
    # beta = torch.rand(B, T, H).sigmoid().cuda()
    beta = torch.rand(B, T, H, K).float().sigmoid().cuda()
    
    print("Testing Parallel...")
    o_parallel, state_parallel = chunk_kda(
                            q=q,
                            k=k,
                            v=v,
                            g=g,
                            beta=beta,
                            initial_state=None,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                            cu_seqlens=None,
                            initial_t=0,
                            T_cycle=8,
                            use_triton=False,
                        )
    print(f"Parallel output shape: {o_parallel.shape}")
    print(f"Parallel output mean: {o_parallel.mean().item():.6f}")
    
    print("\nTesting Recurrent...")
    o_recurrent, state_recurrent = fused_recurrent_kda(
                                            q=q,
                                            k=k,
                                            v=v,
                                            g=g,
                                            beta=beta,
                                            initial_state=None,
                                            output_final_state=True,
                                            use_qk_l2norm_in_kernel=True,
                                            cu_seqlens=None,
                                            initial_t=0,
                                            T_cycle=8,
                                        )
    print(f"Recurrent output shape: {o_recurrent.shape}")
    print(f"Recurrent output mean: {o_recurrent.mean().item():.6f}")
    
    diff = (o_recurrent - o_parallel).abs()
    print(f"\nMax diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")
    
    state_diff = (state_parallel - state_recurrent).abs()
    print(f"\nState Max diff: {state_diff.max().item():.6f}")
    print(f"State Mean diff: {state_diff.mean().item():.6f}")
    
    for i in range(T):
        step_diff = (o_recurrent[:, i] - o_parallel[:, i]).abs()
        if step_diff.max().item() > 1e-5:
            print(f"Step {i}: Max diff: {step_diff.max().item():.6f}, Mean diff: {step_diff.mean().item():.6f}")


if __name__ == "__main__":
    test_chunk_kda()
