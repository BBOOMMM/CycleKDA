### CycleKDA






其实，这个反向传播核心只做了两件事，我们可以把它分为两个部分来推导：
1. **Intra-Segment 计算（段内 Token 输出）**：对应代码中的 `BWD Segment (prev_j, j]`
2. **State Jump 计算（段间状态跳变）**：对应代码中的 `BWD State Update @ j`

我们使用反向传播中的标准记法：设 $L$ 为最终的 Loss，对于任意变量 $x$，记 $dx = \frac{\partial L}{\partial x}$。

---

### 第一部分：段内计算 (Intra-Segment)

在连续的段内 $(prev\_j, j]$ 中，状态 $S$ （维度 $K \times V$）是**固定不变**的。

#### 1. 前向公式 (Forward)
对于时间步 $t$，前向算出了一个输出向量 $o_t$（维度 $1 \times V$）：
$$ \tilde{\beta}_t = \beta_t \cdot \rho_t $$
$$ c_t = \tilde{\beta}_t (q_t k_t^T) \quad \text{其中 } q_t, k_t \text{ 是 } 1 \times K \text{ 行向量} $$
设 $\alpha_t = \exp(g_t)$，为了方便，我们定义：
$$ \mathbf{aq}_t = \alpha_t \odot q_t \quad (\text{代码里的 } \mathtt{aq\_seg}) $$
$$ \mathbf{ak}_t = \alpha_t \odot k_t \quad (\text{代码里的 } \mathtt{ak\_seg}) $$
那么输出公式为：
$$ o_t = \mathbf{aq}_t S - c_t (\mathbf{ak}_t S) + c_t v_t $$

#### 2. 反向推导 (Backward)
我们收到了从外部传来的梯度 $do_t$（维度 $1 \times V$），现在要用链式法则求出各个输入变量的梯度。

**① 先求 $v_t$ 和 $c_t$ 的梯度**
对 $o_t$ 公式求偏导：
*   $dv_t = c_t do_t$  *(代码：`b_dv += c_seg_val * do_seg`)*
*   $dc_t = do_t \cdot (v_t - \mathbf{ak}_t S)^T$  
    令 $\mathbf{S\_ak}_t = \mathbf{ak}_t S$，则 $dc_t = \sum (do_t \odot (v_t - \mathbf{S\_ak}_t))$  *(代码：`d_c_seg_val = tl.sum(do_seg * (v_seg - S_ak), axis=1)`)*

**② 由 $dc_t$ 往 $\beta_t, q_t, k_t$ 传导**
因为 $c_t = \tilde{\beta}_t (q_t k_t^T)$：
*   $d\beta_t = dc_t \cdot (q_t k_t^T) \cdot \rho_t$  *(代码：`b_dbeta += d_c_seg_val * tl.sum(...) * rho`)*
*   记 $d(\text{dot\_qk}) = dc_t \tilde{\beta}_t$，这部分会贡献给 $q_t$ 和 $k_t$。*(代码：`d_dot_qk = d_c_seg_val * b_seg`)*

**③ 由 $o_t$ 往 $\mathbf{aq}_t, \mathbf{ak}_t$ 传导**
*   $d\mathbf{aq}_t = do_t S^T$  *(代码：`d_aq_seg = tl.dot(do_seg, tl.trans(b_S))`)*
*   项 $-c_t (\mathbf{ak}_t S)$ 对 $\mathbf{ak}_t S$ 的梯度是 $-c_t do_t$，记为 $d\mathbf{S\_ak}$ *(代码：`d_S_ak = -c_seg_val * do_seg`)*
*   $d\mathbf{ak}_t = d\mathbf{S\_ak} \cdot S^T$ *(代码：`d_ak_seg = tl.dot(d_S_ak, tl.trans(b_S))`)*

**④ 最终合成 $dq_t, dk_t, dg_t$**
因为 $\mathbf{aq} = \alpha \odot q$，$\mathbf{ak} = \alpha \odot k$：
*   $dq_t = (d\mathbf{aq}_t \odot \alpha_t) + d(\text{dot\_qk}) \cdot k_t$
*   $dk_t = (d\mathbf{ak}_t \odot \alpha_t) + d(\text{dot\_qk}) \cdot q_t$
*   对 $\alpha_t$ 的梯度：$d\alpha_t = d\mathbf{aq}_t \odot q_t + d\mathbf{ak}_t \odot k_t$
*   又因为 $\alpha = \exp(g)$，所以 $dg_t = d\alpha_t \odot \alpha_t$

**⑤ 向过去传递给 $S$ 的梯度 $dS$**
$S$ 参与了 $\mathbf{aq}_t S$ 和 $\mathbf{ak}_t S$。
*   $dS_1 = \mathbf{aq}_t^T do_t$ *(代码：`d_b_S_1`)*
*   $dS_2 = \mathbf{ak}_t^T d\mathbf{S\_ak}$ *(代码：`d_b_S_2`)*
这个 $\Delta dS = dS_1 + dS_2$ 会**累加**到全局的 `d_b_S` 中，继续向序列的前方（过去）倒推。

---

### 第二部分：状态跳变更新 (State Jump Update)

在 $t = j$ 处，状态发生了跳变 $S_{old} \rightarrow S_{new}$。

#### 1. 前向公式 (Forward)
令 $\Lambda_j = \text{diag}(\alpha_j)$（即对列做乘法），定义几个中间变量以明确维度：
*   $\mathbf{a\_S} = \Lambda_j S_{old}$  (维度 $K \times V$)
*   $\mathbf{k\_a\_S} = k_j \mathbf{a\_S}$ (向量点积矩阵，维度 $1 \times V$)
*   $\mathbf{b\_k\_u} = \beta_j k_j$  (行向量，维度 $1 \times K$；但在代码的外积中它是 $K \times 1$)

更新公式：
$$ S_{new} = \mathbf{a\_S} - \mathbf{b\_k\_u}^T \mathbf{k\_a\_S} + \mathbf{b\_k\_u}^T v_j $$

#### 2. 反向推导 (Backward)
此时我们手里拿着来自“未来”的梯度 $dS_{new}$（代码里的 `d_b_S`），我们要推导出跳变点对应 token 的梯度，并求出 $dS_{old}$ 以便继续往“过去”倒推。

**① 求中间变量的梯度**
对 $S_{new}$ 公式求偏导：
*   $d(\mathbf{a\_S}) = dS_{new}$ （但这只是初步，后续还有项加上来）
*   $d(\mathbf{b\_k\_u}) = dS_{new} \cdot (v_j - \mathbf{k\_a\_S})^T$ *(求和消去 $V$ 维，剩 $K$ 维。代码：`d_b_k_u = tl.sum(d_b_S * (u_v - k_a_S), axis=1)`)*
*   $d(\mathbf{k\_a\_S}) = -\mathbf{b\_k\_u} \cdot dS_{new}$ *(求和消去 $K$ 维，剩 $V$ 维。代码：`d_k_a_S = tl.sum(d_b_S * (-b_k_u), axis=0)`)*
*   $dv_j = \mathbf{b\_k\_u} \cdot dS_{new}$ *(代码：`d_u_v = tl.sum(d_b_S * b_k_u, axis=0)`)*

**② 拆解 $k\_a\_S = k_j \cdot \mathbf{a\_S}$**
*   贡献给 $k_j$ 的第一部分梯度：$dk_{j,1} = d(\mathbf{k\_a\_S}) \cdot \mathbf{a\_S}^T$ *(代码：`d_u_k_1 = tl.sum(d_k_a_S * a_S, axis=1)`)*
*   贡献给 $\mathbf{a\_S}$ 的追加梯度：$d(\mathbf{a\_S}) += k_j^T d(\mathbf{k\_a\_S})$ *(代码：`d_a_S += d_k_a_S * u_k`)*

**③ 拆解 $\mathbf{b\_k\_u} = \beta_j k_j$**
*   $d\beta_j = d(\mathbf{b\_k\_u}) \cdot k_j^T$ *(代码：`d_u_beta = tl.sum(d_b_k_u * u_k)`)*
*   贡献给 $k_j$ 的第二部分梯度：$dk_{j,2} = d(\mathbf{b\_k\_u}) \cdot \beta_j$ *(代码：`d_u_k_2 = d_b_k_u * u_beta`)*

**④ 拆解 $\mathbf{a\_S} = \Lambda_j S_{old}$，求解 $dS_{old}$ 和 $dg_j$**
现在的 $d(\mathbf{a\_S})$ 已经是完整的了。
*   $d\alpha_j = \sum_{V} (d(\mathbf{a\_S}) \odot S_{old})$ *(代码：`d_u_alpha = tl.sum(d_a_S * b_S, axis=1)`)*
*   $dg_j = d\alpha_j \odot \alpha_j$
*   **【关键】向后传递的隐状态梯度**：$dS_{old} = d(\mathbf{a\_S}) \odot \alpha_j$ *(代码：`d_b_S = d_a_S * u_alpha`。看！这里用新的梯度覆盖了原变量，完成了时序上的逆推)*

### 整体脉络总结 (BPTT)
在 Triton 中逆序遍历的精髓就是这一个随时间流动的 `d_b_S` 矩阵：
1. `d_b_S` 收集未来段内 token (`do_seg`) 的影响 ($\Delta dS$)。
2. 遇到跳变点时，`d_b_S` 作为 $dS_{new}$ 送入跳变求导公式，计算出自身的 $dS_{old}$ 并**覆盖自己**。
3. 循环往复，直到算完第 0 个 token 的梯度，此时的 `d_b_S` 就是传回给 `initial_state` 的梯度。

结合这个数学推导再去看那个 kernel，你就会发现每一行 `tl.sum` 和 `tl.dot` 都是上述多维矩阵偏导公式极其精准的翻译！