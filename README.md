# CycleKDA

## Kimi Linear

$$\begin{aligned}\mathbf{S}_t&=\left(\mathbf{I}-\beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top\right)\mathrm{Diag}(\boldsymbol{\alpha}_t)\mathbf{S}_{t-1}+\beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top\in \mathbb{R}^{d_k \times d_v};
\\
\boldsymbol{o}_t&=\mathbf{S}_t^\top \boldsymbol{q}_t\in \mathbb{R}^{d_v}.\end{aligned}$$

## CycleKDA

$$\begin{aligned}\rho_t&=\begin{cases}\dfrac{t \bmod T}{T}, & t \bmod T \neq 0 \\
1, & t \bmod T = 0\end{cases}\\
\mathbf{S}'_t&=\left(\mathbf{I}-
\rho_t
(\boldsymbol{\beta}_t \odot \boldsymbol{k}_t)
\boldsymbol{k}_t^\top
\right)
\mathrm{Diag}(\boldsymbol{\alpha}_t)
\mathbf{S}_{t-1}
+
\rho_t
(\boldsymbol{\beta}_t \odot \boldsymbol{k}_t)
\boldsymbol{v}_t^\top ;
\\
\mathbf{S}_t
&=
\begin{cases}
\mathbf{S}_{t-1}, & t \bmod T \neq 0 \\
\mathbf{S}'_t, & t \bmod T = 0
\end{cases}
\\
\boldsymbol{o}_t
&=
\left(
\mathbf{S}'_t
\right)^\top
\boldsymbol{q}_t
\in \mathbb{R}^{d_v}.
\end{aligned}
$$
