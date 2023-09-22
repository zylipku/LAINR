ref:
Long short-term memory embedded
nudging schemes for nonlinear data
assimilation of geophysical flows. Pawar et al. doi: 10.1063/5.0012853

# Forecast step
1. Forward propagation:
$$
x_k^{b,j}=\mathcal{M}_k(x_{k-1}^{a,j})+\varepsilon_k^{M,j},\quad \varepsilon_k^{M,j}\sim\mathcal{N}(0,\Sigma_k^M);
$$
perturbed propagation

# Analysis step
1. Obtain the ensemble mean and anomaly matrix from the ensembles:
$$
\begin{align*}
x_k^b &= \frac1N\sum_{j=1}^Nx_k^{b,j},\\
X_k^b &= \frac1{\sqrt{N-1}}(x_k^{b,j} - x_k^b)_{j=1}^N\in\mathbb{R}^{n\times N},\\
P_k^b &= X_k^bX_k^{bT}
\end{align*}
$$
2. Obtain the perturbed observation vectors:
$$
z_k^{o,j} = y_k^o + \varepsilon_k^{o,j},\quad\varepsilon_k^{o,j}\sim\mathcal{N}(0,\Sigma_k^o);
$$
3. Innovation vectors
$$
d_k^j = z_k^{o,j} - \mathcal{H}_k(x_k^{b,j});
$$
4. Evaluate the Kalman gain matrix
$$
\begin{align*}
K_k &= P_k^bH_k^T(H_kP_k^bH_k^T+\Sigma_k^o)^{-1}\\
&= X_k^bX_k^{bT}H_k^T(H_kX_k^bX_k^{bT}H_k^T+\Sigma_k^o)^{-1}\\
&= P_{xz}(P_{zz}+\Sigma_k^o)^{-1},
\end{align*}
$$
where
$$
P_{xz} = X_k^b(H_kX_k^b)^T,\quad P_{zz} = (H_kX_k^b)(H_kX_k^b)^T.
$$
5. Update the analyzed ensembles:
$$
x_k^{a,j}=x_k^{b,j}+K_kd_k^j.
$$