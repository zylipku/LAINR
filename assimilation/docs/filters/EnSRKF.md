ref:
Data assimilation: Methods, Algorithms and Applications. Bocquet (https://epubs.siam.org/doi/epdf/10.1137/1.9781611974546.ch6)

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
2. Compute the ensemble means and the normalized anomalies for the observations:
$$
\begin{align*}
y_k^{b,j} &= \mathcal{H}_k(x_k^{b,j})\\
y_k^b &= \frac1N\sum_{j=1}^Ny_k^{b,j},\\
Y_k^b &= \frac1{\sqrt{N-1}}(y_k^{b,j} - y_k^b)_{j=1}^N\in\mathbb{R}^{m\times N};
\end{align*}
$$
3. Transforming matrix
$$
T_k = (I_m+Y_k^{bT}(\Sigma_k^o)^{-1}Y_k^b)^{-1} = (I_m + S_k^TS_k)^{-1}
$$
for $S_k=(\Sigma_k^o)^{-1/2}Y_k^b$;

4. normalized innovation vector
$$
\delta_k = (\Sigma_k^o)^{-1/2}(y_k^o-y_k^b)
$$
2. Evaluate the original Kalman gain matrix
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
3. Obtain the modified Kalman gain matrix
$$
\tilde K = K(I + (I + H_kP_k^bH_k^T(\Sigma_k^o)^{-1})^{-1/2})^{-1}
$$
4. Update the analyzed ensembles
$$
\begin{align*}
w_k^a &= (I_m + Y_k^{bT}\Sigma_k^o Y_b)^{-1}Y^{bT}(\Sigma_k^o)^{-1}(y_k^o-y_k^b)=T_kS_k^T\delta_k\\
x_k^a &= x_k^b + X_k^bw_k^a = x_k^b + X_k^bT_kS_k^T\delta_k\\
X_k^a &= (I_n - \tilde{K}_kH_k)X_k^bU,\, U1=1, U\in O(N)\\
x_k^{a,j} &= x_k^a + \sqrt{N-1}X_k^a[:, j]
\end{align*}
$$
OR:
$$
(x_k^{a,j})_j = x_k^b1_N^T + \sqrt{N-1}X_k^a
$$