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
2. Innovation vectors
$$
d_k = y_k^o - \mathcal{H}_k(x_k^b);
$$
3. Kalman gain matrix
$$
\begin{align*}
K_k &= P_k^bH_k^T(H_kP_k^bH_k^T+\Sigma_k^o)^{-1}\\
&= X_k^bX_k^{bT}H_k^T(H_kX_k^bX_k^{bT}H_k^T+\Sigma_k^o)^{-1}\\
&= P_{xz}(P_{zz}+\Sigma_k^o)^{-1},
\end{align*}
$$
4. Assimilate the forecast state estimate with the observation
$$
x_k^a = x_k^b + K_k(y_k^o - \mathcal{H}_k(x_k^b))
$$
5. Compute the analyzed anomalies
$$
X_k^a = X_k-\frac12K_kH_kX_k
$$
1. Compute the analyzed ensembles
$$
x_k^{a,j} = \sqrt{N-1}X_k^a[:, j] + x_k^a
$$