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
2. Obtain the perturbed observation vectors:
$$
z_k^{o,j} = y_k^o + \varepsilon_k^{o,j},\quad\varepsilon_k^{o,j}\sim\mathcal{N}(0,\Sigma_k^o);
$$
3. Compute the ensemble means and the normalized anomalies for the perturbed background estimates for the observations:
$$
\begin{align*}
y_k^{p,j} &= \mathcal{H}_k(x_k^{b,j}) - \varepsilon_k^{o,j}\\
y_k^p &= \frac1N\sum_{j=1}^Ny_k^{p,j},\\
Y_k^p &= \frac1{\sqrt{N-1}}(y_k^{p,j} - y_k^p)_{j=1}^N\in\mathbb{R}^{n\times N}.
\end{align*}
$$
4. Kalman gain matrix
$$
K_k=X_k^bY_k^{pT}(Y_k^pY_k^{pT})^{-1}
$$
5. Update the analyzed ensembles:
$$
x_k^{a,j}=x_k^{b,j}+K_k(z_k^{o,j}-\mathcal{H}_k(x_k^{b,j})).
$$