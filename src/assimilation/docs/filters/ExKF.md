ref:
Long short-term memory embedded
nudging schemes for nonlinear data
assimilation of geophysical flows. Pawar et al. doi: 10.1063/5.0012853

# Forecast step
1. Forward propagation:
$$
x_k^b=\mathcal{M}_k(x_{k-1}^a);
$$
2. Evaluate the covariance matrix $P_k^b$ for $x_k^b$:
$$
P_k^b = M_kP_{k-1}^aM_k^T + \Sigma_k^M.
$$

# Analysis step
1. Innovation vectors
$$
d_k = y_k^o - \mathcal{H}_k(x_k^b);
$$
2. Evaluate the Kalman gain matrix
$$
K_k=P_k^bH_k^T(H_kP_k^bH_k^T+\Sigma_k^o)^{-1};
$$
3. Update the analyzed state:
$$
x_k^a = x_k^b + K_kd_k.
$$
4. Update the analyzed covariance matrix:
$$
P_k^a = P_k^b-KH_kP_k^b = (I-KH_k)P_k^b
$$