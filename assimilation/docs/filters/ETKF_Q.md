ref:
Latent space data assimilation by using deep learning, Peyron 2021

# Initialization

Construct $U\in\mathbb{R}^{N\times(N-1)}$ such that
$$
\begin{pmatrix}
\frac1{\sqrt N}1_N&U
\end{pmatrix}
$$
is orthogonal, and let
$$
\mathscr{U}=
\begin{pmatrix}
\frac1N1_N&\frac{1}{\sqrt{N-1}}U
\end{pmatrix},
$$
then
$$
\mathscr{U}^{-1}=
\begin{pmatrix}
1_N&\sqrt{N-1}U
\end{pmatrix}^T,
$$

# Forecast step

1. Forward propagation:
$$
x_k^{f,j}=\mathcal{M}_k(x_{k-1}^{a,j});
$$
2. obtain the mean and deviation for $x_k^f$
$$
(x_k^f,\Delta_x^f)=(x_k^{f,j})_j\mathscr{U}
$$
3. eigen decomposition (approximately):
$$
(\Delta_x^f\Delta_x^{fT}+\Sigma_k^M)V_k\approx V_k\Lambda_k,
$$
where $V_k\in\mathbb{R}^{n\times(N-1)}$ and $\Lambda_k\in\mathbb{R}^{(N-1)\times(N-1)}$ are the eigenvectors and eigenvalues of $\Delta_x^f\Delta_x^{fT}+\Sigma_k^M$;

4. update the deviation: 
$$\Delta_x^b=V_k\Lambda_k^{1/2}$$
5. update the ensembles: 
$$(x_k^{b,j})_j=(x_k^f,\Delta_x^b)\mathscr{U}^{-1}$$

# Analysis step

1. Obtain the ensemble mean and deviation matrix from the ensembles:
$$
(x_k^b,\Delta_x^b)=(x_k^{b,j})_j\mathscr{U}
$$
2. obtain the mean and deviation for background estimates of observations:
$$
\begin{align*}
y_k^{b,j} &=\mathcal{H}_k(x_k^{b,j})\\
(y_k^b,\Delta_y^b)&=(y_k^{b,j})_j\mathscr{U}
\end{align*}
$$
3. Transforming matrix:
$$
T_k=(I_{N-1}+\Delta_y^{bT}(\Sigma_k^o)^{-1}\Delta_y^b)^{-1}=(I_{N-1} + S_k^TS_k)^{-1}
$$
for $S_k=(\Sigma_k^o)^{-1/2}\Delta_y^b$;

4. normalized innovation vector
$$
\delta_k = (\Sigma_k^o)^{-1/2}(y_k^o-y_k^b)
$$
5. Update the analyzed ensembles: (modified from ETKF)
$$
\begin{align*}
w_k^a &= (I_{N-1} + \Delta_y^{bT}\Sigma_k^o \Delta_y^b)^{-1}\Delta_y^{bT}(\Sigma_k^o)^{-1}(y_k^o-y_k^b)=T_kS_k^T\delta_k\\
x_k^a &= x_k^b + \Delta_x^bw_k^a = x_k^b + \Delta_x^bT_kS_k^T\delta_k\\
\Delta_x^a &= \Delta_x^b(I_{N-1}+\Delta_y^{bT}(\Sigma_k^o)^{-1}\Delta_y^b)^{-1/2}\\
x_k^{a,j} &= (x_k^a,\Delta_x^a)\mathscr{U}^{-1}
\end{align*}