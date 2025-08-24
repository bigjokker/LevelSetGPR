import numpy as np
from scipy.optimize import minimize

_TWO_PI = 2.0 * np.pi

# Kernel functions
def matern52(d2_over_l2, sf2=1.0):
    r = np.sqrt(np.maximum(d2_over_l2, 1e-12))
    return sf2 * (1 + np.sqrt(5)*r + 5*r*r/3.0) * np.exp(-np.sqrt(5)*r)

def kernel_matern52(X, Z, ell, sf2=1.0):
    if np.isscalar(ell):
        invL2 = 1.0 / (ell**2)
    else:
        invL2 = 1.0 / (ell**2)
    d2 = np.sum(((X[:, None, :] - Z[None, :, :])**2) * invL2, axis=2)
    return matern52(d2, sf2=sf2)

def kernel_linear(X1, X2, sf_lin=1.0):
    return sf_lin * np.dot(X1, X2.T)

def kernel_const(X1, X2, sf_const=1.0):
    return sf_const * np.ones((X1.shape[0], X2.shape[0]))

def kernel_full(X1, X2, ell, sf_mat=1.0, sf_lin=1.0, sf_const=1.0):
    return kernel_matern52(X1, X2, ell, sf_mat) + kernel_linear(X1, X2, sf_lin) + kernel_const(X1, X2, sf_const)

# Standardize
class Standardizer:
    def fit(self, X):
        self.mu = X.mean(0, keepdims=True)
        self.sd = X.std(0, keepdims=True) + 1e-12
        return self
    def transform(self, X):
        return (X - self.mu) / self.sd
    def inv(self, Xs):
        return Xs * self.sd + self.mu

# Build augmented system
def build_augmented_system(X, level_groups, N, pre_standardized=False, std=None):
    if pre_standardized:
        Xs = X
        std_obj = None
    else:
        std_obj = std or Standardizer().fit(X)
        Xs = std_obj.transform(X)
    A_val = np.eye(N)
    diffs = []
    deriv_vs = []
    deriv_ms = []
    for g in level_groups:
        if len(g) <= 1:
            continue
        root = g[0]
        for j in g[1:]:
            row = np.zeros(N)
            row[j] = 1.0
            row[root] = -1.0
            diffs.append(row)
            v = (Xs[j] - Xs[root]) / (np.linalg.norm(Xs[j] - Xs[root]) + 1e-12)
            m = (Xs[j] + Xs[root]) / 2
            deriv_vs.append(v)
            deriv_ms.append(m)
    A_base = A_val if not diffs else np.vstack([A_val, np.vstack(diffs)])
    n_val = N
    n_diff = len(diffs)
    n_deriv = len(deriv_vs)
    return A_base, n_val, n_diff, n_deriv, np.array(deriv_vs), np.array(deriv_ms), Xs, std_obj

# GP predict
def gp_predict_with_constraints(X, y, level_groups, Xstar, ell, sf_mat=1.0, sf_lin=1.0, sf_const=1.0,
                                sigma_level=None, sigma_delta=0.01, sigma_deriv=0.01, jitter=1e-6, pre_standardized=False, std=None):
    N, n = X.shape
    A_base, n_val, n_diff, n_deriv, deriv_vs_arr, deriv_ms_arr, Xs, std_obj = build_augmented_system(X, level_groups, N, pre_standardized, std)
    M_base = A_base.shape[0]
    M = M_base + n_deriv
    Xts = Xstar if pre_standardized else std_obj.transform(Xstar)
    K_XX = kernel_full(Xs, Xs, ell, sf_mat, sf_lin, sf_const)
    K_AA_base = A_base @ K_XX @ A_base.T
    K_AA = np.zeros((M, M))
    K_AA[:M_base, :M_base] = K_AA_base
    # Vectorized K_DX = Cov(∂, f(X)) = v · ∇_x k(m, x_j)
    K_DX = np.zeros((n_deriv, N))
    if n_deriv > 0:
        diffs = deriv_ms_arr[:, None, :] - Xs[None, :, :]  # (D, N, k)
        rs = np.linalg.norm(diffs, axis=2) / ell  # (D, N)
        c = -sf_mat * (5.0 / (3.0 * ell**2))
        exp_term = np.exp(-np.sqrt(5.0) * rs)
        scalar = c * exp_term * (1.0 + np.sqrt(5.0) * rs)  # (D, N)
        grad_mat = scalar[:, :, None] * diffs  # (D, N, k)
        grad_lin = sf_lin * Xs[None, :, :]  # (1, N, k)
        grad_total = grad_mat + grad_lin
        K_DX = np.einsum('D N k, D N k -> D N', deriv_vs_arr[:, None, :], grad_total)  # (D, N)
    # Cross blocks
    K_AA[:M_base, M_base:] = A_base @ K_DX.T
    K_AA[M_base:, :M_base] = K_AA[:M_base, M_base:].T
    # DD: tiny ridge + noise
    K_AA[M_base:, M_base:] = 1e-8
    # Noises
    if sigma_level is None:
        sigma_level = np.full(N, 0.05 * np.ptp(y) + 1e-6)
    Sigma = np.concatenate([sigma_level**2, np.full(n_diff, sigma_delta**2), np.full(n_deriv, sigma_deriv**2)])
    K_AA[np.diag_indices_from(K_AA)] += Sigma + jitter * np.trace(K_AA) / max(M, 1)
    # Unit checks
    assert np.allclose(K_AA, K_AA.T, atol=1e-8), "K_AA not symmetric"
    z = np.concatenate([y, np.zeros(n_diff + n_deriv)])
    L = np.linalg.cholesky(K_AA)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
    # Vectorized k_xA = Cov(f(x*), A f + ∂)
    K_xX = kernel_full(Xts, Xs, ell, sf_mat, sf_lin, sf_const)
    k_xA_base = K_xX @ A_base.T
    k_xD = np.zeros((Xts.shape[0], n_deriv))
    if n_deriv > 0:
        diffs_td = deriv_ms_arr[:, None, :] - Xts[None, :, :]  # (D, T, k)
        rs_td = np.linalg.norm(diffs_td, axis=2) / ell  # (D, T)
        c = -sf_mat * (5.0 / (3.0 * ell**2))
        exp_term_td = np.exp(-np.sqrt(5.0) * rs_td)
        scalar_td = c * exp_term_td * (1.0 + np.sqrt(5.0) * rs_td)  # (D, T)
        grad_mat_first = scalar_td[:, :, None] * diffs_td  # (D, T, k)
        grad_mat_second = -grad_mat_first
        grad_lin_second = sf_lin * Xts[None, :, :]  # (1, T, k)
        grad_second = grad_mat_second + grad_lin_second  # (D, T, k)
        k_xD = np.einsum('D T k, D k -> T D', grad_second, deriv_vs_arr)  # (T, D)
    k_xA = np.hstack([k_xA_base, k_xD])
    mu = k_xA @ alpha
    v = np.linalg.solve(L, k_xA.T)
    k_xx_diag = sf_mat + sf_lin * np.sum(Xts**2, axis=1) + sf_const
    var = np.maximum(k_xx_diag - np.sum(v**2, axis=0), 0.0)
    # Unit check
    assert np.all(var >= -1e-10), "Negative variance detected"
    return mu, var

# Drop and remap groups
def drop_and_remap_groups(level_groups, idx):
    max_idx = max(max(grp) for grp in level_groups if grp)
    remap = {j: j if j < idx else j-1 for j in range(max_idx + 1) if j != idx}
    out = []
    for grp in level_groups:
        gg = [remap.get(j) for j in grp if j != idx]
        if gg:
            out.append(gg)
    return out

# Conformal quantile
def conformal_quantile(X, y, level_groups, ell, sf_mat, sf_lin, sf_const,
                       sigma_level, sigma_delta, sigma_deriv, jitter, pre_standardized=False, std=None):
    residuals = []
    N = len(y)
    for i in range(N):
        X_loo = np.delete(X, i, 0)
        y_loo = np.delete(y, i)
        groups_loo = drop_and_remap_groups(level_groups, i)
        sigma_level_loo = np.delete(sigma_level, i)
        mu_loo, _ = gp_predict_with_constraints(X_loo, y_loo, groups_loo, X[i:i+1],
                                                ell, sf_mat, sf_lin, sf_const,
                                                sigma_level=sigma_level_loo, sigma_delta=sigma_delta, sigma_deriv=sigma_deriv,
                                                jitter=jitter, pre_standardized=pre_standardized, std=std)
        residuals.append(abs(y[i] - mu_loo[0]))
    return np.quantile(residuals, 0.95)

# Robust L
def robust_L_levels(X, y, level_groups, eps=1e-6, inflate=1.5):
    levels = []
    for grp in level_groups:
        y_grp = y[np.array(grp)]
        levels.append((np.mean(y_grp), np.array(grp)))
    levels.sort(key=lambda t: t[0])
    Ls = []
    for (_, gA), (_, gB) in zip(levels[:-1], levels[1:]):
        XA, XB = X[gA], X[gB]
        dA = []
        for a in XA:
            d = np.median(np.linalg.norm(XB - a, axis=1) + eps)
            dA.append(d)
        num = abs(np.mean(y[gB]) - np.mean(y[gA]))
        Ls.append(num / np.median(dA))
    return inflate * (max(Ls) if Ls else 0.0)

# Lipschitz bounds
def lipschitz_bounds(X, y, Xstar, L):
    lower = np.full(len(Xstar), -np.inf)
    upper = np.full(len(Xstar), np.inf)
    for i in range(len(X)):
        d = np.linalg.norm(Xstar - X[i], axis=1)
        lower = np.maximum(lower, y[i] - L * d)
        upper = np.minimum(upper, y[i] + L * d)
    return lower, upper

# PCA project
def pca_project(X, d):
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt.T[:, :d]
    X_proj = Xc @ components
    return X_proj, components, mu

# Active sampling
def active_sample(X, y, level_groups, candidates, ell, sf_mat, sf_lin, sf_const,
                  sigma_level, sigma_delta, sigma_deriv, jitter, pre_standardized=False, std=None, rule='max_var', level_c=None,
                  L=None, L_space=None, pca_components=None, pca_mu=None):
    if candidates.size == 0:
        raise ValueError("No candidates provided.")
    mu, var = gp_predict_with_constraints(X, y, level_groups, candidates, ell, sf_mat, sf_lin, sf_const,
                                          sigma_level, sigma_delta, sigma_deriv, jitter, pre_standardized, std)
    sigma = np.sqrt(var)
    if rule == 'max_var':
        scores = var
    elif rule == 'contour_focus' and level_c is not None:
        scores = 1 / (1 + ((mu - level_c) / (sigma + 1e-6))**2)
    elif rule == 'envelope_shrink' and L is not None and L_space is not None:
        # Build the SAME space used when L was computed
        if L_space == 'std':
            X_space = X if pre_standardized else std.transform(X)
            C_space = candidates if pre_standardized else std.transform(candidates)
        elif L_space == 'pca':
            X_std = X if pre_standardized else std.transform(X)
            C_std = candidates if pre_standardized else std.transform(candidates)
            X_space = (X_std - pca_mu) @ pca_components
            C_space = (C_std - pca_mu) @ pca_components
        else:
            raise ValueError("L_space must be 'std' or 'pca' when using envelope_shrink.")
        lower, upper = lipschitz_bounds(X_space, y, C_space, L)
        scores = upper - lower
    else:
        raise ValueError("Invalid rule or missing parameters for chosen rule.")
    return candidates[np.argmax(scores)]

# NLL Computation
def _assemble_KAA_and_z(
    X, y, level_groups,
    ell, sf_mat, sf_lin, sf_const,
    sigma_level, sigma_delta, sigma_deriv, jitter,
    pre_standardized=False, std=None
):
    """Build K_AA and z for the augmented system (values + diffs + derivs)."""
    N = len(y)
    # Build augmented system + standardized Xs
    A_base, n_val, n_diff, n_deriv, deriv_vs_arr, deriv_ms_arr, Xs, std_obj = \
        build_augmented_system(X, level_groups, N, pre_standardized, std)
    # Kernel on anchors
    K_XX = kernel_full(Xs, Xs, ell, sf_mat, sf_lin, sf_const)
    # Base (values + diffs)
    K_AA_base = A_base @ K_XX @ A_base.T
    # Derivative–value block (vectorized)
    if n_deriv > 0:
        diffs = deriv_ms_arr[:, None, :] - Xs[None, :, :] # (D,N,k)
        rs = np.linalg.norm(diffs, axis=2) / ell # (D,N)
        c = -sf_mat * (5.0 / (3.0 * ell**2))
        exp_term = np.exp(-np.sqrt(5.0) * rs) # (D,N)
        scalar = c * exp_term * (1.0 + np.sqrt(5.0) * rs) # (D,N)
        grad_mat = scalar[:, :, None] * diffs # (D,N,k)
        grad_lin = sf_lin * Xs[None, :, :] # (1,N,k)
        grad_total = grad_mat + grad_lin # (D,N,k)
        K_DX = np.einsum('D N k, D N k -> D N', deriv_vs_arr[:, None, :], grad_total)
    else:
        K_DX = np.zeros((0, N))
    # Assemble full K_AA
    M_base = K_AA_base.shape[0]
    M = M_base + n_deriv
    K_AA = np.zeros((M, M))
    K_AA[:M_base, :M_base] = K_AA_base
    if n_deriv > 0:
        # Cross blocks
        K_AA[:M_base, M_base:] = A_base @ K_DX.T
        K_AA[M_base:, :M_base] = K_AA[:M_base, M_base:].T
        # Deriv-deriv block: tiny ridge (dominated by sigma_deriv)
        K_AA[M_base:, M_base:] = 1e-8
    # Noise vector (values, diffs, derivs)
    if sigma_level is None:
        sigma_level = np.full(N, 0.05 * np.ptp(y) + 1e-6)
    Sigma = np.concatenate([
        sigma_level**2,
        np.full(n_diff, sigma_delta**2),
        np.full(n_deriv, sigma_deriv**2)
    ])
    # Add noise + jitter to diagonal
    K_AA[np.diag_indices_from(K_AA)] += Sigma
    K_AA[np.diag_indices_from(K_AA)] += jitter * max(np.trace(K_AA) / max(M, 1), 1.0)
    # Observations vector
    z = np.concatenate([y, np.zeros(n_diff + n_deriv)])
    # Sanity
    assert np.allclose(K_AA, K_AA.T, atol=1e-8), "K_AA not symmetric"
    return K_AA, z

def neg_log_marginal(
    X, y, level_groups,
    ell, sf_mat, sf_lin, sf_const,
    sigma_level, sigma_delta, sigma_deriv, jitter,
    pre_standardized=False, std=None
):
    """Compute NLL = 0.5*z^T K^-1 z + 0.5*log|K| + 0.5*M*log(2π) for augmented system."""
    K_AA, z = _assemble_KAA_and_z(
        X, y, level_groups,
        ell, sf_mat, sf_lin, sf_const,
        sigma_level, sigma_delta, sigma_deriv, jitter,
        pre_standardized, std
    )
    M = K_AA.shape[0]
    # Robust Cholesky with jitter backoff
    jitter_local = 0.0
    for attempt in range(5):
        try:
            L = np.linalg.cholesky(K_AA + np.eye(M)*jitter_local)
            break
        except np.linalg.LinAlgError:
            jitter_local = (10.0 if jitter_local == 0.0 else jitter_local * 10.0)
            print(f"Cholesky retry {attempt+1}: bumped jitter to {jitter_local}")
    else:
        # As a last resort, return a huge penalty
        return 1e50
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
    nll = 0.5 * z.dot(alpha) + np.sum(np.log(np.diag(L))) + 0.5 * M * np.log(_TWO_PI)
    return nll

def optimize_hyperparams(
    X, y, level_groups,
    init, bounds,
    sigma_level=None, jitter=1e-6,
    pre_standardized=False, std=None
):
    """
    Optimize (log) hyperparams with L-BFGS-B.
    init = [log_ell, log_sf_mat, log_sf_lin, log_sf_const, log_sigma_delta, log_sigma_deriv]
    bounds = [(a,b), ...] in log space (use None for unbounded)
    """
    def _objective(theta):
        log_ell, log_sf_mat, log_sf_lin, log_sf_const, log_sig_d, log_sig_g = theta
        ell = np.exp(log_ell)
        sf_mat = np.exp(log_sf_mat)
        sf_lin = np.exp(log_sf_lin)
        sf_const = np.exp(log_sf_const)
        sigma_delta = np.exp(log_sig_d)
        sigma_deriv = np.exp(log_sig_g)
        return neg_log_marginal(
            X, y, level_groups,
            ell, sf_mat, sf_lin, sf_const,
            sigma_level, sigma_delta, sigma_deriv, jitter,
            pre_standardized, std
        )
    res = minimize(_objective, x0=np.array(init), bounds=bounds, method="L-BFGS-B")
    out = res.x
    return {
        "success": res.success,
        "message": res.message,
        "n_iters": res.nit,
        "log_params": {
            "ell": out[0],
            "sf_mat": out[1],
            "sf_lin": out[2],
            "sf_const": out[3],
            "sigma_delta": out[4],
            "sigma_deriv": out[5],
        },
        "params": {
            "ell": float(np.exp(out[0])),
            "sf_mat": float(np.exp(out[1])),
            "sf_lin": float(np.exp(out[2])),
            "sf_const": float(np.exp(out[3])),
            "sigma_delta": float(np.exp(out[4])),
            "sigma_deriv": float(np.exp(out[5])),
        },
        "fun": res.fun,
    }

# Active sampling loop
def active_sampling_loop(
    X, y, level_groups,
    candidates_gen_fn,  # () -> (Q,n) array of candidate points
    oracle_f,  # (x) -> scalar label; replace with real evaluation
    num_add=3,
    rule='max_var',  # 'max_var' | 'contour_focus' | 'envelope_shrink'
    level_c=None,  # needed for 'contour_focus'
    use_pca=False, pca_d=3,  # model-aligned space option
    L_space='std',  # 'std' or 'pca' if using envelope_shrink
    ell0=1.0, sf_mat0=0.5, sf_lin0=1e-4, sf_const0=0.4,
    sig_delta0=0.01, sig_deriv0=0.01,
    bounds=None,  # bounds in log-space; if None, sensible defaults set below
    sigma_level_init=None,  # per-point noise (len N) or None
    jitter=1e-6,
    var_stop=None  # stop if max predictive var <= var_stop
):
    # --- init state ---
    cur_X = X.copy()
    cur_y = y.copy()
    cur_groups = [g.copy() for g in level_groups]
    cur_sigma_level = (sigma_level_init.copy()
                       if sigma_level_init is not None
                       else np.full(len(cur_y), 0.05 * np.ptp(cur_y) + 1e-6))
    # log-space init & bounds
    init = [
        np.log(ell0), np.log(sf_mat0), np.log(sf_lin0),
        np.log(sf_const0), np.log(sig_delta0), np.log(sig_deriv0)
    ]
    if bounds is None:
        bounds = [
            (np.log(1e-2), np.log(1e2)), # ell
            (np.log(1e-6), np.log(1e3)), # sf_mat
            (np.log(1e-8), np.log(1e1)), # sf_lin
            (np.log(1e-8), np.log(1e2)), # sf_const
            (np.log(1e-6), np.log(1e0)), # sigma_delta
            (np.log(1e-6), np.log(1e0)), # sigma_deriv
        ]
    # --- per-iteration loop ---
    params = None
    pca_components, pca_mu = None, None
    for it in range(num_add):
        # Fit standardizer on current X and (optionally) PCA on standardized X
        std = Standardizer().fit(cur_X)
        X_std = std.transform(cur_X)
        if use_pca:
            X_proj, pca_components, pca_mu = pca_project(X_std, pca_d)
        # Re-optimize hyperparams (warm-start if available)
        if params is None:
            init_vec = init
        else:
            # preserve order: [ell, sf_mat, sf_lin, sf_const, sigma_delta, sigma_deriv]
            init_vec = [
                np.log(params['ell']), np.log(params['sf_mat']), np.log(params['sf_lin']),
                np.log(params['sf_const']), np.log(params['sigma_delta']), np.log(params['sigma_deriv'])
            ]
        opt = optimize_hyperparams(
            cur_X, cur_y, cur_groups,
            init_vec, bounds,
            sigma_level=cur_sigma_level, jitter=jitter,
            pre_standardized=False, std=std
        )
        params = opt['params'] # dict with keys: ell, sf_mat, sf_lin, sf_const, sigma_delta, sigma_deriv
        # Early-stop check (optional): max predictive variance at a quick candidate batch
        cand_quick = candidates_gen_fn()
        mu_q, var_q = gp_predict_with_constraints(
            cur_X, cur_y, cur_groups, cand_quick,
            params['ell'], params['sf_mat'], params['sf_lin'], params['sf_const'],
            sigma_level=cur_sigma_level,
            sigma_delta=params['sigma_delta'], sigma_deriv=params['sigma_deriv'],
            jitter=jitter, pre_standardized=False, std=std
        )
        if (var_stop is not None) and (np.max(var_q) <= var_stop):
            break
        # If envelope_shrink: compute L in *the same* space we will score candidates
        if rule == 'envelope_shrink':
            if L_space == 'std':
                L = robust_L_levels(X_std, cur_y, cur_groups)
            elif L_space == 'pca':
                if not use_pca:
                    raise ValueError("use_pca=True required when L_space='pca'")
                L = robust_L_levels(X_proj, cur_y, cur_groups)
            else:
                raise ValueError("L_space must be 'std' or 'pca'")
        else:
            L = None
        # Generate candidates and pick next_x with aligned geometry
        candidates = candidates_gen_fn()
        next_x = active_sample(
            cur_X, cur_y, cur_groups, candidates,
            params['ell'], params['sf_mat'], params['sf_lin'], params['sf_const'],
            cur_sigma_level, params['sigma_delta'], params['sigma_deriv'], jitter,
            pre_standardized=False, std=std,
            rule=rule, level_c=level_c,
            L=L, L_space=L_space,
            pca_components=pca_components, pca_mu=pca_mu
        )
        # Query the oracle (replace with real evaluation)
        next_y = oracle_f(next_x)
        # Update dataset (add as a stand-alone value row; attach to a level if you know it)
        cur_X = np.vstack([cur_X, next_x])
        cur_y = np.append(cur_y, next_y)
        cur_groups = cur_groups + [[len(cur_y) - 1]] # new singleton group
        cur_sigma_level = np.append(cur_sigma_level, np.median(cur_sigma_level))  # Median update; customize if needed
        print(f"[iter {it+1}] added point; y={next_y:.4f}; N={len(cur_y)}; "
              f"maxVar(cand)={np.max(var_q):.3g}; ell={params['ell']:.3g}")
    return cur_X, cur_y, cur_groups, params

# Main (toy example)
# (Insert toy X, y, level_groups here)
n = 10  # From toy
def oracle_f(x):
    return np.sum(x)  # Replace with real

def candidates_gen():
    return np.random.uniform(-1, 1, (100, n))  # Generator

new_X, new_y, new_groups, final_params = active_sampling_loop(
    X, y, level_groups, candidates_gen, oracle_f, num_add=3, rule='max_var'
)
print("Final params:", final_params)