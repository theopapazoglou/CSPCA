"""
Simulation 3 (i.i.d. - correlated features, multivariate response).

DGP:  
X ~ N(0, I_p) for i.i.d., X ~ N(0, Sigma),  Sigma_ij = 0.4^|i-j|  for correlated 
Y1 = 3*X1 - 5*X2 + 4*X3 + e1
Y2 = 2*X2 + 4*X4 -  X5 + e2
Y3 =  -X1 + 3*X3 + 2*X6 + e3
e1, e2, e3 moderately correlated across responses 
n = 100, p = 600, scalar response
The three responses depend on partially overlapping but distinct predictor sets.

Compares PCA, PLS, HSIC-SPCA, LSPCA, sisPCA, CSPCA across
q in {2,4,6,8,10} over 100 replicates. Each replicate uses a 60/20/20
train/validation/test split; hyperparameters are tuned on validation MSE;
variance explained, covariance explained are reported on the training set; MSE
is reported on the held-out test set.

Reproducibility: 
data seed = 0 + rep, split seed = 1000 + rep 

Results are deterministic given these seeds and the pinned dependencies
in requirements.txt.

Dependencies:  see requirements.txt; sispca is installed from GitHub:
      pip install git+https://github.com/JiayuSuPKU/sispca.git
"""

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import autograd.numpy as anp
from pymanopt.manifolds import Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
import pymanopt.function
from sispca import SISPCA, Supervision, SISPCADataset

# Runs both i.i.d. and correlated scenarios
def generate_data(n, p, correlated=False, rho=0.4, noise_sd=0.1,
                  resp_corr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    if correlated:
        idx = np.arange(p)
        Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
        X = rng.standard_normal((n, p)) @ L.T
    else:
        X = rng.standard_normal((n, p))

    k = 3
    Cov_e = (1 - resp_corr) * np.eye(k) + resp_corr * np.ones((k, k))
    Le = np.linalg.cholesky(Cov_e)
    E = noise_sd * (rng.standard_normal((n, k)) @ Le.T)

    Y = np.column_stack([
        3*X[:, 0] - 5*X[:, 1] + 4*X[:, 2],
        2*X[:, 1] + 4*X[:, 3] -   X[:, 4],
         -X[:, 0] + 3*X[:, 2] + 2*X[:, 5],
    ]) + E
    return X, Y


# PCA
def pca_projection(X, q):
    return PCA(n_components=q).fit(X).components_.T

# HSIC-SPCA
def rbf_kernel(Y, sigma):
    D = cdist(Y, Y, 'euclidean')
    return np.exp(-D ** 2 / (2 * sigma ** 2))

def spca_hsic(X, Y, q, sigma):
    K = rbf_kernel(Y, sigma)
    C = X.T @ K @ X
    ev, V = linalg.eigh(C)
    return V[:, np.argsort(ev)[::-1][:q]]

# LSPCA
def lspca_projection(X, Y, q, lam, max_iters=200):
    p, k = X.shape[1], Y.shape[1]
    manifold = Product([Grassmann(p, q), Euclidean(q, k)])

    @pymanopt.function.autograd(manifold)
    def cost(L, beta):
        XL = X @ L
        reg = anp.sum((Y - XL @ beta) ** 2)             
        rec = anp.sum((X - XL @ L.T) ** 2)              
        return reg + lam * rec
    L_init = PCA(n_components=q).fit(X).components_.T
    beta_init = np.linalg.lstsq(X @ L_init, Y, rcond=None)[0]
    optimizer = ConjugateGradient(verbosity=0, max_iterations=max_iters)
    result = optimizer.run(Problem(manifold=manifold, cost=cost),
                               initial_point=[L_init, beta_init])
    return result.point[0]


# sisPCA
def sispca_projection(X, Y, q, lam, max_epochs=50, patience=5):
    supervision = [Supervision(Y, target_type='continuous')]
    dataset = SISPCADataset(X, target_supervision_list=supervision)
    model = SISPCA(dataset, n_latent_sub=[q], lambda_contrast=lam,
                       kernel_subspace="gaussian")
    model.fit(batch_size=-1, max_epochs=max_epochs,
                  early_stopping_patience=patience,
                  enable_progress_bar=False, enable_model_summary=False)
    return model._get_U_subspace_list()[0].detach().cpu().numpy()


# CSPCA
def cspca_exact(X, Y, q, kappa):
    XtY = X.T @ Y
    C = XtY @ XtY.T + kappa * (X.T @ X)
    ev, V = linalg.eigh(C)
    return V[:, np.argsort(ev)[::-1][:q]]

# Evaluation metrics -- Downstream model is ordinary linear regression
def variance_explained(X, W):
    d = np.linalg.norm(X)**2
    return np.linalg.norm(X @ W)**2 / d if d > 0 else 0.0

def covariance_explained(X, Y, W):
    d = np.linalg.norm(X.T @ Y)**2
    return np.linalg.norm(W.T @ X.T @ Y)**2 / d if d > 0 else 0.0

def eval_metrics(W, Xtr, Ytr, Xte, Yte):
    if W is None:
        k = Ytr.shape[1]
        return {"var_expl": np.nan, "mse": np.nan, "cov_expl": np.nan,
                "mse_per": [np.nan]*k}
    pred = LinearRegression().fit(Xtr @ W, Ytr).predict(Xte @ W)
    return {
        "var_expl": variance_explained(Xtr, W),            
        "mse":      mean_squared_error(Yte, pred),          
        "cov_expl": covariance_explained(Xtr, Ytr, W),      
        "mse_per":  np.mean((Yte - pred)**2, axis=0).tolist(),
    }


# Run simulation analysis. 
def run_simulation(correlated, q_list=(2, 4, 6, 8, 10), n_reps=100,
                   n=100, p=600,
                   kappa_grid=(0.001, 0.01, 0.1, 1, 5, 10.0, 25, 50),
                   lspca_lambda_grid=(0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                   sispca_lambda_grid=(0.05, 0.1, 0.5, 1.0, 5, 10.0),
                   sigma_candidates=(0.001, 0.005, 0.01, 0.05,  0.1, 0.5, 1, 10),
                   seed_base=0, verbose=True):
    methods = ['PCR', 'PLS', 'HSIC', 'LSPCA', 'sisPCA', 'CSPCA']
    metrics = ['var_expl', 'mse', 'cov_expl']
    R = {m: {q: {mt: [] for mt in metrics} for q in q_list} for m in methods}
    Rper = {m: {q: [] for q in q_list} for m in methods}  

    for rep in range(n_reps):
        X, Y = generate_data(n, p, correlated=correlated, seed=seed_base + rep)
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(n)
        n_tr, n_val = int(0.6 * n), int(0.2 * n)
        tr, val, te = perm[:n_tr], perm[n_tr:n_tr+n_val], perm[n_tr+n_val:]

        sx = StandardScaler().fit(X[tr]); sy = StandardScaler().fit(Y[tr])
        Xtr, Xval, Xte = sx.transform(X[tr]), sx.transform(X[val]), sx.transform(X[te])
        Ytr, Yval, Yte = sy.transform(Y[tr]), sy.transform(Y[val]), sy.transform(Y[te])       

        def record(name, W):
            m = eval_metrics(W, Xtr, Ytr, Xte, Yte)
            for mt in metrics:
                R[name][q][mt].append(m[mt])
            Rper[name][q].append(m["mse_per"])

        for q in q_list:
            record('PCR', pca_projection(Xtr, q))

            pls = PLSRegression(n_components=q, scale=False).fit(Xtr, Ytr)
            record('PLS', pls.x_weights_)

            best_sig, best = sigma_candidates[0], np.inf
            for sig in sigma_candidates:
                W = spca_hsic(Xtr, Ytr, q, sig)
                mse = mean_squared_error(Yval,
                    LinearRegression().fit(Xtr @ W, Ytr).predict(Xval @ W))
                if mse < best: best, best_sig = mse, sig
            record('HSIC', spca_hsic(Xtr, Ytr, q, best_sig))

        
            best_lam, best = lspca_lambda_grid[0], np.inf
            for lam in lspca_lambda_grid:
                W = lspca_projection(Xtr, Ytr, q, lam)
                if W is None: continue
                mse = mean_squared_error(Yval,
                    LinearRegression().fit(Xtr @ W, Ytr).predict(Xval @ W))
                if mse < best: best, best_lam = mse, lam
            record('LSPCA', lspca_projection(Xtr, Ytr, q, best_lam))

            best_lam, best = sispca_lambda_grid[0], np.inf
            for lam in sispca_lambda_grid:
                    W = sispca_projection(Xtr, Ytr, q, lam)
                    if W is None: continue
                    mse = mean_squared_error(Yval,
                        LinearRegression().fit(Xtr @ W, Ytr).predict(Xval @ W))
                    if mse < best: best, best_lam = mse, lam
            record('sisPCA', sispca_projection(Xtr, Ytr, q, best_lam))
            

            best_kap, best = kappa_grid[0], np.inf
            for kap in kappa_grid:
                W = cspca_exact(Xtr, Ytr, q, kap)
                mse = mean_squared_error(Yval,
                    LinearRegression().fit(Xtr @ W, Ytr).predict(Xval @ W))
                if mse < best: best, best_kap = mse, kap
            record('CSPCA', cspca_exact(Xtr, Ytr, q, best_kap))

        if verbose and (rep + 1) % max(1, n_reps // 10) == 0:
            print(f"    rep {rep + 1}/{n_reps} done", flush=True)

    return R, Rper, methods, list(q_list)

# Run simulation and print results
def print_table(R, methods, q_list, metric, label):
    print(f"\n{'-'*84}\n {label}\n{'-'*84}")
    print(f"{'Method':<9}" + "".join(f"q={q:<15}" for q in q_list))
    for m in methods:
        row = f"{m:<9}"
        for q in q_list:
            v = np.array(R[m][q][metric]); nf = np.sum(~np.isnan(v))
            row += (f"{np.nanmean(v):.4f} ({np.nanstd(v)/np.sqrt(nf):.4f}) "
                    if nf > 0 else "  NA           ")
        print(row)


def print_per_response(Rper, methods, q_list, label):
    print(f"\n{'-'*84}\n Per-response test MSE (mean over reps)  — {label}\n{'-'*84}")
    for q in q_list:
        print(f"  q={q}:")
        for m in methods:
            arr = np.array(Rper[m][q])                 # reps x k
            if arr.size == 0 or np.all(np.isnan(arr)):
                print(f"    {m:<9} NA"); continue
            per = np.nanmean(arr, axis=0)
            print(f"    {m:<9} " + "  ".join(f"Y{r+1}={per[r]:.4f}" for r in range(len(per))))


if __name__ == "__main__":
    n_reps = 100
    q_list = [2, 4, 6, 8, 10]

    for correlated, label in [(False, "i.i.d."), (True, "Toeplitz correlated (rho=0.4)")]:
        print(f"\n{'='*84}")
        print(f" n=100, p=600, {n_reps} replicates, q in {q_list}")
        print(f"{'='*84}")
        R, Rper, methods, qs = run_simulation(correlated=correlated,
                                              q_list=q_list, n_reps=n_reps)
        print_table(R, methods, qs, 'var_expl', "Variance explained — TRAIN (mean, SE)")
        print_table(R, methods, qs, 'mse',      "Mean squared error — TEST (mean, SE)")
        print_table(R, methods, qs, 'cov_expl', "Covariance explained — TEST (mean, SE)")
        print_per_response(Rper, methods, qs, label)