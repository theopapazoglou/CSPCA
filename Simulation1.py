"""
Simulation 1 (i.i.d. - correlated features, single scalar response).

DGP:  
X ~ N(0, I_p) for i.i.d., X ~ N(0, Sigma),  Sigma_ij = 0.4^|i-j|  for correlated 
Y = 3*X1 - 5*X2 + 4*X3 + eps,  eps ~ N(0, 0.1^2)
n = 100, p = 600, scalar response.

Compares PCA, PLS, HSIC-SPCA, Bair, LSPCA, sisPCA, CSPCA across
q in {2,4,6,8,10} over 100 replicates. Each replicate uses a 60/20/20
train/validation/test split; hyperparameters are tuned on validation MSE;
variance explained, covariance explained are reported on the training set; MSE
is reported on the held-out test set.

Reproducibility: 
data seed = 1994 + rep, split seed = 1000 + rep  for iid
data seed = 1924 + rep, split seed = 1000 + rep  for correlated


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

# Generates the data for the i.i.d. scenario
def generate_data(n, p, noise_sd=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    eps = noise_sd * rng.standard_normal(n)
    y = 3 * X[:, 0] - 5 * X[:, 1] + 4 * X[:, 2] + eps
    return X, y.reshape(-1, 1)

# Generates the data for the correlated scenario
def generate_data_cor(n, p, rho=0.4, noise_sd=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    X = rng.standard_normal((n, p)) @ L.T
    eps = noise_sd * rng.standard_normal(n)
    y = 3 * X[:, 0] - 5 * X[:, 1] + 4 * X[:, 2] + eps
    return X, y.reshape(-1, 1)

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

# Bair's
def bair_scores(X, y):
    num = X.T @ y.ravel()
    den = np.sqrt(np.sum(X ** 2, axis=0))
    den[den == 0] = np.inf
    return np.abs(num / den)


def bair_projection(X, scores, threshold, q):
    p = X.shape[1]
    sel = scores >= threshold
    if sel.sum() < q:
        return None
    pb = PCA(n_components=q).fit(X[:, sel])
    W = np.zeros((p, q))
    W[sel, :] = pb.components_.T
    return W

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
    d = np.linalg.norm(X) ** 2
    return np.linalg.norm(X @ W) ** 2 / d if d > 0 else 0.0

def covariance_explained(X, Y, W):
    d = np.linalg.norm(X.T @ Y) ** 2
    return np.linalg.norm(W.T @ X.T @ Y) ** 2 / d if d > 0 else 0.0

def evaluate(W, Xtr, Ytr, Xte, Yte):
    if W is None:
        return {"var_expl": np.nan, "mse": np.nan, "cov_expl": np.nan}
    pred = LinearRegression().fit(Xtr @ W, Ytr).predict(Xte @ W)
    return {"var_expl": variance_explained(Xtr, W),
            "mse":      mean_squared_error(Yte, pred),
            "cov_expl": covariance_explained(Xtr, Ytr, W)}


# Run simulation analysis. To run the correlated scenario uncomment generate_data_cor
def run_simulation(q_list=(2, 4, 6, 8, 10), n_reps=100,
                   n=100, p=600,
                   kappa_grid=(0.01, 0.1, 5, 1.0, 10.0, 50),
                   lspca_lambda_grid=(0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                   sispca_lambda_grid=(0.05, 0.1, 0.5, 1.0, 5, 10.0),
                   threshold_grid=(0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
                   sigma_grid=(0.001, 0.005, 0.01, 0.05,  0.1, 0.5, 1, 10), 
                   seed_base=1994, # seed_base=1924 for the correlated scenario
                   verbose=True):
    methods = ['PCR', 'PLS', 'HSIC', 'Bair', 'LSPCA', 'sisPCA', 'CSPCA']
    metrics = ['var_expl', 'mse', 'cov_expl']
    R = {m: {q: {mt: [] for mt in metrics} for q in q_list} for m in methods}

    for rep in range(n_reps):
        X, Y = generate_data(n, p, seed=seed_base + rep) #iid
        #X, Y = generate_data_cor(n, p, rho=0.4, seed=seed_base + rep) #correlated

        
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(n)
        n_tr, n_val = int(0.6 * n), int(0.2 * n)
        tr, val, te = perm[:n_tr], perm[n_tr:n_tr + n_val], perm[n_tr + n_val:]
        sx = StandardScaler().fit(X[tr]); sy = StandardScaler().fit(Y[tr])
        Xtr, Xval, Xte = sx.transform(X[tr]), sx.transform(X[val]), sx.transform(X[te])
        Ytr, Yval, Yte = sy.transform(Y[tr]), sy.transform(Y[val]), sy.transform(Y[te])

        sigma_candidates = list(sigma_grid)
        bair_sc = bair_scores(Xtr, Ytr)

        for q in q_list:
            W = pca_projection(Xtr, q)
            for mt, v in evaluate(W, Xtr, Ytr, Xte, Yte).items():
                R['PCR'][q][mt].append(v)

            pls = PLSRegression(n_components=q, scale=False).fit(Xtr, Ytr)
            for mt, v in evaluate(pls.x_weights_, Xtr, Ytr, Xte, Yte).items():
                R['PLS'][q][mt].append(v)
            
            best_sig, best_mse = sigma_candidates[0], np.inf
            for sig in sigma_candidates:
                W = spca_hsic(Xtr, Ytr, q, sig)
                pred = LinearRegression().fit(Xtr @ W, Ytr).predict(Xval @ W)
                mse = mean_squared_error(Yval, pred)
                if mse < best_mse: best_mse, best_sig = mse, sig
            W = spca_hsic(Xtr, Ytr, q, best_sig)
            for mt, v in evaluate(W, Xtr, Ytr, Xte, Yte).items():
                R['HSIC'][q][mt].append(v)

            best_thr, best_mse = None, np.inf
            for thr in threshold_grid:
                W = bair_projection(Xtr, bair_sc, thr, q)
                if W is None: continue
                pred = LinearRegression().fit(Xtr @ W, Ytr).predict(Xval @ W)
                mse = mean_squared_error(Yval, pred)
                if mse < best_mse: best_mse, best_thr = mse, thr
            W = bair_projection(Xtr, bair_sc, best_thr, q)
            for mt, v in evaluate(W, Xtr, Ytr, Xte, Yte).items():
                R['Bair'][q][mt].append(v)

            best_lam, best_mse = lspca_lambda_grid[0], np.inf
            for lam in lspca_lambda_grid:
                W = lspca_projection(Xtr, Ytr, q, lam)
                if W is None: continue
                pred = LinearRegression().fit(Xtr @ W, Ytr).predict(Xval @ W)
                mse = mean_squared_error(Yval, pred)
                if mse < best_mse: best_mse, best_lam = mse, lam
            W = lspca_projection(Xtr, Ytr, q, best_lam)
            for mt, v in evaluate(W, Xtr, Ytr, Xte, Yte).items():
                R['LSPCA'][q][mt].append(v)

            best_lam, best_mse = sispca_lambda_grid[0], np.inf
            for lam in sispca_lambda_grid:
                    W = sispca_projection(Xtr, Ytr, q, lam)
                    if W is None: continue
                    pred = LinearRegression().fit(Xtr @ W, Ytr).predict(Xval @ W)
                    mse = mean_squared_error(Yval, pred)
                    if mse < best_mse: best_mse, best_lam = mse, lam
            W = sispca_projection(Xtr, Ytr, q, best_lam)
            for mt, v in evaluate(W, Xtr, Ytr, Xte, Yte).items():
                    R['sisPCA'][q][mt].append(v)

            best_kap, best_mse = kappa_grid[0], np.inf
            for kap in kappa_grid:
                W = cspca_exact(Xtr, Ytr, q, kap)
                pred = LinearRegression().fit(Xtr @ W, Ytr).predict(Xval @ W)
                mse = mean_squared_error(Yval, pred)
                if mse < best_mse: best_mse, best_kap = mse, kap
            W = cspca_exact(Xtr, Ytr, q, best_kap)
            for mt, v in evaluate(W, Xtr, Ytr, Xte, Yte).items():
                R['CSPCA'][q][mt].append(v)

        if verbose and (rep + 1) % max(1, n_reps // 10) == 0:
            print(f"    rep {rep + 1}/{n_reps} done", flush=True)

    return R, methods, list(q_list)

# Run simulation and print results
def print_table(R,methods,q_list,metric,label):
    print(f"\n{'-' * 84}\n"f" {label}\n"f"{'-' * 84}")
    print(f"{'Method':<9}"+ "".join(f"q={q:<15}"for q in q_list))
    for m in methods:
        row = f"{m:<9}"
        for q in q_list:
            v = np.array(R[m][q][metric])
            nfin = np.sum(~np.isnan(v)
            )
            if nfin == 0:
                row += "NA"
            else:
                row += (f"{np.nanmean(v):.4f} "f"({np.nanstd(v) / np.sqrt(nfin):.4f}) ")
        print(row)

if __name__ == "__main__":
    n_reps = 100
    q_list = [2, 4, 6, 8, 10]
    print(f"\n{'=' * 84}")
    print(f" n=100, p=600, {n_reps} replicates, "f"q in {q_list}")
    print(f"{'=' * 84}")
    R, methods, qs = run_simulation(q_list=q_list,n_reps=n_reps)
    print_table(R,methods,qs,'var_expl',"Variance explained (mean, SE)")
    print_table(R,methods,qs,'mse',"Mean squared error (mean, SE)")
    print_table(R,methods,qs,'cov_expl',"Covariance explained (mean, SE)")