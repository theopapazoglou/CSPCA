"""
Bushel's Liver Toxicity Dataset 

Bushel et al. (2007) Simultaneous clustering of gene expression data with clinical chemistry
and pathological evaluations reveals phenotypic prototypes. BMC Systems Biology, 1, 15.

Extracted from the mixOmics library from the Bioconductor R package 1 https://www.rdocumentation.org/packages/mixOmics/versions/6.3.2/topics/liver.toxicity
 
n = 64, p = 3116
We specify Albumin levels (ALB.g.dl) as the continuous outcome

Reproducibility: 
Nystr\"om landmark sampling seed = 24, split seed = 1994 + rep, cv seed = 1924

Compares PCA, PLS, HSIC-SPCA, Bair, LSPCA, sisPCA, CSPCA across
q in {2,3,4,5,6} over 100 replicates. Each replicate uses a stratified 80/20
tuning/test split; hyperparameters are tuned by 5-fold cross-validation
on MSE; variance explained, covariance explained are reported on
the tuning set; MSE is reported on the held-out test set.

Results are deterministic given these seeds and the pinned dependencies
in requirements.txt.

Dependencies:  see requirements.txt; sispca is installed from GitHub:
      pip install git+https://github.com/JiayuSuPKU/sispca.git
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.linalg import eigh, svd, qr
from scipy.spatial.distance import cdist
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


# PCA
def pca_projection(X, q):
    return PCA(n_components=q).fit(X).components_.T

# HSIC-SPCA
def rbf_kernel_from_dist(dist_matrix, sigma):
    return np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

def rbf_kernel(Y, sigma):
    dist_matrix = cdist(Y, Y, 'euclidean')
    return np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))


def spca_hsic(X, K, q):
    n = X.shape[0]

    H = np.eye(n) - np.ones((n, n)) / n
    L = H @ K @ H
    L = (L + L.T) / 2.0                   
    w, Vk = linalg.eigh(L)
    w = np.clip(w, 0.0, None)
    D = np.diag(np.sqrt(w)) @ Vk.T       
    Xc = X.T                                 
    C = Xc @ H @ D.T                        

    CtC = C.T @ C
    CtC = (CtC + CtC.T) / 2.0
    evals, Vc = linalg.eigh(CtC)
    order = np.argsort(evals)[::-1]
    evals = evals[order][:q]
    Vc = Vc[:, order][:, :q]

    pos = evals > 1e-12 * max(evals.max(), 1e-300)
    evals_pos = evals[pos]
    Vc = Vc[:, pos]

    S_inv = np.diag(1.0 / np.sqrt(evals_pos))
    U = C @ Vc @ S_inv                       # p x q

    return {'W': U, 'singular_values': evals_pos}


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
def lspca_projection(X, Y, q, lam, max_iters=150):
    p, k = X.shape[1], Y.shape[1]
    pca_fallback = PCA(n_components=q).fit(X).components_.T
    manifold = Product([Grassmann(p, q), Euclidean(q, k)])

    @pymanopt.function.autograd(manifold)
    def cost(L, beta):
        XL = X @ L
        reg = anp.sum((Y - XL @ beta) ** 2)            
        rec = anp.sum((X - XL @ L.T) ** 2)          
        return reg + lam * rec

    L_init = pca_fallback
    beta_init = np.linalg.lstsq(X @ L_init, Y, rcond=None)[0]
    optimizer = ConjugateGradient(verbosity=0, max_iterations=max_iters)
    result = optimizer.run(Problem(manifold=manifold, cost=cost),
                               initial_point=[L_init, beta_init])
    W = result.point[0]
    return W
    

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


# Nystr\"om CSPCA, m=\lceil\sqrt{p}\rceil"
def cspca_nystrom(X, Y, q, m, kappa, seed=24):
    p = X.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(p, size=m, replace=False)
    Xm = X[:, idx]
    S = X.T @ (Y @ (Y.T @ Xm)) + kappa * (X.T @ Xm)    
    Cm = S[idx, :]                                      
    ev, Um = eigh(Cm)
    o = np.argsort(ev)[::-1]; ev, Um = ev[o], Um[:, o]
    pos = ev > 1e-10 * ev.max()
    B = S @ Um[:, pos] @ np.diag(1.0 / np.sqrt(ev[pos]))
    Q, Rmat = qr(B, mode='economic')
    theta, V = eigh(Rmat @ Rmat.T)
    oo = np.argsort(theta)[::-1]
    W = (Q @ V[:, oo])[:, :q]
    return W


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


# Data loading
def load_data():
    Y_df = pd.read_csv('liver_toxicity_Y.csv')
    X_df = pd.read_csv('liver_toxicity_X.csv')
    Y = Y_df.values
    X = X_df.values
    if len(Y.shape) == 1 or Y.shape[1] == 1:
        Y = Y.reshape(-1, 1)
    print("Y shape:", Y.shape)
    print("X shape:", X.shape)
    return X, Y


# Tuning via 5-fold cross-validation on MSE
def _kfold_indices(n, k, rng):
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def cv_select(candidates, fit_predict_mse, default):
    best_cand, best_cv = default, np.inf
    for cand in candidates:
        fold_mses = fit_predict_mse(cand)
        if fold_mses is None or len(fold_mses) == 0:
            continue
        mean_cv = float(np.mean(fold_mses))
        if np.isfinite(mean_cv) and mean_cv < best_cv:
            best_cv, best_cand = mean_cv, cand
    return best_cand

# Run regression analysis
def run_analysis(X, Y, q_list=(2, 3, 4, 5, 6), n_reps=100,
                 n_folds=10,
                 kappa_grid=(0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100),
                 lspca_lambda_grid=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1),
                 sispca_lambda_grid=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 5, 10.0),
                 threshold_grid=(0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7),
                 sigma_grid=(0.001, 0.01, 0.1, 0.5, 1, 10),
                 nystrom_m=45, verbose=True):
    methods = ['PCR', 'PLS', 'HSIC', 'Bair', 'LSPCA', 'sisPCA', 'CSPCA']
    metrics = ['var_expl', 'mse', 'cov_expl']
    R = {m: {q: {mt: [] for mt in metrics} for q in q_list} for m in methods}

    n, p = X.shape
    
    for rep in range(n_reps):

        rng = np.random.default_rng(1994 + rep)
        perm = rng.permutation(n)
        n_tune = int(0.8 * n)
        tune_idx, te = perm[:n_tune], perm[n_tune:]

        X_tune_raw, Y_tune_raw = X[tune_idx], Y[tune_idx]
        X_te_raw,   Y_te_raw   = X[te],       Y[te]
        sx = StandardScaler().fit(X_tune_raw); sy = StandardScaler().fit(Y_tune_raw)
        Xtune = sx.transform(X_tune_raw); Ytune = sy.transform(Y_tune_raw)
        Xte   = sx.transform(X_te_raw);   Yte   = sy.transform(Y_te_raw)
        cv_rng = np.random.default_rng(1924 + rep)
        folds = _kfold_indices(len(tune_idx), n_folds, cv_rng)

        fold_data = []
        for tr_i, va_i in folds:
            sxf = StandardScaler().fit(X_tune_raw[tr_i])
            syf = StandardScaler().fit(Y_tune_raw[tr_i])
            Xtr_f = sxf.transform(X_tune_raw[tr_i]); Ytr_f = syf.transform(Y_tune_raw[tr_i])
            Xva_f = sxf.transform(X_tune_raw[va_i]); Yva_f = syf.transform(Y_tune_raw[va_i])
            fold_data.append((Xtr_f, Ytr_f, Xva_f, Yva_f,
                              cdist(Ytr_f, Ytr_f, 'euclidean'),
                              bair_scores(Xtr_f, Ytr_f)))

        bair_sc_full = bair_scores(Xtune, Ytune)

        def fold_mse(make_W):
            out = []
            for (Xtr_f, Ytr_f, Xva_f, Yva_f, dist_f, bair_f) in fold_data:
                W = make_W(Xtr_f, Ytr_f, dist_f, bair_f)
                if W is None:
                    continue
                pred = LinearRegression().fit(Xtr_f @ W, Ytr_f).predict(Xva_f @ W)
                out.append(mean_squared_error(Yva_f, pred))
            return out

        for q in q_list:
            W = pca_projection(Xtune, q)
            for mt, v in evaluate(W, Xtune, Ytune, Xte, Yte).items():
                R['PCR'][q][mt].append(v)

            pls = PLSRegression(n_components=q, scale=False).fit(Xtune, Ytune)
            for mt, v in evaluate(pls.x_weights_, Xtune, Ytune, Xte, Yte).items():
                R['PLS'][q][mt].append(v)

            best_sig = cv_select(
                sigma_grid,
                lambda sig: fold_mse(
                    lambda Xtr, Ytr, dist, bair:
                        spca_hsic(Xtr, rbf_kernel_from_dist(dist, sig), q)['W']),
                default=sigma_grid[0])
            K = rbf_kernel_from_dist(cdist(Ytune, Ytune, 'euclidean'), best_sig)
            W = spca_hsic(Xtune, K, q)['W']
            for mt, v in evaluate(W, Xtune, Ytune, Xte, Yte).items():
                R['HSIC'][q][mt].append(v)

            best_thr = cv_select(
                threshold_grid,
                lambda thr: fold_mse(
                    lambda Xtr, Ytr, dist, bair:
                        bair_projection(Xtr, bair, thr, q)),
                default=threshold_grid[0])
            W = bair_projection(Xtune, bair_sc_full, best_thr, q)
            for mt, v in evaluate(W, Xtune, Ytune, Xte, Yte).items():
                R['Bair'][q][mt].append(v)

            best_lam = cv_select(
                lspca_lambda_grid,
                lambda lam: fold_mse(
                    lambda Xtr, Ytr, dist, bair:
                        lspca_projection(Xtr, Ytr, q, lam)),
                default=lspca_lambda_grid[0])
            W = lspca_projection(Xtune, Ytune, q, best_lam)
            for mt, v in evaluate(W, Xtune, Ytune, Xte, Yte).items():
                R['LSPCA'][q][mt].append(v)

            best_lam = cv_select(
                sispca_lambda_grid,
                lambda lam: fold_mse(
                    lambda Xtr, Ytr, dist, bair:
                        sispca_projection(Xtr, Ytr, q, lam)),
                default=sispca_lambda_grid[0])
            W = sispca_projection(Xtune, Ytune, q, best_lam)
            for mt, v in evaluate(W, Xtune, Ytune, Xte, Yte).items():
                R['sisPCA'][q][mt].append(v)

            best_kap = cv_select(
                kappa_grid,
                lambda kap: fold_mse(
                    lambda Xtr, Ytr, dist, bair:
                        cspca_nystrom(Xtr, Ytr, q, nystrom_m, kap)),
                default=kappa_grid[0])
            W = cspca_nystrom(Xtune, Ytune, q, nystrom_m, best_kap)
            for mt, v in evaluate(W, Xtune, Ytune, Xte, Yte).items():
                R['CSPCA'][q][mt].append(v)

        if verbose and (rep + 1) % max(1, n_reps // 10) == 0:
            print(f"    rep {rep + 1}/{n_reps} done", flush=True)

    return R, methods, list(q_list)


# Run analysis and print results 
def print_table(R, methods, q_list, metric, label):
    print(f"\n{'-' * 84}\n {label}\n{'-' * 84}")
    print(f"{'Method':<9}" + "".join(f"q={q:<15}" for q in q_list))
    for m in methods:
        row = f"{m:<9}"
        for q in q_list:
            v = np.array(R[m][q][metric])
            nfin = np.sum(~np.isnan(v))
            if nfin == 0:
                row += "NA".ljust(17)
            else:
                row += f"{np.nanmean(v):.4f} ({np.nanstd(v) / np.sqrt(nfin):.4f}) "
        print(row)


if __name__ == "__main__":
    n_reps = 100
    q_list = [2,3,4,5,6]

    X, Y = load_data()

    n_folds = 5
    print(f"\n{'=' * 84}")
    print(f" n={X.shape[0]}, p={X.shape[1]}, {n_reps} replicates, q in {q_list}")
    print(f"{'=' * 84}")

    R, methods, qs = run_analysis(X, Y, q_list=q_list, n_reps=n_reps, n_folds=n_folds)

    print_table(R, methods, qs, 'var_expl', "Variance explained (mean, SE)")
    print_table(R, methods, qs, 'mse',      "Mean squared error (mean, SE)")
    print_table(R, methods, qs, 'cov_expl', "Covariance explained (mean, SE)")