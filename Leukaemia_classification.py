"""
Golub's Leukaemia Binary Classification Dataset 

Golub et al. (1999) Molecular classification of cancer: class discovery and class prediction by
gene expression monitoring. Science, 286, 531-537.

Extracted from OpenML 3 https://www.openml.org/search?type=data&sort=runs&id=1104&status=active
 
n = 72, p = 7129
The dataset is already divide into training (n_tr=38) and test (n_te=34) sets.

The analysis was run on a high performance computing cluster, parallelised over the 100 replicates with
joblib; n_jobs = 50 (one core per replicate). BLAS threading is pinned
to one thread per process (see environment variables below) to prevent
oversubscription, so the job uses 50 cores total.

Reproducibility: 
Nystr\"om landmark sampling seed = 24, cv seed = 3000 + rep

Compares PCA, HSIC-SPCA, Bair, SLCE, LSPCA, sisPCA, CSPCA across
q in {2,3,4,5,6} over 100 replicates. Data already divided into training and test; 
hyperparameters are tuned by 5-fold cross-validation
on classification accuracy; variance explained is reported on
the training set; accuracy, precision and AUC are reported on the test set.

Results are deterministic given these seeds and the pinned dependencies
in requirements.txt.

Dependencies:  see requirements.txt; sispca is installed from GitHub:
      pip install git+https://github.com/JiayuSuPKU/sispca.git
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.linalg import eigh, qr, svd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
from joblib import Parallel, delayed
import autograd.numpy as anp
from pymanopt.manifolds import Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
import pymanopt.function
from sispca import SISPCA, Supervision, SISPCADataset

# Data loading and cleaning
y = pd.read_csv("actual.csv")
data_2 = pd.read_csv("data_set_ALL_AML_independent.csv")
data_3 = pd.read_csv("data_set_ALL_AML_train.csv")
print(y.shape)
print(data_2.shape)
print(data_3.shape)

Y = y.replace({'ALL':0,'AML':1})
labels = ['ALL', 'AML']
print(Y)

train_to_keep = [col for col in data_3.columns if "call" not in col]
test_to_keep = [col for col in data_2.columns if "call" not in col]

X_train_tr = data_3[train_to_keep]
X_test_tr = data_2[test_to_keep]

train_columns_titles = ['Gene Description', 'Gene Accession Number', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
       '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']

X_train_tr = X_train_tr.reindex(columns=train_columns_titles)

test_columns_titles = ['Gene Description', 'Gene Accession Number','39', '40', '41', '42', '43', '44', '45', '46',
       '47', '48', '49', '50', '51', '52', '53',  '54', '55', '56', '57', '58', '59',
       '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72']

X_test_tr = X_test_tr.reindex(columns=test_columns_titles)

X_train = X_train_tr.T
X_test = X_test_tr.T
print(X_train.shape)
print(X_test.shape)

X_train.columns = X_train.iloc[1]
X_train = X_train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)
X_test.columns = X_test.iloc[1]
X_test = X_test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

print(X_train.shape)
print(X_test.shape)
X_train.head()
X_train = X_train.reset_index(drop=True)
Y_train = Y[Y.patient <= 38].reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
Y_test = Y[Y.patient > 38].reset_index(drop=True)
X_train_fl = X_train.astype(float, 64)
X_test_fl = X_test.astype(float, 64)

scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train_fl)
X_test_scl = scaler.transform(X_test_fl)

Y_train = Y_train['cancer']
Y_test = Y_test['cancer']
print(Y_train.shape)
print(Y_test.shape)

# Helper functions
def delta_kernel(y):
    y = np.asarray(y).ravel()
    return (y[:, None] == y[None, :]).astype(float)

def pm1_coding(y):
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    assert len(classes) == 2
    return np.where(y == classes[1], 1.0, -1.0).reshape(-1, 1)

def centroid_matrix(X, y):
    y = np.asarray(y).ravel()
    Cbar = np.zeros_like(X)
    for c in np.unique(y):
        m = (y == c)
        Cbar[m, :] = X[m, :].mean(axis=0)
    return Cbar


# PCA
def pca_projection(X, q):
    return PCA(n_components=q).fit(X).components_.T

# HSIC-SPCA
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
    if pos.sum() == 0:
        return None
    evals_pos = evals[pos]
    Vc = Vc[:, pos]
    S_inv = np.diag(1.0 / np.sqrt(evals_pos))
    U = C @ Vc @ S_inv
    return U

# Bair's
def bair_scores_binary(X, y_pm1):
    num = X.T @ y_pm1.ravel()
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

# SLCE
def slce_projection(X, y, q):
    Cbar = centroid_matrix(X, y)
    M = Cbar.T @ X + X.T @ Cbar - X.T @ X
    M = (M + M.T) / 2.0
    ev, V = eigh(M)
    return V[:, np.argsort(ev)[::-1][:q]]

# LSPCA
def lspca_logistic_projection(X, y_pm1, q, lam, max_iters=150):
    p = X.shape[1]
    yv = y_pm1.ravel()
    pca_fallback = PCA(n_components=q).fit(X).components_.T
    manifold = Product([Grassmann(p, q), Euclidean(q, 1)])

    @pymanopt.function.autograd(manifold)
    def cost(L, beta):
        XL = X @ L
        margins = yv * (XL @ beta).ravel()
        logistic = anp.sum(anp.logaddexp(0.0, -margins))
        rec = anp.sum((X - XL @ L.T) ** 2)
        return logistic + lam * rec

    L_init = pca_fallback
    beta_init = np.zeros((q, 1))
    optimizer = ConjugateGradient(verbosity=0, max_iterations=max_iters)
    result = optimizer.run(Problem(manifold=manifold, cost=cost),
                               initial_point=[L_init, beta_init])
    W = result.point[0]  
    return W

# sisPCA
def sispca_projection(X, y, q, lam, max_epochs=50, patience=5):
    yv = np.asarray(y).ravel().reshape(-1, 1)
    supervision = [Supervision(yv, target_type='categorical')]
    dataset = SISPCADataset(X, target_supervision_list=supervision)
    model = SISPCA(dataset, n_latent_sub=[q], lambda_contrast=lam,
                   kernel_subspace="gaussian")
    model.fit(batch_size=-1, max_epochs=max_epochs,
              early_stopping_patience=patience,
              enable_progress_bar=False, enable_model_summary=False)
    return model._get_U_subspace_list()[0].detach().cpu().numpy()

# Nystr\"om CSPCA, m=\lceil\sqrt{p}\rceil
def cspca_delta_nystrom(X, K, q, m, kappa, seed=24):
    p = X.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(p, size=min(m, p), replace=False)
    Xm = X[:, idx]
    S = X.T @ (K @ Xm) + kappa * (X.T @ Xm)
    Cm = S[idx, :]
    Cm = (Cm + Cm.T) / 2.0
    ev, Um = eigh(Cm)
    o = np.argsort(ev)[::-1]; ev, Um = ev[o], Um[:, o]
    pos = ev > 1e-10 * max(ev.max(), 1e-300)
    if pos.sum() == 0:
        return None
    B = S @ Um[:, pos] @ np.diag(1.0 / np.sqrt(ev[pos]))
    Q, Rmat = qr(B, mode='economic')
    theta, V = eigh(Rmat @ Rmat.T)
    oo = np.argsort(theta)[::-1]
    return (Q @ V[:, oo])[:, :q]


# Evaluation metrics -- Downstream model is logistic regression
def variance_explained(X, W):
    d = np.linalg.norm(X) ** 2
    return np.linalg.norm(X @ W) ** 2 / d if d > 0 else 0.0


def evaluate(W, Xtr, ytr, Xte, yte):
    if W is None:
        return {"accuracy": np.nan, "auc": np.nan,
                "precision": np.nan, "var_expl": np.nan}
    clf = LogisticRegression(penalty=None, max_iter=5000)
    clf.fit(Xtr @ W, ytr)
    proba = clf.predict_proba(Xte @ W)[:, 1]
    pred = clf.predict(Xte @ W)
    return {
        "accuracy":  accuracy_score(yte, pred),
        "auc":       roc_auc_score(yte, proba),
        "precision": precision_score(yte, pred, zero_division=0.0),
        "var_expl":  variance_explained(Xtr, W),
    }


# Tuning via stratified 5-fold cross-validation within the training set on classification accuracy
def _stratified_kfold(y, k, rng):
    y = np.asarray(y).ravel()
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        for j, chunk in enumerate(np.array_split(idx, k)):
            folds[j].extend(chunk.tolist())
    splits = []
    for i in range(k):
        va = np.array(sorted(folds[i]), dtype=int)
        tr = np.array(sorted(sum((folds[j] for j in range(k) if j != i), [])),
                      dtype=int)
        splits.append((tr, va))
    return splits

def _fold_acc_for_W(W, Xtr, ytr, Xva, yva):
    if W is None:
        return None
    clf = LogisticRegression(penalty=None, max_iter=5000)
    clf.fit(Xtr @ W, ytr)
    return accuracy_score(yva, clf.predict(Xva @ W))


def _cv_tune_bair(fold_data, q, threshold_grid):
    best_thr, best_cv = threshold_grid[0], -np.inf
    for thr in threshold_grid:
        accs = []
        for Xtr_f, ytr_f, Xva_f, yva_f, K_f, ypm1_f, bair_f in fold_data:
            W = bair_projection(Xtr_f, bair_f, thr, q)
            accs.append(_fold_acc_for_W(W, Xtr_f, ytr_f, Xva_f, yva_f))
        valid = [a for a in accs if a is not None and np.isfinite(a)]
        if valid and np.mean(valid) > best_cv:
            best_cv, best_thr = np.mean(valid), thr
    return best_thr


def _cv_tune_lspca(fold_data, q, lambda_grid):
    best_lam, best_cv = lambda_grid[0], -np.inf
    for lam in lambda_grid:
        accs = []
        for Xtr_f, ytr_f, Xva_f, yva_f, K_f, ypm1_f, bair_f in fold_data:
            W = lspca_logistic_projection(Xtr_f, ypm1_f, q, lam)
            accs.append(_fold_acc_for_W(W, Xtr_f, ytr_f, Xva_f, yva_f))
        valid = [a for a in accs if a is not None and np.isfinite(a)]
        if valid and np.mean(valid) > best_cv:
            best_cv, best_lam = np.mean(valid), lam
    return best_lam


def _cv_tune_sispca(fold_data, q, lambda_grid):
    best_lam, best_cv = lambda_grid[0], -np.inf
    for lam in lambda_grid:
        accs = []
        for Xtr_f, ytr_f, Xva_f, yva_f, K_f, ypm1_f, bair_f in fold_data:
            W = sispca_projection(Xtr_f, ytr_f, q, lam)
            accs.append(_fold_acc_for_W(W, Xtr_f, ytr_f, Xva_f, yva_f))
        valid = [a for a in accs if a is not None and np.isfinite(a)]
        if valid and np.mean(valid) > best_cv:
            best_cv, best_lam = np.mean(valid), lam
    return best_lam


def _cv_tune_cspca(fold_data, q, kappa_grid, nystrom_m):
    best_kap, best_cv = kappa_grid[0], -np.inf
    for kap in kappa_grid:
        accs = []
        for Xtr_f, ytr_f, Xva_f, yva_f, K_f, ypm1_f, bair_f in fold_data:
            W = cspca_delta_nystrom(Xtr_f, K_f, q, nystrom_m, kap)
            accs.append(_fold_acc_for_W(W, Xtr_f, ytr_f, Xva_f, yva_f))
        valid = [a for a in accs if a is not None and np.isfinite(a)]
        if valid and np.mean(valid) > best_cv:
            best_cv, best_kap = np.mean(valid), kap
    return best_kap


# Run classification analysis
def _run_single_rep(rep, X_train, y_train, X_test, y_test,
                    X_train_raw, q_list, n_folds,
                    K_train, ypm1_train, bair_full,
                    kappa_grid, lspca_lambda_grid, sispca_lambda_grid,
                    threshold_grid, nystrom_m):
    
    results = {}
    cv_rng = np.random.default_rng(3000 + rep)
    folds = _stratified_kfold(y_train, n_folds, cv_rng)

    fold_data = []
    for tr_i, va_i in folds:
        sxf = StandardScaler().fit(X_train_raw[tr_i])
        Xtr_f = sxf.transform(X_train_raw[tr_i])
        Xva_f = sxf.transform(X_train_raw[va_i])
        ytr_f, yva_f = y_train[tr_i], y_train[va_i]
        fold_data.append((Xtr_f, ytr_f, Xva_f, yva_f,
                          delta_kernel(ytr_f),
                          pm1_coding(ytr_f),
                          bair_scores_binary(Xtr_f, pm1_coding(ytr_f))))

    for q in q_list:
        W = pca_projection(X_train, q)
        for mt, v in evaluate(W, X_train, y_train, X_test, y_test).items():
            results[('PCA', q, mt)] = v

        W = spca_hsic(X_train, K_train, q)
        for mt, v in evaluate(W, X_train, y_train, X_test, y_test).items():
            results[('HSIC', q, mt)] = v

        W = slce_projection(X_train, y_train, q)
        for mt, v in evaluate(W, X_train, y_train, X_test, y_test).items():
            results[('SLCE', q, mt)] = v

        best_thr = _cv_tune_bair(fold_data, q, threshold_grid)
        W = bair_projection(X_train, bair_full, best_thr, q)
        for mt, v in evaluate(W, X_train, y_train, X_test, y_test).items():
            results[('Bair', q, mt)] = v

        best_lam = _cv_tune_lspca(fold_data, q, lspca_lambda_grid)
        W = lspca_logistic_projection(X_train, ypm1_train, q, best_lam)
        for mt, v in evaluate(W, X_train, y_train, X_test, y_test).items():
            results[('LSPCA', q, mt)] = v

        best_lam = _cv_tune_sispca(fold_data, q, sispca_lambda_grid)
        W = sispca_projection(X_train, y_train, q, best_lam)
        for mt, v in evaluate(W, X_train, y_train, X_test, y_test).items():
            results[('sisPCA', q, mt)] = v

        best_kap = _cv_tune_cspca(fold_data, q, kappa_grid, nystrom_m)
        W = cspca_delta_nystrom(X_train, K_train, q, nystrom_m, best_kap)
        for mt, v in evaluate(W, X_train, y_train, X_test, y_test).items():
            results[('CSPCA', q, mt)] = v

    return results


# Run classification analysis
def run_analysis(X_train_raw, y_train, X_test_raw, y_test,
                 q_list=(2, 3, 4, 5, 6), n_reps=100, n_folds=5,
                 kappa_grid=(0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100),
                 lspca_lambda_grid=(0.001, 0.0025, 0.005, 0.075, 0.01, 0.05, 0.075, 0.1, 0.5, 1),
                 sispca_lambda_grid=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 5),
                 threshold_grid=(0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7),
                 nystrom_m=None, n_jobs=50, verbose=False):
    
    methods = ['PCA', 'HSIC', 'Bair', 'SLCE', 'LSPCA', 'sisPCA', 'CSPCA']
    metrics = ['accuracy', 'auc', 'precision', 'var_expl']
    R = {m: {q: {mt: [] for mt in metrics} for q in q_list} for m in methods}

    n_train, p = X_train_raw.shape
    if nystrom_m is None:
        nystrom_m = int(np.ceil(np.sqrt(p)))

    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    sx = StandardScaler().fit(X_train_raw)
    X_train = sx.transform(X_train_raw)
    X_test = sx.transform(X_test_raw)

    K_train    = delta_kernel(y_train)
    ypm1_train = pm1_coding(y_train)
    bair_full  = bair_scores_binary(X_train, ypm1_train)

    verbosity = 25 if verbose else 0
    all_results = Parallel(n_jobs=n_jobs, verbose=verbosity)(
        delayed(_run_single_rep)(
            rep, X_train, y_train, X_test, y_test,
            X_train_raw, list(q_list), n_folds,
            K_train, ypm1_train, bair_full,
            list(kappa_grid), list(lspca_lambda_grid),
            list(sispca_lambda_grid), list(threshold_grid), nystrom_m
        )
        for rep in range(n_reps)
    )

    for rep_results in all_results:
        for (method, q, metric), value in rep_results.items():
            R[method][q][metric].append(value)

    return R, methods, list(q_list)


# Run analysis and print results
def print_table(R, methods, q_list, metric, label):
    print(f"\n{'-' * 90}\n {label}\n{'-' * 90}")
    print(f"{'Method':<9}" + "".join(f"q={q:<15}" for q in q_list))
    for m in methods:
        row = f"{m:<9}"
        for q in q_list:
            v = np.array(R[m][q][metric], dtype=float)
            nfin = np.sum(~np.isnan(v))
            if nfin == 0:
                row += "NA".ljust(17)
            else:
                row += f"{np.nanmean(v):.4f} ({np.nanstd(v)/np.sqrt(nfin):.4f}) "
        print(row)



if __name__ == "__main__":

    n_reps = 100
    q_list = [2,3,4,5,6]
    n_folds = 5
    n_jobs = 50 # Uses 50 CPU cores
    p = X_train_scl.shape[1]

    print(f"\n{'=' * 90}")
    print(f" n_train={X_train_scl.shape[0]}, n_test={X_test_scl.shape[0]}, "
          f"p={p}, {n_reps} replicates, q in {q_list}")
    print(f" Nystrom landmarks m = ceil(sqrt({p})) = {int(np.ceil(np.sqrt(p)))}")
    print(f" Classes: {np.unique(Y_train)}, "
          f"train counts: {np.bincount(Y_train - Y_train.min())}, "
          f"test counts: {np.bincount(Y_test - Y_test.min())}")
    print(f" Parallelism: n_jobs={n_jobs}")
    print(f"{'=' * 90}")

    R, methods, qs = run_analysis(X_train_scl, Y_train, X_test_scl, Y_test,
                                  q_list=q_list, n_reps=n_reps,
                                  n_folds=n_folds, n_jobs=n_jobs)

    print_table(R, methods, qs, 'accuracy',  "Accuracy (mean, SE)")
    print_table(R, methods, qs, 'auc',       "AUC (mean, SE)")
    print_table(R, methods, qs, 'precision', "Precision (mean, SE)")
    print_table(R, methods, qs, 'var_expl', "Variance explained, training (mean, SE)")