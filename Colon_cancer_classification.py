"""
Alon's Colon Cancer Binary Classification Dataset

U. Alon et al. (1999): Broad patterns of gene expression revealed by clustering analysis of tumor
and normal colon tissue probed by oligonucleotide arrays. Proc. Natl. Acad. Sci. USA 96, 6745-
6750

Extracted from the colonCA library from the Bioconductor R package https://bioconductor.org/packages/release/data/experiment/html/colonCA.html
 
n = 62, p = 2000

Reproducibility: 
Nystr\"om landmark sampling seed = 24, split seed = 1994 + rep, cv seed = 1924 + rep

Compares PCA, HSIC-SPCA, Bair, SLCE, LSPCA, sisPCA, CSPCA across
q in {2,3,4,5,6} over 100 replicates. Each replicate uses a stratified 80/20
tuning/test split; hyperparameters are tuned by stratified 5-fold cross-validation
on classification accuracy; variance explained is reported on
the tuning set; accuracy, precision and AUC are reported on the held-out test set.

Results are deterministic given these seeds and the pinned dependencies
in requirements.txt.

Dependencies:  see requirements.txt; sispca is installed from GitHub:
      pip install git+https://github.com/JiayuSuPKU/sispca.git
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.linalg import eigh, qr, svd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
import autograd.numpy as anp
from pymanopt.manifolds import Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
import pymanopt.function
from sispca import SISPCA, Supervision, SISPCADataset



def delta_kernel(y):
    """Delta kernel: Delta_ij = 1 if y_i == y_j, else 0."""
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


# Nystr\"om CSPCA (m\lceil\sqrt{p}\rceil)
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


# Evaluation metrics - Downstream model is logistic regression
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


def fold_accuracy(W, Xtr, ytr, Xva, yva):
    if W is None:
        return None
    clf = LogisticRegression(penalty=None, max_iter=5000)
    clf.fit(Xtr @ W, ytr)
    pred = clf.predict(Xva @ W)
    return accuracy_score(yva, pred)


# Data loading
def load_data(path="colonCA_combined.csv"):
    data = pd.read_csv(path).to_numpy()
    print("Dataset shape:", data.shape)
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype(float)
    print("X shape:", X.shape, "| classes:", np.unique(y),
          "| counts:", np.bincount(y - y.min()))
    return X, y


# Tuning via stratified 5-fold cross-validation based on classification accuracy
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


def cv_select_max(candidates, fold_scores_fn, default):
    best_cand, best_cv = default, -np.inf
    for cand in candidates:
        scores = fold_scores_fn(cand)
        scores = [s for s in (scores or []) if s is not None and np.isfinite(s)]
        if not scores:
            continue
        mean_cv = float(np.mean(scores))
        if mean_cv > best_cv:
            best_cv, best_cand = mean_cv, cand
    return best_cand


# Run binary classification analysis
def run_analysis(X, y, q_list=(2, 3, 4, 5, 6), n_reps=100, n_folds=5,
                 kappa_grid=(0.001, 0.01, 0.1, 0.5, 1, 5, 10, 20, 50, 100),
                 lspca_lambda_grid=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1),
                 sispca_lambda_grid=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 5, 10.0),
                 threshold_grid=(0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7),
                 nystrom_m=45, verbose=True):
    methods = ['PCA', 'HSIC', 'Bair', 'SLCE', 'LSPCA', 'sisPCA', 'CSPCA']
    metrics = ['accuracy', 'auc', 'precision', 'var_expl']
    R = {m: {q: {mt: [] for mt in metrics} for q in q_list} for m in methods}

    n, p = X.shape

    for rep in range(n_reps):
        rng = np.random.default_rng(1994 + rep)
        tune_idx, te = [], []
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            n_tune_c = int(round(0.8 * len(idx)))
            tune_idx.extend(idx[:n_tune_c].tolist())
            te.extend(idx[n_tune_c:].tolist())
        tune_idx = np.array(sorted(tune_idx)); te = np.array(sorted(te))

        X_tune_raw, y_tune = X[tune_idx], y[tune_idx]
        X_te_raw,   y_te   = X[te],       y[te]
        sx = StandardScaler().fit(X_tune_raw)
        Xtune, Xte = sx.transform(X_tune_raw), sx.transform(X_te_raw)
        K_tune    = delta_kernel(y_tune)
        ypm1_tune = pm1_coding(y_tune)
        bair_full = bair_scores_binary(Xtune, ypm1_tune)
        cv_rng = np.random.default_rng(1924 + rep)
        folds = _stratified_kfold(y_tune, n_folds, cv_rng)

        fold_data = []
        for tr_i, va_i in folds:
            sxf = StandardScaler().fit(X_tune_raw[tr_i])
            Xtr_f = sxf.transform(X_tune_raw[tr_i])
            Xva_f = sxf.transform(X_tune_raw[va_i])
            ytr_f, yva_f = y_tune[tr_i], y_tune[va_i]
            fold_data.append((Xtr_f, ytr_f, Xva_f, yva_f,
                              delta_kernel(ytr_f),
                              pm1_coding(ytr_f),
                              bair_scores_binary(Xtr_f, pm1_coding(ytr_f))))

        def fold_accs(make_W):
            out = []
            for (Xtr_f, ytr_f, Xva_f, yva_f, K_f, ypm1_f, bair_f) in fold_data:
                W = make_W(Xtr_f, ytr_f, K_f, ypm1_f, bair_f)
                out.append(fold_accuracy(W, Xtr_f, ytr_f, Xva_f, yva_f))
            return out

        for q in q_list:
            W = pca_projection(Xtune, q)
            for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te).items():
                R['PCA'][q][mt].append(v)

            W = spca_hsic(Xtune, K_tune, q)
            for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te).items():
                R['HSIC'][q][mt].append(v)

            W = slce_projection(Xtune, y_tune, q)
            for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te).items():
                R['SLCE'][q][mt].append(v)

            best_thr = cv_select_max(
                threshold_grid,
                lambda thr: fold_accs(
                    lambda Xtr, ytr, K, ypm1, bair:
                        bair_projection(Xtr, bair, thr, q)),
                default=threshold_grid[0])
            W = bair_projection(Xtune, bair_full, best_thr, q)
            for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te).items():
                R['Bair'][q][mt].append(v)

            best_lam = cv_select_max(
                lspca_lambda_grid,
                lambda lam: fold_accs(
                    lambda Xtr, ytr, K, ypm1, bair:
                        lspca_logistic_projection(Xtr, ypm1, q, lam)),
                default=lspca_lambda_grid[0])
            W = lspca_logistic_projection(Xtune, ypm1_tune, q, best_lam)
            for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te).items():
                R['LSPCA'][q][mt].append(v)

            best_lam = cv_select_max(
                sispca_lambda_grid,
                lambda lam: fold_accs(
                    lambda Xtr, ytr, K, ypm1, bair:
                        sispca_projection(Xtr, ytr, q, lam)),
                default=sispca_lambda_grid[0])
            W = sispca_projection(Xtune, y_tune, q, best_lam)
            for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te).items():
                R['sisPCA'][q][mt].append(v)

            best_kap = cv_select_max(
                kappa_grid,
                lambda kap: fold_accs(
                    lambda Xtr, ytr, K, ypm1, bair:
                        cspca_delta_nystrom(Xtr, K, q, nystrom_m, kap)),
                default=kappa_grid[0])
            W = cspca_delta_nystrom(Xtune, K_tune, q, nystrom_m, best_kap)
            for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te).items():
                R['CSPCA'][q][mt].append(v)

        if verbose and (rep + 1) % max(1, n_reps // 10) == 0:
            print(f"    rep {rep + 1}/{n_reps} done", flush=True)

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
    q_list = [2, 3, 4, 5, 6]
    n_folds = 5

    X, y = load_data("colonCA_combined.csv")

    print(f"\n{'=' * 90}")
    print(f" n={X.shape[0]}, p={X.shape[1]}, {n_reps} replicates, q in {q_list}")
    print(f"{'=' * 90}")

    R, methods, qs = run_analysis(X, y, q_list=q_list, n_reps=n_reps,
                                  n_folds=n_folds)

    print_table(R, methods, qs, 'accuracy',  "Accuracy (mean, SE)")
    print_table(R, methods, qs, 'auc',       "AUC (mean, SE)")
    print_table(R, methods, qs, 'precision', "Precision (mean, SE)")
    print_table(R, methods, qs, 'var_expl',  "Variance explained, training (mean, SE)")