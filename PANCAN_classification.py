"""
PANCAN Multiclass Classification Dataset 

Cancer Genome Atlas Research Network et al. (2013). The Cancer Genome Atlas Pan-Cancer analysis project. Nature genetics, 45(10), 1113–1120.

Extracted from UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq
 
n = 801, p = 20531, M = 5 classes (BRCA, KIRC, LUAD, PRAD, COAD)

The analysis was run on a high performance computing cluster, parallelised over the 100 replicates with
joblib; n_jobs = 50 (one core per replicate). BLAS threading is pinned
to one thread per process (see environment variables below) to prevent
oversubscription, so the job uses 50 cores total.

Reproducibility: 
Nystr\"om landmark sampling seed = 24, split seed = 1994 + rep, cv seed = 1924 + rep

Compares PCA, HSIC-SPCA, Bair, SLCE, LSPCA, sisPCA, CSPCA across
q in {2,3,4,5,6} over 100 replicates. Each replicate uses a stratified 80/20
tuning/test split; hyperparameters are tuned by stratified 5-fold cross-validation
on classification accuracy; variance explained is reported on
the tuning set; accuracy, precision, F1 score and AUC are reported on the held-out test set.
Per-class results on precision, recall and F1 are also reported on the held-out test set.

Results are deterministic given these seeds and the pinned dependencies
in requirements.txt.

Dependencies:  see requirements.txt; sispca is installed from GitHub:
      pip install git+https://github.com/JiayuSuPKU/sispca.git
"""

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.linalg import eigh, qr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,precision_score, recall_score, f1_score,
precision_recall_fscore_support)
from joblib import Parallel, delayed
import autograd.numpy as anp
from pymanopt.manifolds import Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
import pymanopt.function
from sispca import SISPCA, Supervision, SISPCADataset
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# Data Loading
def load_icmr(data_path="~/Documents/TP/data.csv", labels_path="~/Documents/TP/labels.csv"):
    data_df = pd.read_csv(data_path)
    labels_df = pd.read_csv(labels_path)

    for df in [data_df, labels_df]:
        unnamed = [c for c in df.columns if "Unnamed" in str(c)]
        if unnamed:
            df.drop(columns=unnamed, inplace=True)

    merged = pd.concat([data_df.reset_index(drop=True),
                        labels_df.reset_index(drop=True)], axis=1)

    y_raw = merged['Class'].values
    classes = np.unique(y_raw)
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[v] for v in y_raw])
    X = merged.drop(columns=['Class']).values.astype(float)

    print(f"Loaded ICMR: X {X.shape}, {len(classes)} classes")
    print(f"  Classes: {list(classes)}")
    print(f"  Counts:  {dict(zip(classes, np.bincount(y)))}")
    return X, y, classes


# Helper functions
def delta_kernel(y):
    y = np.asarray(y).ravel()
    return (y[:, None] == y[None, :]).astype(float)


def one_hot(y, C=None):
    y = np.asarray(y).ravel().astype(int)
    if C is None:
        C = int(y.max() + 1)
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def centroid_matrix(X, y):
    y = np.asarray(y).ravel()
    Cbar = np.zeros_like(X)
    for c in np.unique(y):
        Cbar[y == c, :] = X[y == c, :].mean(axis=0)
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
    S_inv = np.diag(1.0 / np.sqrt(evals[pos]))
    return C @ Vc[:, pos] @ S_inv

# Bair's
def bair_scores_multiclass(X, y):
    n = X.shape[0]
    classes = np.unique(y)
    C = len(classes)
    grand_mean = X.mean(axis=0)
    ss_between = np.zeros(X.shape[1])
    ss_within = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        ss_between += Xc.shape[0] * (mc - grand_mean) ** 2
        ss_within += ((Xc - mc) ** 2).sum(axis=0)
    ms_between = ss_between / max(C - 1, 1)
    ms_within = ss_within / max(n - C, 1)
    ms_within[ms_within == 0] = np.inf
    return ms_between / ms_within


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
def lspca_softmax_projection(X, y, q, lam, max_iters=150):
     p = X.shape[1]
     C = int(np.max(y)) + 1
     yv = np.asarray(y).ravel().astype(int)
     pca_fallback = PCA(n_components=q).fit(X).components_.T
     manifold = Product([Grassmann(p, q), Euclidean(q, C)])

     @pymanopt.function.autograd(manifold)
     def cost(L, beta):
         XL = X @ L
         scores = XL @ beta
         max_s = anp.max(scores, axis=1, keepdims=True)
         log_denom = max_s.ravel() + anp.log(anp.sum(
             anp.exp(scores - max_s), axis=1))
         picked = anp.sum(scores * one_hot(yv, C), axis=1)
         nll = anp.sum(log_denom - picked)
         rec = anp.sum((X - XL @ L.T) ** 2)
         return nll + lam * rec

     L_init = pca_fallback
     beta_init = np.zeros((q, C))
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
    return (Q @ V[:, np.argsort(theta)[::-1]])[:, :q]


# Evaluation metrics, downstream model is multinomial logistic regression
def variance_explained(X, W):
    d = np.linalg.norm(X) ** 2
    return np.linalg.norm(X @ W) ** 2 / d if d > 0 else 0.0


def evaluate(W, Xtr, ytr, Xte, yte, n_classes):
    if W is None:
        out = {"accuracy": np.nan, "auc": np.nan,
               "precision_macro": np.nan, "recall_macro": np.nan,
               "f1_macro": np.nan, "var_expl": np.nan}
        for c in range(n_classes):
            out[f"precision_c{c}"] = np.nan
            out[f"recall_c{c}"] = np.nan
            out[f"f1_c{c}"] = np.nan
        return out

    clf = LogisticRegression(penalty=None, max_iter=5000)
    clf.fit(Xtr @ W, ytr)
    proba = clf.predict_proba(Xte @ W)
    pred = clf.predict(Xte @ W)

    auc = roc_auc_score(yte, proba, multi_class='ovr',
                        average='macro', labels=np.arange(n_classes))

    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(
        yte, pred, labels=np.arange(n_classes), average=None,
        zero_division=0.0)

    out = {
        "accuracy":        accuracy_score(yte, pred),
        "auc":             auc,
        "precision_macro": precision_score(yte, pred, average='macro',
                                           zero_division=0.0),
        "recall_macro":    recall_score(yte, pred, average='macro',
                                        zero_division=0.0),
        "f1_macro":        f1_score(yte, pred, average='macro',
                                    zero_division=0.0),
        "var_expl":        variance_explained(Xtr, W),
    }
    for c in range(n_classes):
        out[f"precision_c{c}"] = prec_per[c]
        out[f"recall_c{c}"]    = rec_per[c]
        out[f"f1_c{c}"]        = f1_per[c]
    return out


def fold_accuracy(W, Xtr, ytr, Xva, yva):
    if W is None:
        return None
    clf = LogisticRegression(penalty=None, max_iter=5000)
    clf.fit(Xtr @ W, ytr)
    return accuracy_score(yva, clf.predict(Xva @ W))


# Tuning via stratified 5-fold cross-validation on classification accuracy
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


def _cv_tune_bair(fold_data, q, threshold_grid):
    best_thr, best_cv = threshold_grid[0], -np.inf
    for thr in threshold_grid:
        accs = []
        for Xtr_f, ytr_f, Xva_f, yva_f, K_f, bair_f in fold_data:
            W = bair_projection(Xtr_f, bair_f, thr, q)
            accs.append(fold_accuracy(W, Xtr_f, ytr_f, Xva_f, yva_f))
        valid = [a for a in accs if a is not None and np.isfinite(a)]
        if valid and np.mean(valid) > best_cv:
            best_cv, best_thr = np.mean(valid), thr
    return best_thr


def _cv_tune_lspca(fold_data, q, lambda_grid):
     best_lam, best_cv = lambda_grid[0], -np.inf
     for lam in lambda_grid:
         accs = []
         for Xtr_f, ytr_f, Xva_f, yva_f, K_f, bair_f in fold_data:
             W = lspca_softmax_projection(Xtr_f, ytr_f, q, lam)
             accs.append(fold_accuracy(W, Xtr_f, ytr_f, Xva_f, yva_f))
         valid = [a for a in accs if a is not None and np.isfinite(a)]
         if valid and np.mean(valid) > best_cv:
             best_cv, best_lam = np.mean(valid), lam
     return best_lam


def _cv_tune_sispca(fold_data, q, lambda_grid):
    best_lam, best_cv = lambda_grid[0], -np.inf
    for lam in lambda_grid:
        accs = []
        for Xtr_f, ytr_f, Xva_f, yva_f, K_f, bair_f in fold_data:
            W = sispca_projection(Xtr_f, ytr_f, q, lam)
            accs.append(fold_accuracy(W, Xtr_f, ytr_f, Xva_f, yva_f))
        valid = [a for a in accs if a is not None and np.isfinite(a)]
        if valid and np.mean(valid) > best_cv:
            best_cv, best_lam = np.mean(valid), lam
    return best_lam


def _cv_tune_cspca(fold_data, q, kappa_grid, nystrom_m):
    best_kap, best_cv = kappa_grid[0], -np.inf
    for kap in kappa_grid:
        accs = []
        for Xtr_f, ytr_f, Xva_f, yva_f, K_f, bair_f in fold_data:
            W = cspca_delta_nystrom(Xtr_f, K_f, q, nystrom_m, kap)
            accs.append(fold_accuracy(W, Xtr_f, ytr_f, Xva_f, yva_f))
        valid = [a for a in accs if a is not None and np.isfinite(a)]
        if valid and np.mean(valid) > best_cv:
            best_cv, best_kap = np.mean(valid), kap
    return best_kap


# Run multiclass classification analysis 
def _run_single_rep(rep, X, y, n_classes, q_list, n_folds,
                    kappa_grid, sispca_lambda_grid,lspca_lambda_grid,
                    threshold_grid, nystrom_m):
    results = {}

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
    bair_full = bair_scores_multiclass(Xtune, y_tune)

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
                          bair_scores_multiclass(Xtr_f, ytr_f)))

    for q in q_list:
        W = pca_projection(Xtune, q)
        for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te, n_classes).items():
            results[('PCA', q, mt)] = v

        W = spca_hsic(Xtune, K_tune, q)
        for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te, n_classes).items():
            results[('HSIC', q, mt)] = v

        W = slce_projection(Xtune, y_tune, q)
        for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te, n_classes).items():
            results[('SLCE', q, mt)] = v

        best_thr = _cv_tune_bair(fold_data, q, threshold_grid)
        W = bair_projection(Xtune, bair_full, best_thr, q)
        for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te, n_classes).items():
            results[('Bair', q, mt)] = v

        best_lam = _cv_tune_lspca(fold_data, q, lspca_lambda_grid)
        W = lspca_softmax_projection(Xtune, y_tune, q, best_lam)
        for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te, n_classes).items():
             results[('LSPCA', q, mt)] = v

        best_lam = _cv_tune_sispca(fold_data, q, sispca_lambda_grid)
        W = sispca_projection(Xtune, y_tune, q, best_lam)
        for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te, n_classes).items():
            results[('sisPCA', q, mt)] = v

        best_kap = _cv_tune_cspca(fold_data, q, kappa_grid, nystrom_m)
        W = cspca_delta_nystrom(Xtune, K_tune, q, nystrom_m, best_kap)
        for mt, v in evaluate(W, Xtune, y_tune, Xte, y_te, n_classes).items():
            results[('CSPCA', q, mt)] = v

    return results


# Run analysis and print results
def run_analysis(X, y, n_classes, class_names,
                 q_list=(2, 3, 4, 5, 6), n_reps=100, n_folds=5,
                 kappa_grid=(0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100),
                 sispca_lambda_grid=(0.01, 0.05, 0.1, 0.5, 1.0, 5, 10.0),
                 lspca_lambda_grip= (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 1),
                 threshold_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                 nystrom_m=None, n_jobs=50, verbose=True):

    methods = ['PCA', 'HSIC', 'Bair', 'LSPCA' ,'SLCE', 'sisPCA', 'CSPCA']
    macro_metrics = ['accuracy', 'auc', 'precision_macro', 'recall_macro',
                     'f1_macro', 'var_expl']
    per_class = []
    for c in range(n_classes):
        per_class += [f'precision_c{c}', f'recall_c{c}', f'f1_c{c}']
    all_metrics = macro_metrics + per_class

    R = {m: {q: {mt: [] for mt in all_metrics} for q in q_list}
         for m in methods}

    p = X.shape[1]
    if nystrom_m is None:
        nystrom_m = int(np.ceil(np.sqrt(p)))

    verbosity = 10 if verbose else 0
    all_results = Parallel(n_jobs=n_jobs, verbose=verbosity)(
        delayed(_run_single_rep)(
            rep, X, y, n_classes, list(q_list), n_folds,
            list(kappa_grid), list(sispca_lambda_grid), list(lspca_lambda_grip),
            list(threshold_grid), nystrom_m
        )
        for rep in range(n_reps)
    )

    for rep_results in all_results:
        for (method, q, metric), value in rep_results.items():
            if method in R and q in R[method] and metric in R[method][q]:
                R[method][q][metric].append(value)

    return R, methods, list(q_list)



def print_macro_table(R, methods, q_list, metric, label):
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


def print_perclass_table(R, methods, q_list, metric_prefix, label,
                         n_classes, class_names):
    for q in q_list:
        print(f"\n{'-' * 70}")
        print(f" {label} at q={q}")
        print(f"{'-' * 70}")
        header = f"{'Method':<9}" + "".join(f"{cn:<12}" for cn in class_names) + "  macro"
        print(header)
        for m in methods:
            row = f"{m:<9}"
            for c in range(n_classes):
                key = f"{metric_prefix}_c{c}"
                v = np.array(R[m][q][key], dtype=float)
                nfin = np.sum(~np.isnan(v))
                if nfin == 0:
                    row += "NA".ljust(12)
                else:
                    row += f"{np.nanmean(v):.3f}       "
            macro_key = f"{metric_prefix}_macro"
            v = np.array(R[m][q][macro_key], dtype=float)
            nfin = np.sum(~np.isnan(v))
            if nfin == 0:
                row += "NA"
            else:
                row += f"{np.nanmean(v):.3f} ({np.nanstd(v)/np.sqrt(nfin):.3f})"
            print(row)



if __name__ == "__main__":
    X, y, class_names = load_icmr()
    n_classes = len(class_names)
    p = X.shape[1]

    n_reps = 100
    q_list = [2,3,4,5,6]
    n_folds = 5
    n_jobs = 50 # Uses 50 CPU cores

    print(f"\n{'=' * 90}")
    print(f" n={X.shape[0]}, p={p}, {n_reps} replicates, q in {q_list}")
    print(f" Classes: {list(class_names)}")
    print(f" Tuning: stratified {n_folds}-fold CV on classification accuracy")
    print(f" Downstream classifier: unpenalised multinomial logistic regression")
    print(f" Nystrom landmarks: m = ceil(sqrt({p})) = {int(np.ceil(np.sqrt(p)))}")
    print(f" Parallelism: n_jobs={n_jobs}")
    print(f"{'=' * 90}")

    R, methods, qs = run_analysis(X, y, n_classes, class_names,
                                  q_list=q_list, n_reps=n_reps,
                                  n_folds=n_folds, n_jobs=n_jobs)

    # Macro results
    print_macro_table(R, methods, qs, 'accuracy',        "Accuracy (mean, SE)")
    print_macro_table(R, methods, qs, 'auc',             "AUC macro OvR (mean, SE)")
    print_macro_table(R, methods, qs, 'precision_macro',  "Precision macro (mean, SE)")
    print_macro_table(R, methods, qs, 'recall_macro',     "Recall macro (mean, SE)")
    print_macro_table(R, methods, qs, 'f1_macro',         "F1 macro (mean, SE)")
    print_macro_table(R, methods, qs, 'var_expl',         "Variance explained (mean, SE)")

    # Per-class results
    print(f"\n{'=' * 70}")
    print(" PER-CLASS BREAKDOWN")
    print(f"{'=' * 70}")
    print_perclass_table(R, methods, qs, 'precision', 'Precision per class',
                         n_classes, class_names)
    print_perclass_table(R, methods, qs, 'recall', 'Recall per class',
                         n_classes, class_names)
    print_perclass_table(R, methods, qs, 'f1', 'F1 per class',
                         n_classes, class_names)