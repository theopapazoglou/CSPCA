# CSPCA: Covariance Supervised Principal Component Analysis

This repository contains reference implementation and experiments for CSPCA, a supervised dimensionality reduction technique that combines covariance maximisation with variance preservation in a single eigenvalue problem, controlled by a regularisation parameter κ>0. Within this repository, the code needed to reproduce all simulation studies and real-data analyses reported in the paper may be found.

# Method
CSPCA extracts a projection matrix $W$ as the top-$q$ eigenvectors of $C=X^\top YY^\top X+\kappa X^\top X$. The supervised term captures the association between the features and the response, while the reconstruction term preserves the feature variance and guarantees a well-defined $q$-dimensional projection even when the supervised signal alone is rank-deficient. For high-dimensional problems, a Nystr\"om approximation of $C$ is used to avoid forming the full $p\times p$ matrix. For categorical responses, $YY^\top$ is replaced by the delta kernel.

# Repository Structure
Each script is self-contained. It includes library imports, all helper functions, the analysis routine, and a __main__ block that runs the experiment and prints the result tables. A header docstring at the top of each file describesthe specific experiment it runs.

# Simulations
Simulation1.py contains the single-scalar linear simulation scenario for both i.i.d.\ and correlated DGPs, while Simulation2.py contains the single-scalar non-linear scenario for both i.i.d.\ and correlated data DGPs. Simulation3.py contains the multivariate linear response scenario for both i.i.d.\ and correlated DGPs, while Simulation4.py contains the multivariate non-linear scenario for both i.i.d.\ and correlated DGPs.

# Real Data Analyses
Liver_toxicity_regression.py contains the regression analysis based on a rat liver toxicity expression dataset (Bushel et al., 2007) extracted from the mixOmics library of the Bioconductor R Package (https://www.rdocumentation.org/packages/mixOmics/versions/6.3.2/topics/liver.toxicity). Colon_cancer_classification.py contains the binary classification analysis based on a colon cancer expression dataset (Alon et al., 1999) extracted from the colonCA library of the Bioconductor R Package (https://bioconductor.org/packages/release/data/experiment/html/colonCA.html). Leukaemia_classification.py contains the binary classification analysis based on a gene expression study (Golub et al., 1999) extracted from OpenML (https://www.openml.org/search?type=data&sort=runs&id=1104&status=active). Finally, PANCAN_classification.py contains the multiclass classification analysis based on a TCGA Pan-Cancer RNA-Seq dataset (Weinstein et al., 2013) extracted from the UCI Machine Learning repository (https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+
rna+seq).

# Running an experiment
To setup an analysis: pip install -r requirements.txt
Each script runs standalone: 
python Simulation1.py
python Liver_toxicity_regression.py

Each prints the results tables with Monte Carlo means and standard errors for every reported metric. Regression tasks report variance and covariance explained, along with mean squared error, while classification studies report accuracy, AUC, precision, and variance explained--- the multiclass experiment additionally reports F1 score, and per-class precision, recall and F1 score.

# Data
The simulation scripts generate their data internally and require no downloads. The real-data expect the corresponding dataset files in the working directory (see provided links above).

# Reproducibility
All randomness is seeded within each script (data generation, splits, and cross-validation fold assignment), so results are deterministic given the pinned dependency versions in requirements.txt. The two largest analyses (Leukaemia, PANCAN) were run with process-level parallelism across replicates on a high-performance computing cluster employing 50 CPU cores.

# References
(1) Bushel, P. R., Wolfinger, R. D. and Gibson, G. (2007) Simultaneous clustering of gene expression data with clinical chemistry and pathological evaluations reveals phenotypic prototypes. BMC Systems Biology, 1, 15.
(2) Alon, U., Barkai, N., Notterman, D. A., Gish, K., Ybarra, S., Mack, D. and Levine, A. J. (1999) Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by oligonucleotide arrays. Proceedings of the National Academy of Sciences of the United States of America, 96, 6745–6750.
(3) Golub, T. R., Slonim, D. K., Tamayo, P., Huard, C., Gaasenbeek, M., Mesirov, J. P., Coller, H., Loh, M. L., Downing, J. R., Caligiuri, M. A., Bloomfield, C. D. and Lander, E. S. (1999) Molecular classification of cancer: class discovery and class prediction by gene expression monitoring. Science, 286, 531–537.
(4) Weinstein, J. N., Collisson, E. A., Mills, G. B., Shaw, K. R. M., Ozenberger, B. A., Ellrott, K., Shmulevich, I., Sander, C. and Stuart, J. M. (2013) The cancer genome atlas pan-cancer analysis project. Nat Genet, 45, 1113–1120.
