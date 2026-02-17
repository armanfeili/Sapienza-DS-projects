#' ---
#' title: "Conformal Prediction Methods for Functional Data"
#' subtitle: "Implementation of Lei, Rinaldo, Wasserman — Sections 3.1 and 4"
#' author: 
#'   - "Arman Feili (feili.2101835@studenti.uniroma1.it)"
#' date: "`r Sys.Date()`"
#' output:
#'   html_document:
#'     toc: true
#'     toc_depth: 3
#'     toc_float: true
#'     number_sections: true
#'     theme: flatly
#'     highlight: tango
#'     code_folding: hide
#'     fig_width: 10
#'     fig_height: 6
#' ---
#'
#' # Introduction
#' 
#' ## Background
#' 
#' In **Functional Data Analysis (FDA)**, each observation is not a single number but an entire 
#' function sampled at discrete time points. This report applies **conformal prediction** to 
#' functional data, constructing prediction bands that contain future curves with guaranteed 
#' coverage probability.
#' 
#' We implement the methodology from:
#' 
#' > Lei, Rinaldo, Wasserman (2024). *"A Conformal Prediction Approach to Explore Functional Data"*
#' 
#' ## Our Implementation
#' 
#' We implement both **Section 3.1** and **Section 4** from the paper:
#' 
#' **Section 3.1 — Gaussian Mixture Approximation (Algorithm 2):**
#' 
#' 1. **Algorithm 1 (Sec 2.2)**: Inductive conformal split-sample approach
#' 2. **Algorithm 2 (Sec 3)**: Functional conformal bands via Gaussian Mixture Model (GMM)
#' 3. **Eq.(6)**: Outer bound construction using union of ellipsoids
#' 4. **Closed-form ellipsoid projections**: Using the projection lemma from course notes
#'
#' **Section 4 — Pseudo-Density Methods:**
#'
#' 5. **Eq.(10)**: Pseudo-density estimator $\hat{p}_h(u)$ with Gaussian kernel
#' 6. **Eq.(11)**: Conformal set approximation $C^+_{n,\alpha}$ and sample approximation
#' 7. **Anomalies/Median/High-density subsets**: Based on pseudo-density ranking
#' 8. **Mean-shift prototypes**: Functional modes via Cheng (1995) algorithm
#' 9. **Conformal cluster tree**: Graph $G_{\alpha,\epsilon}$ and connected components
#' 
#' ## Dataset
#' 
#' We collected our own **tri-axial accelerometer data** using a smartphone (Samsung Galaxy A70) 
#' with the phyphox app. The dataset consists of 183 functional curves representing 10-second 
#' windows of acceleration magnitude during three activities: Standing, Walking, and Fast Walking.
#' 
#' For detailed documentation of our data collection procedure, see 
#' [Acceleration_Datasets.md](Acceleration_Datasets.md).
#' 
#' ## Report Structure
#' 
#' 1. **Dataset Creation** — Processing raw phyphox signals into FDA format
#' 2. **Exploratory Data Analysis** — Visualizing and understanding the functional data
#' 3. **Section 3: Conformal Prediction Bands** — GMM-based bands (Algorithm 2, Eq. 6)
#' 4. **Section 4: Pseudo-Density Methods** — Anomaly detection, prototypes, cluster trees (Eq. 10-11)
#' 5. **Discussion** — Coverage analysis, implementation checklist, and conclusions
#' 

#+ setup, include=FALSE
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE,
  fig.align = "center",
  out.width = "100%",
  class.source = "fold-show"
)

# Suppress package startup messages globally
options(warn = -1)

#'
#' # Setup and Configuration
#' 
#' ## Random Seed
#' 
#' We set a fixed seed for reproducibility of all random operations (data splits, sampling).

#+ global-seed
set.seed(2026)
options(warn = 0)  # Reset warnings after setup

#'
#' ## Dataset
#' 
#' We collected **4 long tri-axial accelerometer time series** using a Samsung Galaxy A70 
#' smartphone with the phyphox app at approximately 203 Hz.
#' 
#' **Activities recorded:**
#' 
#' | Activity | Description |
#' |----------|-------------|
#' | Standing | Stationary position, phone in pocket |
#' | Walking (×2) | Normal pace walking |
#' | Fast Walking | Brisk walking pace |
#' 
#' **Processing pipeline** (performed in `dataset_creation.R`):
#' 
#' 1. **Trimming**: Remove first/last 5 seconds (handling artifacts)
#' 2. **Windowing**: 10-second non-overlapping windows (reduces temporal dependence)
#' 3. **Resampling**: Linear interpolation to 200 grid points on $[0,1]$
#' 4. **Centering**: Subtract window mean to remove gravity/orientation offsets
#'
#' **To create the dataset, run `dataset_creation.R` first.**

#+ load-dataset-config
data_file <- "accel_fda_dataset.rds"

## --- Load the pre-processed dataset ---
if (!file.exists(data_file)) {
  stop(sprintf(
    "Dataset file '%s' not found.\n\nPlease run 'dataset_creation.R' first to create the dataset from raw phyphox recordings.",
    data_file
  ))
}

#'
#' ## Conformal Parameters
#' 
#' The key parameters controlling our conformal prediction procedure:

#+ configuration
## --- Conformal parameters ---
alpha       <- 0.10       # Miscoverage level (1-alpha = 90% coverage)
split_ratio <- 0.5        # Fraction for training (rest for calibration)

## --- Projection parameters ---
p           <- 5          # Number of basis functions (truncation dimension)
K           <- 3          # Number of GMM components (matches 3 activities)

## --- Numerical stability ---
ridge_eps   <- 1e-8       # Small ridge for near-singular covariances

## --- File paths ---
output_dir  <- "outputs"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

#' | Parameter | Value | Description |
#' |-----------|-------|-------------|
#' | $\alpha$ | `r alpha` | Miscoverage level (target coverage = `r 100*(1-alpha)`%) |
#' | split_ratio | `r split_ratio` | Training fraction (rest for calibration) |
#' | $p$ | `r p` | Basis functions (Fourier truncation dimension) |
#' | $K$ | `r K` | GMM components (one per activity type) |
#' | seed | 2026 | Random seed for reproducibility |
#'

#'
#' ## Required Libraries

#+ libraries, message=FALSE, warning=FALSE
required_pkgs <- c("knitr", "mclust", "mvtnorm")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Package '%s' required. Install with: install.packages('%s')", pkg, pkg))
  }
}
suppressPackageStartupMessages({
  library(knitr)    # For kable() tables
  library(mclust)   # Gaussian Mixture Model fitting
  library(mvtnorm)  # Multivariate normal density/sampling
})

#'
#' # Mathematical Building Blocks
#' 
#' This section defines the core mathematical functions needed for conformal prediction
#' on functional data. These functions transform curves into a form where we can apply
#' standard statistical methods (like GMM fitting) and then transform back to get
#' prediction bands.
#' 
#' ## Cosine Basis Functions
#' 
#' **Why we need a basis:** Each curve in our dataset has 200 points, which is 
#' high-dimensional. Instead of working directly with 200-dimensional vectors, we 
#' project curves onto a smaller set of "basis functions" — smooth mathematical 
#' curves that can be combined to approximate any reasonable function.
#' 
#' **Why cosine basis:** Cosine functions are smooth, periodic, and orthogonal to each
#' other. The first few basis functions capture the overall shape (low-frequency patterns),
#' while higher-order ones capture fine details (high-frequency oscillations). By keeping
#' only $p=5$ coefficients, we retain the essential shape while filtering out noise.
#' 
#' **Implementation note:** The paper's Algorithm 2 uses empirical eigenfunctions (FPCA basis)
#' estimated from the training data. For simplicity, we use a **fixed orthonormal cosine basis**
#' instead, which is a standard choice in FDA (see course notes).
#'
#' **Notation mapping and assumption:** The paper's Section 3.1 uses empirical eigenfunctions
#' $\{\phi_j\}$ estimated via FPCA on the training set. We replace these with a fixed cosine
#' basis. This changes the coefficient representation (our $\xi_i$ are cosine coefficients, not
#' FPCA scores), but preserves all mathematical steps that require only orthonormality:
#' projection $\xi_i = \langle X_i, \phi_j \rangle$, ellipsoid optimization via projection lemma,
#' and reconstruction. We do **not** claim equivalence to FPCA results — only that Algorithm 2
#' remains valid under any orthonormal basis.
#' 
#' We use an orthonormal cosine basis on $[0,1]$:
#' 
#' $$\phi_0(x) = 1, \quad \phi_j(x) = \sqrt{2}\cos(\pi j x) \text{ for } j > 0$$
#' 
#' This basis satisfies $\langle \phi_i, \phi_j \rangle = \delta_{ij}$ (Kronecker delta),
#' meaning each basis function is independent of the others — a crucial property for
#' clean statistical analysis.

#' The following code defines all helper functions. Click "Show code" to see the
#' implementation details. Each function serves a specific purpose:
#' 
#' - `cos_basis`: Evaluates a single basis function at given points
#' - `basis_vector`: Collects all $p$ basis functions into a vector for a given time $t$
#' - `project_curve`: Converts a 200-point curve into $p$ coefficients (dimensionality reduction)
#' - `gmm_density`: Evaluates the fitted GMM at a coefficient vector
#' - `ensure_spd`: Fixes numerical issues with covariance matrices
#' - `compute_ellipsoid_radius_sq`: Converts a density threshold into an ellipsoid size
#' - `ellipsoid_projection`: The key function that converts ellipsoids back to prediction intervals
#' - `merge_intervals`: Combines overlapping intervals from different GMM components
#' - `reconstruct_curve`: Converts coefficients back to a curve for visualization

#+ helper-functions, class.source = 'fold-hide'
## --- Cosine basis (orthonormal on [0,1]) ---

cos_basis <- function(x, j) {
  # Returns φ_j(x) for the cosine basis on [0,1]
  # Note: We use fixed cosine basis (course notes) instead of paper's FPCA basis
  if (j == 0) {
    return(rep(1, length(x)))
  } else {
    return(sqrt(2) * cos(pi * j * x))
  }
}

## --- 3.2: Evaluate basis vector φ(t) = (φ_0(t), ..., φ_{p-1}(t))^T ---
basis_vector <- function(t, p) {
  # Returns p-dimensional vector of basis evaluations at point t
  # Paper Sec 3.1: φ(t) = (φ_1(t),...,φ_p(t))^T (we use 0-indexed internally)
  sapply(0:(p - 1), function(j) cos_basis(t, j))
}

## --- 3.3: Project curve onto basis (numerical inner product) ---
## Paper Algorithm 2, Step 2: "Compute basis projection coefficients ξ_ij = <X_i, φ_j>"
## Approximated via trapezoidal rule

project_curve <- function(y, grid, p) {
  # y: curve values at grid points (length M)
  # grid: grid points in [0,1] (length M)
  # p: number of basis functions
  # Returns: p-dimensional coefficient vector ξ
  
  M_local <- length(grid)  # Use local M for generality
  dt <- grid[2] - grid[1]  # Uniform grid spacing
  
  xi <- numeric(p)
  for (j in 0:(p - 1)) {
    phi_j <- cos_basis(grid, j)
    # Trapezoidal rule: ∫ f ≈ dt * (f_1/2 + f_2 + ... + f_{M-1} + f_M/2)
    integrand <- y * phi_j
    xi[j + 1] <- dt * (integrand[1]/2 + sum(integrand[2:(M_local-1)]) + integrand[M_local]/2)
  }
  
  return(xi)
}

## --- 3.4: GMM density evaluation ---
## Paper Sec 3.1: "f̂(ξ) = Σ_k π_k φ(ξ; μ_k, Σ_k)"

gmm_density <- function(xi, pi_k, mu_list, sigma_list) {
  # Evaluate mixture density at point xi
  # Paper Sec 3.1: conformity score g(x) = f̂(ξ) where ξ = projection of x
  
  K <- length(pi_k)
  density <- 0
  
  for (k in 1:K) {
    density <- density + pi_k[k] * dmvnorm(xi, mean = mu_list[[k]], sigma = sigma_list[[k]])
  }
  
  return(density)
}

## --- 3.5: Ensure covariance matrix is SPD with ridge if needed ---
## Loop until Cholesky works with geometric ridge increase, or error after max tries

ensure_spd <- function(sigma, eps = 1e-8, max_tries = 10) {
  # Add ridge (geometrically increasing) until matrix is positive definite
  # Returns regularized matrix and flag indicating if ridge was added
  # Errors if PD cannot be achieved after max_tries
  
  p <- nrow(sigma)
  
  # First try: original matrix
  chol_result <- tryCatch(chol(sigma), error = function(e) NULL)
  if (!is.null(chol_result)) {
    return(list(sigma = sigma, regularized = FALSE, ridge_used = 0))
  }
  
  # Loop with geometrically increasing ridge
  current_ridge <- eps
  for (attempt in 1:max_tries) {
    sigma_reg <- sigma + diag(current_ridge, p)
    chol_result <- tryCatch(chol(sigma_reg), error = function(e) NULL)
    
    if (!is.null(chol_result)) {
      return(list(sigma = sigma_reg, regularized = TRUE, ridge_used = current_ridge))
    }
    
    # Geometric increase
    current_ridge <- current_ridge * 10
  }
  
  # Failed after max_tries
  stop(sprintf("ensure_spd: Could not make matrix PD after %d attempts (final ridge = %.2e). Matrix may be severely ill-conditioned.",
               max_tries, current_ridge))
}

## --- 3.6: Compute ellipsoid radius from density threshold ---
## Paper Eq.(6): T_{n,k} = {ξ : φ(ξ; μ_k, Σ_k) ≥ λ/(K π_k)}
## Gaussian density inequality → ellipsoid constraint
## r_k² = -2log(τ_k) - p·log(2π) - log|Σ|

compute_ellipsoid_radius_sq <- function(tau_k, sigma_k, p) {
  # Paper Eq.(6): τ_k = λ/(K π_k) is density threshold for component k
  # Returns: r_k² (squared radius), or NA if ellipsoid is empty/invalid
  
  # Handle invalid tau_k
  if (!is.finite(tau_k) || tau_k <= 0) {
    return(NA)
  }
  
  log_det_sigma <- determinant(sigma_k, logarithm = TRUE)$modulus
  
  # Paper derivation: φ(ξ;μ,Σ) ≥ τ ⟺ (ξ-μ)ᵀΣ⁻¹(ξ-μ) ≤ r²
  # where r² = -2log(τ) - p·log(2π) - log|Σ|
  r_sq <- -2 * log(tau_k) - p * log(2 * pi) - as.numeric(log_det_sigma)
  
  if (!is.finite(r_sq) || r_sq < 0) {
    return(NA)  # Empty ellipsoid (threshold too high)
  }
  
  return(r_sq)
}

## --- 3.7: Ellipsoid projection (closed-form) ---
## FoSL notes: For ellipsoid E = {ξ : (ξ-c)ᵀ Q⁻¹ (ξ-c) ≤ 1}
##   max_{ξ ∈ E} ξᵀφ = cᵀφ + √(φᵀQφ)
##   min_{ξ ∈ E} ξᵀφ = cᵀφ - √(φᵀQφ)
##
## Our ellipsoid from Paper Eq.(6): (ξ-μ)ᵀ Σ⁻¹ (ξ-μ) ≤ r²
## Rewrite as: (ξ-μ)ᵀ (r²Σ)⁻¹ (ξ-μ) ≤ 1, so Q = r²Σ
## Therefore: max/min = μᵀφ ± r·√(φᵀΣφ)

ellipsoid_projection <- function(mu, sigma, r, phi_t) {
  # FoSL projection lemma: closed-form linear optimization over ellipsoid
  # mu: center of ellipsoid (p-vector)
  # sigma: covariance matrix (p x p)  
  # r: radius (scalar, sqrt of r_sq)
  # phi_t: basis vector at time t (p-vector)
  # Returns: list(lower, upper) for the interval [ℓ_k(t), u_k(t)]
  
  center_term <- sum(mu * phi_t)  # μᵀφ(t)
  quad_form <- as.numeric(t(phi_t) %*% sigma %*% phi_t)  # φ(t)ᵀΣφ(t)
  
  if (quad_form < 0) {
    # Numerical issue, shouldn't happen with valid SPD covariance
    quad_form <- 0
  }
  
  half_width <- r * sqrt(quad_form)  # r·√(φᵀΣφ)
  
  # Paper Sec 3.1: u_k(t) = sup, ℓ_k(t) = inf
  return(list(
    lower = center_term - half_width,
    upper = center_term + half_width
  ))
}

## --- 3.8: Merge overlapping intervals ---
## Paper Sec 3.1: "B_n(t) = ∪_k [ℓ_k(t), u_k(t)]" may have disconnected slices

merge_intervals <- function(intervals) {
  # intervals: list of (lower, upper) pairs
  # Returns: list of merged non-overlapping intervals
  
  if (length(intervals) == 0) return(list())
  
  # Convert to matrix for easier handling
  int_mat <- do.call(rbind, lapply(intervals, function(x) c(x$lower, x$upper)))
  
  # Remove NA rows
  valid <- complete.cases(int_mat)
  if (sum(valid) == 0) return(list())
  int_mat <- int_mat[valid, , drop = FALSE]
  
  # Sort by lower bound
  int_mat <- int_mat[order(int_mat[, 1]), , drop = FALSE]
  
  # Merge overlapping
  merged <- list()
  current <- int_mat[1, ]
  
  if (nrow(int_mat) > 1) {
    for (i in 2:nrow(int_mat)) {
      if (int_mat[i, 1] <= current[2]) {
        # Overlapping, extend
        current[2] <- max(current[2], int_mat[i, 2])
      } else {
        # Non-overlapping, save current and start new
        merged[[length(merged) + 1]] <- list(lower = current[1], upper = current[2])
        current <- int_mat[i, ]
      }
    }
  }
  
  # Save last interval
  merged[[length(merged) + 1]] <- list(lower = current[1], upper = current[2])
  
  return(merged)
}

## --- 3.9: Reconstruct curve from coefficients ---
## For visualization: X(t) ≈ Σ_j ξ_j φ_j(t)

reconstruct_curve <- function(xi, grid) {
  # xi: coefficient vector (length p)
  # grid: evaluation grid
  # Returns: reconstructed curve values
  
  p <- length(xi)
  y <- rep(0, length(grid))
  
  for (j in 0:(p - 1)) {
    y <- y + xi[j + 1] * cos_basis(grid, j)
  }
  
  return(y)
}

#'
#' # Exploratory Data Analysis
#' 
#' ## Dataset Overview
#' 
#' We load and explore the pre-processed functional accelerometer dataset.

#+ load-dataset
if (!file.exists(data_file)) {
  stop(sprintf("Dataset file not found: %s", data_file))
}

dataset <- readRDS(data_file)

## Extract components
X_raw      <- dataset$X_raw       # n x M matrix of raw curves
X_centered <- dataset$X_centered  # n x M matrix of centered curves
grid01     <- dataset$grid01      # Grid in [0,1]
meta       <- dataset$meta        # Metadata with activity labels
M          <- dataset$M           # Number of grid points
n          <- nrow(X_raw)

## Validate data integrity
stopifnot(nrow(X_raw) == nrow(meta), length(grid01) == M, ncol(X_raw) == M)

## Use centered curves for analysis (removes gravity/orientation offset)
X <- X_centered

#' **Dataset Summary:**
#' 
#' | Property | Value |
#' |----------|-------|
#' | Total curves ($n$) | `r n` |
#' | Grid points ($M$) | `r M` |
#' | Activities | `r paste(unique(meta$activity), collapse = ", ")` |
#' 
#' **Distribution by Activity:**

#+ activity-table
knitr::kable(as.data.frame(table(Activity = meta$activity)), 
             col.names = c("Activity", "Count"))

#'
#' ## Functional Data Visualization
#' 
#' In functional data analysis, each curve is a single observation. The plots below 
#' show the characteristic patterns for each activity type.

#+ eda-setup
## Define activity colors (using activity_code values as keys)
cols <- c("Stand" = "#E41A1C", "Walk" = "#377EB8", "Fast_Walk" = "#4DAF4A", "Unknown" = "gray70")
## Map activity codes to pretty labels for legends and titles
activity_labels <- c("Stand" = "Standing", "Walk" = "Walking", "Fast_Walk" = "Fast Walking")

#'
#' ### Sample Curves
#' 
#' A random sample of 50 curves colored by activity. The **left panel** shows raw 
#' acceleration magnitude (includes gravity offset ~9.8 m/s²), while the **right panel** 
#' shows centered curves (mean subtracted per window).
#' 
#' **What to look for:**
#' 
#' - Red curves (Standing) cluster tightly around 9.8 m/s² in raw data — this is gravity
#' - Blue curves (Walking) show oscillations around 10 m/s² from footstep impacts
#' - Green curves (Fast Walking) have the largest amplitude swings
#' - After centering (right panel), all curves oscillate around zero, making activities comparable

#+ eda-plot-sample-curves, fig.width=12, fig.height=6
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

sample_idx <- sample(1:n, min(50, n))

# Raw curves
plot(NULL, xlim = c(0, 1), ylim = range(X_raw[sample_idx, ]),
     xlab = "Normalized time", ylab = "Absolute acceleration (m/s²)",
     main = "Raw Curves (n=50)")
for (i in sample_idx) {
  lines(grid01, X_raw[i, ], col = adjustcolor(cols[meta$activity_code[i]], alpha.f = 0.4), lwd = 0.8)
}
legend("topright", c("Standing", "Walking", "Fast Walking"), col = cols[1:3], lwd = 2, cex = 0.8, bty = "n")

# Centered curves
plot(NULL, xlim = c(0, 1), ylim = range(X[sample_idx, ]),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = "Centered Curves (n=50)")
for (i in sample_idx) {
  lines(grid01, X[i, ], col = adjustcolor(cols[meta$activity_code[i]], alpha.f = 0.4), lwd = 0.8)
}
legend("topright", c("Standing", "Walking", "Fast Walking"), col = cols[1:3], lwd = 2, cex = 0.8, bty = "n")

#+ eda-save-plot1, include=FALSE
png(file.path(output_dir, "01_sample_curves.png"), width = 1000, height = 600)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(NULL, xlim = c(0, 1), ylim = range(X_raw[sample_idx, ]),
     xlab = "Normalized time", ylab = "Absolute acceleration (m/s²)",
     main = "Raw Curves (n=50)")
for (i in sample_idx) {
  lines(grid01, X_raw[i, ], col = adjustcolor(cols[meta$activity_code[i]], alpha.f = 0.4), lwd = 0.8)
}
legend("topright", c("Standing", "Walking", "Fast Walking"), col = cols[1:3], lwd = 2, cex = 0.8, bty = "n")
plot(NULL, xlim = c(0, 1), ylim = range(X[sample_idx, ]),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = "Centered Curves (n=50)")
for (i in sample_idx) {
  lines(grid01, X[i, ], col = adjustcolor(cols[meta$activity_code[i]], alpha.f = 0.4), lwd = 0.8)
}
legend("topright", c("Standing", "Walking", "Fast Walking"), col = cols[1:3], lwd = 2, cex = 0.8, bty = "n")
dev.off()

#' **Observation:** The raw curves show that standing produces nearly constant readings 
#' around gravity (9.8 m/s²), while walking activities create periodic spikes from each 
#' footstep. The centered version removes the gravity baseline, revealing the true 
#' dynamic signal: standing curves are nearly flat, while walking curves show clear 
#' oscillatory patterns.

#'
#' ### Mean Curves by Activity
#' 
#' The **mean curve** (solid line) with **±2 SD bands** (shaded) summarize each activity.
#' The sample sizes shown in each panel title are computed from the data. Generally:
#' 
#' - **Standing**: Minimal movement — the band is very narrow (low SD)
#' - **Walking**: Moderate variability from regular footsteps
#' - **Fast Walking**: Highest variability from more forceful steps

#+ eda-plot-mean-by-activity, fig.width=12, fig.height=5
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))

## Use activity_code for lookups (matches "Stand", "Walk", "Fast_Walk")
activities <- c("Stand", "Walk", "Fast_Walk")
for (act in activities) {
  idx <- which(meta$activity_code == act)
  if (length(idx) == 0) next
  X_act <- X[idx, , drop = FALSE]
  
  mean_curve <- colMeans(X_act)
  sd_curve <- apply(X_act, 2, sd)
  ylim_range <- range(c(mean_curve - 2*sd_curve, mean_curve + 2*sd_curve))
  
  plot(grid01, mean_curve, type = "l", lwd = 2, col = cols[act],
       xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
       main = sprintf("%s (n=%d)", activity_labels[act], length(idx)), ylim = ylim_range)
  
  polygon(c(grid01, rev(grid01)),
          c(mean_curve - 2*sd_curve, rev(mean_curve + 2*sd_curve)),
          col = adjustcolor(cols[act], alpha.f = 0.2), border = NA)
  lines(grid01, mean_curve, lwd = 2, col = cols[act])
}

#+ eda-save-plot2, include=FALSE
png(file.path(output_dir, "02_mean_curves_by_activity.png"), width = 1000, height = 500)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
for (act in activities) {
  idx <- which(meta$activity_code == act)
  if (length(idx) == 0) next
  X_act <- X[idx, , drop = FALSE]
  mean_curve <- colMeans(X_act)
  sd_curve <- apply(X_act, 2, sd)
  ylim_range <- range(c(mean_curve - 2*sd_curve, mean_curve + 2*sd_curve))
  plot(grid01, mean_curve, type = "l", lwd = 2, col = cols[act],
       xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
       main = sprintf("%s (n=%d)", activity_labels[act], length(idx)), ylim = ylim_range)
  polygon(c(grid01, rev(grid01)),
          c(mean_curve - 2*sd_curve, rev(mean_curve + 2*sd_curve)),
          col = adjustcolor(cols[act], alpha.f = 0.2), border = NA)
  lines(grid01, mean_curve, lwd = 2, col = cols[act])
}
dev.off()

#' **Observation:** The ±2 SD bands visually separate the three activities. Standing has 
#' a tight band close to zero, indicating little movement. Walking and Fast Walking both 
#' show wider bands, but Fast Walking's band extends further, reflecting more energetic 
#' motion. The mean curves themselves are all near zero because we centered each window.

#'
#' ### Overall Variability
#' 
#' When we pool all activities together, the overall mean stays near zero (by construction), 
#' but the standard deviation band captures the full range of human motion variability.

#+ eda-plot-overall, fig.width=10, fig.height=6
par(mar = c(4, 4, 3, 1))

mean_overall <- colMeans(X)
sd_overall <- apply(X, 2, sd)

plot(grid01, mean_overall, type = "l", lwd = 2, col = "black",
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = "Overall Mean Curve ± 2 SD",
     ylim = range(c(mean_overall - 2*sd_overall, mean_overall + 2*sd_overall)))

polygon(c(grid01, rev(grid01)),
        c(mean_overall - 2*sd_overall, rev(mean_overall + 2*sd_overall)),
        col = adjustcolor("gray", alpha.f = 0.3), border = NA)
lines(grid01, mean_overall, lwd = 2, col = "black")

#+ eda-save-plot3, include=FALSE
png(file.path(output_dir, "03_overall_mean_variability.png"), width = 800, height = 500)
par(mar = c(4, 4, 3, 1))
plot(grid01, mean_overall, type = "l", lwd = 2, col = "black",
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = "Overall Mean Curve ± 2 SD",
     ylim = range(c(mean_overall - 2*sd_overall, mean_overall + 2*sd_overall)))
polygon(c(grid01, rev(grid01)),
        c(mean_overall - 2*sd_overall, rev(mean_overall + 2*sd_overall)),
        col = adjustcolor("gray", alpha.f = 0.3), border = NA)
lines(grid01, mean_overall, lwd = 2, col = "black")
dev.off()

#' **Observation:** The wide gray band (approximately ±4 m/s²) reflects the mixture of 
#' all three activities. This heterogeneity is precisely why we use a Gaussian Mixture 
#' Model rather than a single Gaussian — a single distribution would poorly fit this 
#' multi-modal data.

#'
#' ## Summary of Exploratory Findings
#' 
#' Our accelerometer dataset exhibits clear structure that makes it well-suited for 
#' functional conformal prediction:
#' 
#' 1. **Three distinct activity patterns**: Standing, walking, and fast walking each 
#'    produce characteristic acceleration signatures
#' 2. **Standing is nearly constant**: Variability is very low (~0.3 m/s² SD), reflecting 
#'    only sensor noise and subtle body sway
#' 3. **Walking creates periodic signals**: Footstep impacts create regular oscillations 
#'    at ~2 Hz (typical walking cadence)
#' 4. **Fast walking amplifies patterns**: Same periodic structure as walking, but with 
#'    larger amplitude due to more forceful ground contact
#' 5. **Centering removes baseline**: After subtracting each window's mean, all activities 
#'    become comparable and centered at zero
#' 
#' ## Author Observations on the Accelerometer Dataset
#' 
#' ### Qualitative Differences Between Activities
#' 
#' Examining our self-collected accelerometer data reveals striking qualitative differences
#' between the three activity types:
#' 
#' - **Standing curves** exhibit an almost flat profile with minimal fluctuations. The 
#'   centered curves hover near zero with amplitude typically below ±0.5 m/s². This reflects
#'   the natural body sway and sensor noise when a person stands still. The curves are 
#'   remarkably homogeneous across all Standing windows.
#'
#' - **Walking curves** display clear periodic oscillations corresponding to the gait cycle.
#'   Each footstep produces a characteristic impact signature: a sharp acceleration spike
#'   followed by a brief deceleration phase. The period of these oscillations (~0.5-1 second
#'   per step) corresponds to a typical walking cadence of 1-2 steps per second. The 
#'   amplitude of these oscillations ranges from ±2 to ±4 m/s².
#'
#' - **Fast Walking curves** share the periodic structure of normal walking but with 
#'   amplified features. The impact spikes are larger (±4 to ±8 m/s²), and the frequency
#'   is slightly higher (~2-3 steps per second). Additionally, there is more variability
#'   between curves, likely reflecting the more dynamic and less controlled nature of
#'   brisk walking.
#'
#' ### Variability Patterns Over Time
#'
#' An interesting observation is how variability changes across the normalized time axis:
#'
#' - For **Standing**, the variability is uniformly low throughout the 10-second window,
#'   as expected for a stationary activity.
#'
#' - For **Walking** and **Fast Walking**, we observe that the variability appears relatively
#'   uniform across the window, but with occasional peaks that align with the stochastic
#'   nature of individual footstep timings. Since our 10-second windows are not synchronized
#'   to specific gait phases, the periodic patterns "average out" when looking at the 
#'   ensemble of curves, resulting in a relatively constant SD band across time.
#'
#' - The overall SD band (±4 m/s² when pooling all activities) is much wider than any 
#'   single activity's band, highlighting the heterogeneous mixture structure.
#'
#' ### Evidence of Multimodality
#'
#' The most salient feature of this dataset is its clear **multimodal structure**:
#'
#' - The coefficient space visualization (shown later) reveals three distinct clusters
#'   corresponding to the three activities. This separation is not perfectly clean—there
#'   is some overlap between Walking and Fast Walking curves—but the Standing cluster
#'   is well-separated from the locomotion activities.
#'
#' - Within the walking activities, we occasionally observe curves that appear "transitional,"
#'   perhaps captured during moments of acceleration/deceleration or stride adjustments.
#'   These curves may appear as outliers or fall between clusters.
#'
#' - The stride phases within individual curves create local multimodality: each curve
#'   contains multiple peaks (footsteps), but since curves are not aligned to stride phase,
#'   this within-curve periodicity manifests as increased point-wise variance rather than
#'   a systematic pattern in the ensemble mean.
#'
#' This multimodal structure motivates our use of a Gaussian Mixture Model (Section 3.1)
#' rather than a single Gaussian, and also makes this dataset suitable for the pseudo-density
#' clustering approach (Section 4).

#'
#' # Conformal Prediction Procedure
#' 
#' We now implement **Algorithm 2 (Section 3.1)** from the paper to construct 
#' conformal prediction bands for functional data.
#' 
#' ## Algorithm Overview
#' 
#' **The big picture:** We want to build a "prediction band" — a region that will contain
#' future acceleration curves with high probability. The key innovation of conformal 
#' prediction is that this guarantee holds without assuming any specific distribution 
#' for the data (only exchangeability).
#' 
#' The procedure consists of the following steps:
#' 
#' 1. **Split data** into training set $\mathcal{D}_1$ and calibration set $\mathcal{D}_2$.
#'    The split prevents overfitting: training fits the model, calibration sets the threshold.
#' 2. **Project curves** onto $p$-dimensional orthonormal basis: $\xi_i = \langle X_i, \phi \rangle$.
#'    This reduces 200-dimensional curves to 5-dimensional coefficient vectors.
#' 3. **Fit GMM** with $K$ components on training projections.
#'    The GMM captures the multi-modal structure (different activities = different modes).
#' 4. **Compute threshold** $\lambda$ using calibration density scores.
#'    $\lambda$ is chosen so that $(1-\alpha)$ of calibration curves have density ≥ $\lambda$.
#' 5. **Build ellipsoids** $T_{n,k}$ from density level sets (Paper Eq. 6).
#'    Each GMM component contributes an ellipsoid in coefficient space.
#' 6. **Construct band** $B_n(t) = \bigcup_k [\ell_k(t), u_k(t)]$ via projection lemma.
#'    The ellipsoids are projected back to function space, giving intervals at each time $t$.
#'

#+ conformal-header

#'
#' ## Step 1: Data Splitting
#' 
#' Following **Algorithm 1, Step 1**, we randomly split the data into:
#' 
#' - **Training set** $\mathcal{D}_1$: Used to fit the GMM density estimator
#' - **Calibration set** $\mathcal{D}_2$: Used to compute the conformal threshold $\lambda$

#+ conformal-split
n1 <- floor(n * split_ratio)
n2 <- n - n1

train_idx <- sample(1:n, n1)
calib_idx <- setdiff(1:n, train_idx)

X_train <- X[train_idx, , drop = FALSE]
X_calib <- X[calib_idx, , drop = FALSE]

#' | Set | Size | Purpose |
#' |-----|------|---------|
#' | Training ($n_1$) | `r n1` | Fit GMM density |
#' | Calibration ($n_2$) | `r n2` | Compute threshold $\lambda$ |

#'
#' ## Step 2: Basis Projection
#' 
#' Following **Algorithm 2, Step 2**, we project each curve onto the $p$-dimensional 
#' cosine basis to obtain coefficient vectors $\xi_i = (\langle X_i, \phi_0 \rangle, \ldots, \langle X_i, \phi_{p-1} \rangle)^T$.
#' 
#' This reduces each infinite-dimensional curve to a finite-dimensional representation.

#+ conformal-project
Xi_train <- t(apply(X_train, 1, function(y) project_curve(y, grid01, p)))
Xi_calib <- t(apply(X_calib, 1, function(y) project_curve(y, grid01, p)))

#'
#' ## Step 3: Fit Gaussian Mixture Model
#' 
#' Following **Paper Section 3.1**, we fit a $K$-component Gaussian mixture model (GMM) 
#' on the training projections:
#' 
#' $$\hat{f}(\xi) = \sum_{k=1}^K \pi_k \phi(\xi; \mu_k, \Sigma_k)$$
#' 
#' where $\phi(\cdot; \mu, \Sigma)$ is the multivariate normal density.

#+ conformal-gmm
gmm_fit <- Mclust(Xi_train, G = K, modelNames = "VVV", verbose = FALSE)

if (is.null(gmm_fit)) {
  gmm_fit <- Mclust(Xi_train, G = K, verbose = FALSE)
}
stopifnot(!is.null(gmm_fit))

# Extract GMM parameters
pi_k <- gmm_fit$parameters$pro
mu_list <- lapply(1:K, function(k) gmm_fit$parameters$mean[, k])
sigma_list_raw <- lapply(1:K, function(k) gmm_fit$parameters$variance$sigma[, , k])

# Ensure covariance matrices are positive definite
sigma_list <- lapply(sigma_list_raw, function(sig) ensure_spd(sig, ridge_eps)$sigma)

#' **GMM Parameters:**
#' 
#' | Component | Mixing Weight $\pi_k$ |
#' |-----------|----------------------|
#' | 1 | `r round(pi_k[1], 3)` |
#' | 2 | `r round(pi_k[2], 3)` |
#' | 3 | `r round(pi_k[3], 3)` |

#'
#' ## Step 4: Compute Conformal Threshold
#' 
#' Following **Algorithm 1, Steps 3-4**, we compute conformity scores on the calibration 
#' set and determine the threshold $\lambda$:
#' 
#' 1. Compute density scores $\sigma_i = \hat{f}(\xi_i)$ for each calibration curve
#' 2. Sort scores: $\sigma_{(1)} \leq \cdots \leq \sigma_{(n_2)}$
#' 3. Set $\lambda = \sigma_{(\lceil (n_2+1)\alpha \rceil - 1)}$
#' 
#' The prediction region is $C_n = \{\xi : \hat{f}(\xi) \geq \lambda\}$.

#+ conformal-threshold
# Compute density scores for calibration set
f_calib <- apply(Xi_calib, 1, function(xi) gmm_density(xi, pi_k, mu_list, sigma_list))

# Sort and compute threshold
# Paper Algorithm 1, Step 4: λ = σ_{(⌈(n₂+1)α⌉ - 1)} (exact formula from paper)
# For density-based scores where larger = more conforming, this gives coverage ≥ (1-α)
f_sorted <- sort(f_calib)
lambda_idx_raw <- ceiling((n2 + 1) * alpha) - 1
# Clamp to valid range [1, n2] for edge cases (very small or large α)
lambda_idx <- max(1, min(n2, lambda_idx_raw))
lambda_idx_clamped <- (lambda_idx != lambda_idx_raw)
lambda <- f_sorted[lambda_idx]

# Expected inclusion rate on calibration set
# Points included: those at ranks lambda_idx, lambda_idx+1, ..., n2
expected_in_region <- n2 - lambda_idx + 1
expected_rate <- expected_in_region / n2

#'
#' ### Threshold Index Verification
#'
#' **Why the "−1" in the formula?** The paper's Algorithm 1 uses $\lambda = \sigma_{(\lceil (n_2+1)\alpha \rceil - 1)}$.
#' This indexing convention ensures that exactly $\lceil (n_2+1)\alpha \rceil - 1$ calibration points have
#' scores below the threshold, giving coverage $\geq (1-\alpha)$.
#'
#' We verify this choice by comparing alternative indexing conventions:

#+ threshold-index-verification
# Compare different indexing conventions
idx_alt1 <- ceiling((n2 + 1) * alpha) - 1  # Paper's formula (our choice)
idx_alt2 <- ceiling(n2 * alpha)            # Alternative: ceiling(n*alpha)
idx_alt3 <- floor(n2 * alpha)              # Alternative: floor(n*alpha)

lambda_alt1 <- f_sorted[max(1, min(n2, idx_alt1))]
lambda_alt2 <- f_sorted[max(1, min(n2, idx_alt2))]
lambda_alt3 <- f_sorted[max(1, min(n2, idx_alt3))]

inclusion_alt1 <- mean(f_calib >= lambda_alt1)
inclusion_alt2 <- mean(f_calib >= lambda_alt2)
inclusion_alt3 <- mean(f_calib >= lambda_alt3)

threshold_comparison <- data.frame(
  Formula = c("⌈(n₂+1)α⌉ - 1 (paper)", "⌈n₂α⌉", "⌊n₂α⌋"),
  Index = c(idx_alt1, idx_alt2, idx_alt3),
  Inclusion_Rate = sprintf("%.3f", c(inclusion_alt1, inclusion_alt2, inclusion_alt3)),
  Target = sprintf("%.3f", rep(1 - alpha, 3))
)
knitr::kable(threshold_comparison, caption = "Comparison of threshold indexing conventions")

#' **Result:** We follow the paper's indexing formula. The alternative conventions differ by at most
#' one rank position and yield similar (but not identical) inclusion rates. The paper's formula
#' ensures the finite-sample coverage guarantee holds exactly.

#'
#' ## Step 5: Construct Ellipsoids
#' 
#' Using **Paper Equation (6)**, we construct the outer bound as a union of ellipsoids:
#' 
#' $$T_n \subseteq \bigcup_{k=1}^K T_{n,k} = \bigcup_{k=1}^K \left\{\xi : \phi(\xi; \mu_k, \Sigma_k) \geq \frac{\lambda}{K\pi_k}\right\}$$
#' 
#' Each component contributes an ellipsoid with radius derived from the density threshold 
#' $\tau_k = \lambda/(K\pi_k)$.

#+ conformal-ellipsoids
ellipsoids <- list()

for (k in 1:K) {
  tau_k <- lambda / (K * pi_k[k])
  r_sq <- compute_ellipsoid_radius_sq(tau_k, sigma_list[[k]], p)
  
  ellipsoids[[k]] <- list(
    mu = mu_list[[k]],
    sigma = sigma_list[[k]],
    r = if (!is.na(r_sq) && r_sq > 0) sqrt(r_sq) else NA,
    r_sq = r_sq,
    tau_k = tau_k,
    valid = !is.na(r_sq) && r_sq > 0
  )
}

n_valid <- sum(sapply(ellipsoids, function(e) e$valid))

#'
#' ## Step 6: Build Prediction Band
#' 
#' Following **Paper Section 3.1** and the **projection lemma** from course notes, 
#' we compute the band slices:
#' 
#' $$B_n(t) = \bigcup_{k=1}^K [\ell_k(t), u_k(t)]$$
#' 
#' where the bounds are computed in closed form:
#' 
#' $$u_k(t) = \mu_k^T\phi(t) + r_k\sqrt{\phi(t)^T\Sigma_k\phi(t)}, \quad 
#'   \ell_k(t) = \mu_k^T\phi(t) - r_k\sqrt{\phi(t)^T\Sigma_k\phi(t)}$$

#+ conformal-band
band_lower <- numeric(M)
band_upper <- numeric(M)
band_intervals <- vector("list", M)

for (m in 1:M) {
  phi_t <- basis_vector(grid01[m], p)
  intervals_at_t <- list()
  
  for (k in 1:K) {
    if (!ellipsoids[[k]]$valid) next
    proj <- ellipsoid_projection(ellipsoids[[k]]$mu, ellipsoids[[k]]$sigma, 
                                  ellipsoids[[k]]$r, phi_t)
    intervals_at_t[[length(intervals_at_t) + 1]] <- proj
  }
  
  merged <- merge_intervals(intervals_at_t)
  band_intervals[[m]] <- merged
  
  if (length(merged) > 0) {
    band_lower[m] <- min(sapply(merged, function(x) x$lower))
    band_upper[m] <- max(sapply(merged, function(x) x$upper))
  } else {
    band_lower[m] <- NA
    band_upper[m] <- NA
  }
}

#'
#' # Validation
#' 
#' Before trusting our results, we verify that the implementation is correct. This 
#' section performs sanity checks on the mathematical properties that should hold
#' if our code is working correctly.
#' 
#' ## Basis Orthonormality
#' 
#' **What we're checking:** The cosine basis functions should be orthonormal, meaning:
#' 
#' - Each function has "length" 1 when integrated over $[0,1]$ (normality)
#' - Different functions are perpendicular — their inner product is 0 (orthogonality)
#' 
#' **Why it matters:** If the basis isn't orthonormal, our projections would be distorted,
#' and the GMM would fit a wrong shape. The inner product matrix should look like the 
#' identity matrix (1s on diagonal, 0s elsewhere).
#' 
#' Verify that the cosine basis satisfies $\langle \phi_i, \phi_j \rangle \approx \delta_{ij}$.

#+ validation-basis, class.source = 'fold-hide'
inner_product_check <- function(j1, j2, grid) {
  M_local <- length(grid)
  phi_j1 <- cos_basis(grid, j1)
  phi_j2 <- cos_basis(grid, j2)
  dt <- grid[2] - grid[1]
  dt * (phi_j1[1]*phi_j2[1]/2 + sum(phi_j1[2:(M_local-1)]*phi_j2[2:(M_local-1)]) + phi_j1[M_local]*phi_j2[M_local]/2)
}

n_check <- min(p, 5)
ortho_check <- matrix(NA, n_check, n_check)
for (j1 in 0:(n_check - 1)) {
  for (j2 in 0:(n_check - 1)) {
    ortho_check[j1 + 1, j2 + 1] <- inner_product_check(j1, j2, grid01)
  }
}

diag_vals <- diag(ortho_check)
offdiag_vals <- ortho_check[row(ortho_check) != col(ortho_check)]
basis_ok <- max(abs(diag_vals - 1)) < 0.01 && max(abs(offdiag_vals)) < 0.01

#' Inner product matrix (should be identity):

#+ validation-basis-show
knitr::kable(round(ortho_check, 4), col.names = paste0("j=", 0:(n_check-1)))

#' **Result:** `r if(basis_ok) "Orthonormality verified (max error < 0.01)" else "Warning: orthonormality may be violated"`
#' 
#' ## Calibration Inclusion Rate
#' 
#' **What we're checking:** After choosing the threshold $\lambda$ from calibration data,
#' approximately $(1-\alpha)$ of the calibration curves should have density above the threshold.
#' 
#' **Why it matters:** This is a direct consequence of how we chose $\lambda$ — we picked the 
#' $\alpha$-quantile of calibration scores, so by definition about $(1-\alpha)$ should pass.
#' If this check fails badly, something is wrong with the threshold computation.
#' 
#' By construction, the threshold $\lambda$ should yield approximately $(1-\alpha)$ 
#' inclusion rate on the calibration set.

#+ validation-inclusion
in_region <- (f_calib >= lambda)
inclusion_rate <- mean(in_region)

#' | Metric | Value |
#' |--------|-------|
#' | Curves with $\hat{f}(\xi) \geq \lambda$ | `r sum(in_region)` / `r n2` |
#' | Inclusion rate | `r sprintf("%.1f%%", 100 * inclusion_rate)` |
#' | Target $(1-\alpha)$ | `r sprintf("%.1f%%", 100 * (1 - alpha))` |
#' 
#' ## Out-of-Sample Coverage
#' 
#' **What we're checking:** The calibration inclusion rate above is "in-sample" — we used 
#' those curves to compute $\lambda$. The real test is whether **new, unseen curves** 
#' (that weren't used for training or threshold selection) fall within the prediction region.
#' 
#' **How we test it:** We repeat the entire procedure 20 times with random 3-way splits
#' (40% training, 30% calibration, 30% test). For each split, we train the GMM, compute
#' the threshold, and check how many test curves are included.
#' 
#' **Why it matters:** Conformal prediction guarantees that coverage should be at least
#' $(1-\alpha) = 90\%$ in expectation. If our out-of-sample coverage is much lower, it
#' suggests a violation of the exchangeability assumption or a bug in the implementation.
#'
#' **Important:** Coverage is evaluated in **coefficient space** via membership in 
#' $\{\xi : \hat{f}(\xi) \geq \lambda\}$. The plotted envelope band is a visualization 
#' superset — a curve may fall within the envelope but still be excluded if its coefficient
#' vector has density below $\lambda$.
#' 
#' To properly evaluate coverage, we perform repeated three-way splits (train/calibration/test) 
#' and measure the fraction of **held-out test curves** that fall within the prediction region.

#+ validation-coverage, class.source = 'fold-hide'
n_repeats <- 20
coverage_estimates <- numeric(n_repeats)

for (rep_i in 1:n_repeats) {
  perm <- sample(1:n)
  n_train_rep <- floor(n * 0.4)
  n_calib_rep <- floor(n * 0.3)
  
  train_idx_rep <- perm[1:n_train_rep]
  calib_idx_rep <- perm[(n_train_rep + 1):(n_train_rep + n_calib_rep)]
  test_idx_rep  <- perm[(n_train_rep + n_calib_rep + 1):n]
  
  Xi_train_rep <- t(apply(X[train_idx_rep, , drop = FALSE], 1, function(y) project_curve(y, grid01, p)))
  Xi_calib_rep <- t(apply(X[calib_idx_rep, , drop = FALSE], 1, function(y) project_curve(y, grid01, p)))
  Xi_test_rep  <- t(apply(X[test_idx_rep, , drop = FALSE], 1, function(y) project_curve(y, grid01, p)))
  
  gmm_rep <- tryCatch(Mclust(Xi_train_rep, G = K, modelNames = "VVV", verbose = FALSE), error = function(e) NULL)
  if (is.null(gmm_rep)) { coverage_estimates[rep_i] <- NA; next }
  
  pi_k_rep <- gmm_rep$parameters$pro
  mu_list_rep <- lapply(1:K, function(k) gmm_rep$parameters$mean[, k])
  sigma_list_rep <- lapply(1:K, function(k) ensure_spd(gmm_rep$parameters$variance$sigma[, , k], ridge_eps)$sigma)
  
  f_calib_rep <- sapply(1:nrow(Xi_calib_rep), function(i) gmm_density(Xi_calib_rep[i, ], pi_k_rep, mu_list_rep, sigma_list_rep))
  f_sorted_rep <- sort(f_calib_rep)
  n2_rep <- length(f_calib_rep)
  lambda_rep <- f_sorted_rep[max(1, min(n2_rep, ceiling((n2_rep + 1) * alpha) - 1))]
  
  f_test_rep <- sapply(1:nrow(Xi_test_rep), function(i) gmm_density(Xi_test_rep[i, ], pi_k_rep, mu_list_rep, sigma_list_rep))
  coverage_estimates[rep_i] <- mean(f_test_rep >= lambda_rep)
}

valid_estimates <- coverage_estimates[!is.na(coverage_estimates)]
mean_coverage <- mean(valid_estimates)
sd_coverage <- sd(valid_estimates)

#' **Out-of-sample coverage results** (over `r length(valid_estimates)` valid splits):
#' 
#' | Metric | Value |
#' |--------|-------|
#' | Mean coverage | `r sprintf("%.1f%%", 100 * mean_coverage)` |
#' | Standard deviation | `r sprintf("%.1f%%", 100 * sd_coverage)` |
#' | Target $(1-\alpha)$ | `r sprintf("%.1f%%", 100 * (1 - alpha))` |
#' 
#' **Interpretation:** The empirical coverage is close to the 90% target. Small deviations 
#' are expected due to finite sample sizes. The key insight is that conformal prediction 
#' provides a *distribution-free* guarantee — we did not assume any specific parametric 
#' form for the data, yet we achieve approximately the nominal coverage.

#'
#' # Results
#' 
#' ## Conformal Prediction Band
#' 
#' This is the **main output** of our analysis: a prediction band that should contain 
#' approximately 90% of future acceleration curves from the same distribution.
#' 
#' The blue shaded region shows the **envelope** — the outer boundary of the union of 
#' all GMM component bands. The gray lines are 30 randomly selected curves projected 
#' back to function space after basis truncation.

#+ plot-conformal-band, fig.width=12, fig.height=7
par(mar = c(4, 4, 3, 1))

# Plot band envelope
plot(NULL, xlim = c(0, 1), ylim = range(c(band_lower, band_upper), na.rm = TRUE) * 1.1,
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("Conformal Prediction Band ENVELOPE (%.0f%% target, p=%d, K=%d)", 
                    100*(1-alpha), p, K))

# Shade the envelope
polygon(c(grid01, rev(grid01)),
        c(band_lower, rev(band_upper)),
        col = adjustcolor("steelblue", alpha.f = 0.3), border = NA)

# Plot some curves
n_plot <- min(30, n)
plot_idx <- sample(1:n, n_plot)
for (i in plot_idx) {
  # Reconstruct projected curve
  xi_i <- project_curve(X[i, ], grid01, p)
  y_proj <- reconstruct_curve(xi_i, grid01)
  lines(grid01, y_proj, col = adjustcolor("gray40", alpha.f = 0.5), lwd = 0.5)
}

# Add envelope bounds
lines(grid01, band_lower, col = "steelblue", lwd = 2)
lines(grid01, band_upper, col = "steelblue", lwd = 2)

legend("topright", 
       c("Envelope (superset of union)", "Projected curves"),
       col = c("steelblue", "gray40"),
       lwd = c(2, 1), cex = 0.9, bty = "n")

#+ save-plot4, include=FALSE
png(file.path(output_dir, "04_conformal_band.png"), width = 1000, height = 600)
par(mar = c(4, 4, 3, 1))
plot(NULL, xlim = c(0, 1), ylim = range(c(band_lower, band_upper), na.rm = TRUE) * 1.1,
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("Conformal Prediction Band ENVELOPE (%.0f%% target, p=%d, K=%d)", 
                    100*(1-alpha), p, K))
polygon(c(grid01, rev(grid01)),
        c(band_lower, rev(band_upper)),
        col = adjustcolor("steelblue", alpha.f = 0.3), border = NA)
for (i in plot_idx) {
  xi_i <- project_curve(X[i, ], grid01, p)
  y_proj <- reconstruct_curve(xi_i, grid01)
  lines(grid01, y_proj, col = adjustcolor("gray40", alpha.f = 0.5), lwd = 0.5)
}
lines(grid01, band_lower, col = "steelblue", lwd = 2)
lines(grid01, band_upper, col = "steelblue", lwd = 2)
legend("topright", c("Envelope (superset of union)", "Projected curves"),
       col = c("steelblue", "gray40"), lwd = c(2, 1), cex = 0.9, bty = "n")
dev.off()

#' **Interpretation:** The band spans roughly ±6 m/s² around zero. This is wide enough 
#' to capture the variability from all three activities (standing, walking, fast walking) 
#' while still being informative. Notice that most projected curves (gray lines) stay 
#' well within the band boundaries. The band width is relatively uniform across time 
#' because the cosine basis functions have similar magnitudes throughout the interval.
#' 
#' **Practical meaning:** If we observe a new 10-second acceleration window from the 
#' same phone/person/conditions, we expect it to fall within this band with ~90% probability.
#'
#' ## Union-of-Intervals Visualization
#'
#' **Why multiple intervals?** Our prediction band comes from 3 GMM components. At each 
#' time point $t$, each component contributes an interval $[\ell_k(t), u_k(t)]$. When these
#' intervals overlap, they merge into one. But when components are well-separated (e.g.,
#' standing vs. walking), the intervals may be disjoint.
#'
#' **The envelope vs. the union:** The blue band in the previous plot shows the 
#' *envelope* — just the overall min and max. But the actual prediction set might be
#' smaller: a curve must fall within at least one component's interval, not just anywhere
#' between the overall min and max.
#'
#' The prediction band $B_n(t) = \bigcup_k [\ell_k(t), u_k(t)]$ (Paper Section 3.1) can have 
#' **disconnected slices** at each time point. The envelope plot above shows only the outer 
#' boundary (min/max), which is a superset of the actual union. Here we visualize the true
#' structure by plotting each interval separately.

#+ plot-union-intervals, fig.width=12, fig.height=7
par(mar = c(4, 4, 3, 1))

# Compute number of intervals at each timepoint
n_intervals <- sapply(band_intervals, length)

# Plot the union-of-intervals as stacked vertical segments
plot(NULL, xlim = c(0, 1), ylim = range(c(band_lower, band_upper), na.rm = TRUE) * 1.1,
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = "Union-of-Intervals Band B_n(t) (Paper Eq. 6)")

# Add light background for context
polygon(c(grid01, rev(grid01)),
        c(band_lower, rev(band_upper)),
        col = adjustcolor("gray90", alpha.f = 0.5), border = NA)

# Plot each interval as a vertical segment at each timepoint
for (m in seq(1, M, by = 2)) {  # Plot every other point to reduce clutter
  intervals_m <- band_intervals[[m]]
  if (length(intervals_m) == 0) next
  
  for (int_idx in seq_along(intervals_m)) {
    int <- intervals_m[[int_idx]]
    # Color-code by interval index
    int_col <- c("steelblue", "coral", "forestgreen", "purple", "orange")[((int_idx - 1) %% 5) + 1]
    segments(x0 = grid01[m], y0 = int$lower, x1 = grid01[m], y1 = int$upper,
             col = int_col, lwd = 1.5)
  }
}

# Add envelope bounds for reference
lines(grid01, band_lower, col = "black", lwd = 1, lty = 2)
lines(grid01, band_upper, col = "black", lwd = 1, lty = 2)

legend("topright", 
       c("Interval segments (by component)", "Envelope boundary"),
       col = c("steelblue", "black"),
       lwd = c(2, 1), lty = c(1, 2), cex = 0.9, bty = "n")

#+ save-plot-union, include=FALSE
png(file.path(output_dir, "04b_union_intervals.png"), width = 1000, height = 600)
par(mar = c(4, 4, 3, 1))
plot(NULL, xlim = c(0, 1), ylim = range(c(band_lower, band_upper), na.rm = TRUE) * 1.1,
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = "Union-of-Intervals Band B_n(t) (Paper Eq. 6)")
polygon(c(grid01, rev(grid01)), c(band_lower, rev(band_upper)),
        col = adjustcolor("gray90", alpha.f = 0.5), border = NA)
for (m in seq(1, M, by = 2)) {
  intervals_m <- band_intervals[[m]]
  if (length(intervals_m) == 0) next
  for (int_idx in seq_along(intervals_m)) {
    int <- intervals_m[[int_idx]]
    int_col <- c("steelblue", "coral", "forestgreen", "purple", "orange")[((int_idx - 1) %% 5) + 1]
    segments(x0 = grid01[m], y0 = int$lower, x1 = grid01[m], y1 = int$upper, col = int_col, lwd = 1.5)
  }
}
lines(grid01, band_lower, col = "black", lwd = 1, lty = 2)
lines(grid01, band_upper, col = "black", lwd = 1, lty = 2)
legend("topright", c("Interval segments (by component)", "Envelope boundary"),
       col = c("steelblue", "black"), lwd = c(2, 1), lty = c(1, 2), cex = 0.9, bty = "n")
dev.off()

#'
#' ## Diagnostics for Disconnected Slices
#'
#' **Why we track this:** Understanding the interval structure helps interpret the band:
#'
#' - **Single interval (n=1):** All GMM components overlap at this time — typical when
#'   activities are similar (e.g., all near zero in centered data)
#' - **Multiple intervals (n>1):** Components are separated — a new curve must match
#'   one of the distinct patterns, not just fall anywhere in between
#'
#' The band can have multiple disjoint intervals at each time slice if the GMM components
#' produce non-overlapping ellipsoid projections. We track this via `n_intervals[m]`, the
#' number of merged intervals at each timepoint.

#+ plot-n-intervals, fig.width=12, fig.height=5
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# Line plot of number of intervals over time
plot(grid01, n_intervals, type = "l", lwd = 2, col = "darkblue",
     xlab = "Normalized time", ylab = "Number of intervals",
     main = "Number of Intervals in B_n(t) over Time")
abline(h = 1, col = "red", lty = 2)
text(0.1, 1.1, "Single interval", col = "red", cex = 0.8)

# Histogram of number of intervals
hist(n_intervals, breaks = seq(0.5, max(n_intervals) + 0.5, by = 1),
     col = "steelblue", border = "white",
     xlab = "Number of intervals", ylab = "Frequency (timepoints)",
     main = "Distribution of Interval Counts")

#+ save-plot-nintervals, include=FALSE
png(file.path(output_dir, "04c_n_intervals_diagnostics.png"), width = 1000, height = 400)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(grid01, n_intervals, type = "l", lwd = 2, col = "darkblue",
     xlab = "Normalized time", ylab = "Number of intervals",
     main = "Number of Intervals in B_n(t) over Time")
abline(h = 1, col = "red", lty = 2)
hist(n_intervals, breaks = seq(0.5, max(n_intervals) + 0.5, by = 1),
     col = "steelblue", border = "white",
     xlab = "Number of intervals", ylab = "Frequency (timepoints)",
     main = "Distribution of Interval Counts")
dev.off()

#' **Summary statistics for interval counts:**

#+ n-intervals-summary
n_int_summary <- data.frame(
  Statistic = c("Min", "Max", "Mean", "Median", "% with >1 interval"),
  Value = c(min(n_intervals), max(n_intervals), 
            round(mean(n_intervals), 2), median(n_intervals),
            sprintf("%.1f%%", 100 * mean(n_intervals > 1)))
)
knitr::kable(n_int_summary)

#'
#' ### Sample of Intervals at Selected Timepoints
#'
#' The following table shows the actual interval structure at 10 evenly-spaced timepoints:

#+ intervals-table
# Select 10 evenly-spaced timepoints
sample_times <- round(seq(1, M, length.out = 10))
interval_table <- data.frame(
  Time_Index = sample_times,
  Time_Value = round(grid01[sample_times], 3),
  N_Intervals = n_intervals[sample_times],
  Intervals = sapply(sample_times, function(m) {
    ints <- band_intervals[[m]]
    if (length(ints) == 0) return("(empty)")
    paste(sapply(ints, function(int) sprintf("[%.2f, %.2f]", int$lower, int$upper)), 
          collapse = " ∪ ")
  })
)
knitr::kable(interval_table, col.names = c("Grid Index", "Time (t)", "# Intervals", "Intervals"))

#' **Observation:** In general, when GMM components are well-separated in coefficient space,
#' the band slices can be disconnected (multiple disjoint intervals). When components 
#' overlap substantially, the intervals merge into one.
#'
#' **In our fit:** If the diagnostics above show `max(n_intervals) = 1`, this means 
#' all ellipsoid projections overlap enough that slices merged into a single interval 
#' at every time point. This is expected when activities share similar baseline structure
#' (all centered curves oscillate around zero). The theoretical possibility of disconnected
#' slices remains — it would occur with more strongly separated activity patterns or 
#' different basis/threshold choices.

#'
#' ## Coefficient Space Visualization
#' 
#' This plot shows how the functional data looks after projection onto the first two 
#' basis coefficients. Each dot is one curve reduced to two numbers ($\xi_1$, $\xi_2$).

#+ plot-gmm-clusters, fig.width=10, fig.height=8
par(mar = c(4, 4, 3, 1))

# Project all curves
Xi_all <- t(apply(X, 1, function(y) project_curve(y, grid01, p)))

# Use first two coefficient dimensions for visualization
if (p >= 2) {
  # Handle unknown activities safely (use activity_code for color lookup)
  plot_cols <- cols[meta$activity_code]
  plot_cols[is.na(plot_cols)] <- cols["Unknown"]
  
  plot(Xi_all[, 1], Xi_all[, 2],
       col = adjustcolor(plot_cols, alpha.f = 0.6),
       pch = 19, cex = 0.8,
       xlab = expression(xi[1] ~ "(1st coefficient)"), 
       ylab = expression(xi[2] ~ "(2nd coefficient)"),
       main = "Coefficient Space (first 2 dimensions, NOT PCA)")
  
  # Add GMM component centers
  points(sapply(mu_list, function(m) m[1]),
         sapply(mu_list, function(m) m[2]),
         pch = 4, cex = 2, lwd = 3, col = "black")
  
  # Add ellipses for valid components (approximate 2D projection)
  for (k in 1:K) {
    if (ellipsoids[[k]]$valid) {
      # Draw 2D ellipse (using first 2x2 block of covariance)
      sigma_2d <- ellipsoids[[k]]$sigma[1:2, 1:2]
      mu_2d <- ellipsoids[[k]]$mu[1:2]
      r_k <- ellipsoids[[k]]$r
      
      # Eigendecomposition for ellipse
      eig <- eigen(sigma_2d)
      angles <- seq(0, 2*pi, length.out = 100)
      ellipse_pts <- r_k * cbind(cos(angles), sin(angles)) %*% t(eig$vectors %*% diag(sqrt(eig$values)))
      ellipse_pts <- sweep(ellipse_pts, 2, mu_2d, "+")
      
      lines(ellipse_pts[, 1], ellipse_pts[, 2], col = "black", lwd = 2, lty = 2)
    }
  }
  
  # Legend with activities present in data (use activity_code and map to pretty labels)
  present_activities <- intersect(names(cols), unique(meta$activity_code))
  legend("topright", 
         c(activity_labels[present_activities], "GMM centers", "Ellipsoid (2D proj)"),
         col = c(cols[present_activities], "black", "black"),
         pch = c(rep(19, length(present_activities)), 4, NA),
         lty = c(rep(NA, length(present_activities)), NA, 2),
         lwd = c(rep(NA, length(present_activities)), 3, 2),
         cex = 0.8, bty = "n")
}

#+ save-plot5, include=FALSE
png(file.path(output_dir, "05_gmm_clusters.png"), width = 800, height = 600)
par(mar = c(4, 4, 3, 1))
if (p >= 2) {
  plot_cols <- cols[meta$activity_code]
  plot_cols[is.na(plot_cols)] <- cols["Unknown"]
  plot(Xi_all[, 1], Xi_all[, 2],
       col = adjustcolor(plot_cols, alpha.f = 0.6), pch = 19, cex = 0.8,
       xlab = expression(xi[1] ~ "(1st coefficient)"), 
       ylab = expression(xi[2] ~ "(2nd coefficient)"),
       main = "Coefficient Space (first 2 dimensions, NOT PCA)")
  points(sapply(mu_list, function(m) m[1]),
         sapply(mu_list, function(m) m[2]),
         pch = 4, cex = 2, lwd = 3, col = "black")
  for (k in 1:K) {
    if (ellipsoids[[k]]$valid) {
      sigma_2d <- ellipsoids[[k]]$sigma[1:2, 1:2]
      mu_2d <- ellipsoids[[k]]$mu[1:2]
      r_k <- ellipsoids[[k]]$r
      eig <- eigen(sigma_2d)
      angles <- seq(0, 2*pi, length.out = 100)
      ellipse_pts <- r_k * cbind(cos(angles), sin(angles)) %*% t(eig$vectors %*% diag(sqrt(eig$values)))
      ellipse_pts <- sweep(ellipse_pts, 2, mu_2d, "+")
      lines(ellipse_pts[, 1], ellipse_pts[, 2], col = "black", lwd = 2, lty = 2)
    }
  }
  present_activities <- intersect(names(cols), unique(meta$activity_code))
  legend("topright", c(activity_labels[present_activities], "GMM centers", "Ellipsoid (2D proj)"),
         col = c(cols[present_activities], "black", "black"),
         pch = c(rep(19, length(present_activities)), 4, NA),
         lty = c(rep(NA, length(present_activities)), NA, 2),
         lwd = c(rep(NA, length(present_activities)), 3, 2), cex = 0.8, bty = "n")
}
dev.off()

#' **Interpretation:** The three activity types form somewhat distinct clusters in 
#' coefficient space:
#' 
#' - **Standing** (red): Concentrated near the center — low values on both axes because 
#'   standing produces minimal acceleration variation
#' - **Walking** (blue): Spread more widely — footstep patterns create larger coefficients
#' - **Fast Walking** (green): Similar to walking but with even more spread
#' 
#' The GMM captures this structure with three components (black crosses mark centers). 
#' The dashed ellipses show the 2D projection of the prediction region — points inside 
#' these ellipses have density above the threshold λ.

#'
#' ## Component-wise Bands
#' 
#' The prediction band is built from **three separate ellipsoids** in coefficient space, 
#' one per GMM component. This plot shows how each component contributes to the final band.

#+ plot-band-components, fig.width=12, fig.height=7
par(mar = c(4, 4, 3, 1))

# Compute component-wise bands
comp_colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00")

plot(NULL, xlim = c(0, 1), 
     ylim = range(c(band_lower, band_upper), na.rm = TRUE) * 1.1,
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("Component-wise Bands (K=%d) — Union forms B_n(t)", K))

for (k in 1:K) {
  if (!ellipsoids[[k]]$valid) next
  
  comp_lower <- numeric(M)
  comp_upper <- numeric(M)
  
  for (m in 1:M) {
    phi_t <- basis_vector(grid01[m], p)
    proj <- ellipsoid_projection(ellipsoids[[k]]$mu, ellipsoids[[k]]$sigma, 
                                  ellipsoids[[k]]$r, phi_t)
    comp_lower[m] <- proj$lower
    comp_upper[m] <- proj$upper
  }
  
  polygon(c(grid01, rev(grid01)),
          c(comp_lower, rev(comp_upper)),
          col = adjustcolor(comp_colors[k], alpha.f = 0.3), border = NA)
  
  lines(grid01, comp_lower, col = comp_colors[k], lwd = 1.5)
  lines(grid01, comp_upper, col = comp_colors[k], lwd = 1.5)
}

legend("topright",
       paste("Component", 1:K, "(π =", round(pi_k, 2), ")"),
       fill = adjustcolor(comp_colors[1:K], alpha.f = 0.3),
       border = comp_colors[1:K], cex = 0.8, bty = "n")

#+ save-plot6, include=FALSE
png(file.path(output_dir, "06_band_components.png"), width = 1000, height = 600)
par(mar = c(4, 4, 3, 1))
plot(NULL, xlim = c(0, 1), ylim = range(c(band_lower, band_upper), na.rm = TRUE) * 1.1,
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("Component-wise Bands (K=%d) — Union forms B_n(t)", K))
for (k in 1:K) {
  if (!ellipsoids[[k]]$valid) next
  comp_lower <- numeric(M)
  comp_upper <- numeric(M)
  for (m in 1:M) {
    phi_t <- basis_vector(grid01[m], p)
    proj <- ellipsoid_projection(ellipsoids[[k]]$mu, ellipsoids[[k]]$sigma, 
                                  ellipsoids[[k]]$r, phi_t)
    comp_lower[m] <- proj$lower
    comp_upper[m] <- proj$upper
  }
  polygon(c(grid01, rev(grid01)), c(comp_lower, rev(comp_upper)),
          col = adjustcolor(comp_colors[k], alpha.f = 0.3), border = NA)
  lines(grid01, comp_lower, col = comp_colors[k], lwd = 1.5)
  lines(grid01, comp_upper, col = comp_colors[k], lwd = 1.5)
}
legend("topright", paste("Component", 1:K, "(π =", round(pi_k, 2), ")"),
       fill = adjustcolor(comp_colors[1:K], alpha.f = 0.3),
       border = comp_colors[1:K], cex = 0.8, bty = "n")
dev.off()

#' **Interpretation:** The three colored bands correspond to the three GMM components. 
#' The mixing weights (π) in the legend show how much each component contributes:
#' 
#' - Components with higher π contribute more to the overall density
#' - Narrower bands (like the red one) capture low-variability activities (standing)
#' - Wider bands capture high-variability activities (walking)
#' 
#' The final conformal band (shown in the previous plot) is the **union** of these three 
#' bands — any point within any of the three colored regions is considered "conforming."

#+ save-results-sec3, include=FALSE
## Save Section 3 results only (Section 4 results saved at end of Section 4)
results_sec3 <- list(
  config = list(alpha = alpha, split_ratio = split_ratio, p = p, K = K, seed = 2026),
  gmm = list(pi_k = pi_k, mu_list = mu_list, sigma_list = sigma_list),
  lambda = lambda, lambda_idx = lambda_idx, lambda_idx_clamped = lambda_idx_clamped,
  ellipsoids = ellipsoids,
  band = list(grid = grid01, lower = band_lower, upper = band_upper, intervals = band_intervals),
  data_info = list(n = n, n1 = n1, n2 = n2, M = M)
)
saveRDS(results_sec3, file.path(output_dir, "conformal_results_sec3.rds"))

#'
#' # Section 4 — Pseudo-Density Methods and Conformal Cluster Tree
#'
#' **A different approach:** Section 3 reduced curves to 5 coefficients and applied 
#' standard multivariate methods. Section 4 takes a different route — it works directly 
#' with the full curves using a "pseudo-density" that measures how crowded the 
#' neighborhood around each curve is.
#'
#' **Why "pseudo"?** In infinite-dimensional spaces (like the space of all functions), 
#' true probability densities don't exist mathematically. But we can still define a 
#' function that behaves like a density for practical purposes — higher values mean 
#' the curve is in a "dense" region with many similar neighbors.
#'
#' This approach is particularly useful for:
#' 
#' - **Anomaly detection:** Curves with low pseudo-density are outliers
#' - **Finding prototypes:** Local maxima of pseudo-density represent "typical" patterns
#' - **Cluster discovery:** High-density regions form natural groups
#'
#' ## 4.1 L2 Distance and Pairwise Distance Matrix
#'
#' **What is L2 distance?** It measures how different two curves are by integrating 
#' the squared difference over time. Two curves that track each other closely have 
#' small L2 distance; curves with different shapes have large distance.
#'
#' **Why compute a full matrix?** To evaluate pseudo-density at any curve, we need its 
#' distance to all other curves. Computing this matrix once (183×183 = 33,489 pairs) 
#' saves time compared to recomputing distances repeatedly.
#'
#' We first define the $L_2$ distance between functional curves on the grid:
#' $$d(X_i, X_j) = \left(\int_0^1 (X_i(t) - X_j(t))^2 dt\right)^{1/2}$$
#' 
#' This is discretized using the trapezoidal rule over our 200-point grid.

#+ sec4-distance-setup

## --- Section 4: Distance function and matrix ---

# L2 distance between two curves (discretized via trapezoidal rule)
l2_distance <- function(y1, y2, grid) {
  # y1, y2: curve values at grid points (length M)
  # grid: grid points in [0,1] (length M)
  # Returns: L2 distance
  
  M_local <- length(grid)
  dt <- grid[2] - grid[1]
  
  diff_sq <- (y1 - y2)^2
  
  # Trapezoidal rule: ∫f ≈ dt * (f_1/2 + f_2 + ... + f_{M-1} + f_M/2)
  integral <- dt * (diff_sq[1]/2 + sum(diff_sq[2:(M_local-1)]) + diff_sq[M_local]/2)
  
  return(sqrt(integral))
}

# Compute full pairwise distance matrix (symmetric, diagonal = 0)
cat("Computing pairwise distance matrix...\n")
D_matrix <- matrix(0, nrow = n, ncol = n)
for (i in 1:(n-1)) {
  for (j in (i+1):n) {
    d_ij <- l2_distance(X[i, ], X[j, ], grid01)
    D_matrix[i, j] <- d_ij
    D_matrix[j, i] <- d_ij
  }
}
cat("Distance matrix computed.\n")

#' **Distance matrix validation:**
#'
#' We verify the distance matrix has the expected mathematical properties:
#'
#' - **Symmetric:** $d(X_i, X_j) = d(X_j, X_i)$ — distance doesn't depend on direction
#' - **Zero diagonal:** $d(X_i, X_i) = 0$ — a curve has zero distance to itself
#' - **Finite values:** No computation errors producing NA or infinity

#+ sec4-distance-validation
dist_validation <- data.frame(
  Check = c("Symmetric", "Diagonal ~0", "No NA/Inf", "Min off-diag", "Max off-diag", "Mean off-diag"),
  Result = c(
    all(abs(D_matrix - t(D_matrix)) < 1e-10),
    all(abs(diag(D_matrix)) < 1e-10),
    all(is.finite(D_matrix)),
    round(min(D_matrix[row(D_matrix) != col(D_matrix)]), 3),
    round(max(D_matrix), 3),
    round(mean(D_matrix[row(D_matrix) != col(D_matrix)]), 3)
  )
)
knitr::kable(dist_validation)

#'
#' ## 4.2 Pseudo-Density Estimator (Paper Eq. 10)
#'
#' Following Ferraty & Vieu (2006), we define the pseudo-density:
#' $$\hat{p}_h(u) = \frac{1}{n} \sum_{i=1}^n K\left(\frac{d(u, X_i)}{h}\right)$$
#'
#' where $K(z) = \exp(-z^2/2)$ is the Gaussian kernel satisfying $K(z) \leq K(0) = 1$.
#'
#' **Note:** This is not a true density (no dominating σ-finite measure exists in infinite
#' dimensions), but it captures the notion of "local density" in function space.

#+ sec4-pseudo-density

## --- Section 4: Gaussian kernel and pseudo-density ---

# Gaussian kernel satisfying K(z) <= K(0) = 1
K_gaussian <- function(z) {
  exp(-z^2 / 2)
}
K0 <- K_gaussian(0)  # K(0) = 1

# Pseudo-density estimator (Eq. 10)
# Computes p_hat_h(X_i) for all i using precomputed distance matrix
compute_pseudo_density <- function(D_mat, h) {
  # D_mat: n x n distance matrix
  # h: bandwidth
  # Returns: n-vector of pseudo-density values
  
  n_local <- nrow(D_mat)
  p_hat <- numeric(n_local)
  
  for (i in 1:n_local) {
    # Sum of K(d(X_i, X_j)/h) for all j
    p_hat[i] <- mean(K_gaussian(D_mat[i, ] / h))
  }
  
  return(p_hat)
}

#'
#' ### Bandwidth Selection (Paper Figure 5 caption)
#'
#' The paper suggests choosing $h$ to **maximize the variance** of $\hat{p}_h(X_i)$ over
#' a grid of candidate values. This ensures the pseudo-density discriminates well between
#' regions of high and low "density."

#+ sec4-bandwidth-selection, fig.width=10, fig.height=5

# Candidate bandwidths: quantiles of pairwise distances
offdiag_dists <- D_matrix[row(D_matrix) != col(D_matrix)]
h_candidates <- quantile(offdiag_dists, probs = seq(0.05, 0.95, by = 0.05))

# Compute variance of p_hat for each candidate h
h_variances <- sapply(h_candidates, function(h) {
  p_hat_h <- compute_pseudo_density(D_matrix, h)
  var(p_hat_h)
})

# Select h that maximizes variance
h_opt_idx <- which.max(h_variances)
h_selected <- h_candidates[h_opt_idx]

par(mar = c(4, 4, 3, 1))
plot(h_candidates, h_variances, type = "b", pch = 19, col = "darkblue",
     xlab = "Bandwidth h", ylab = "Var(p_hat_h)",
     main = "Bandwidth Selection: Maximize Variance of Pseudo-Density")
abline(v = h_selected, col = "red", lwd = 2, lty = 2)
text(h_selected, max(h_variances) * 0.9, sprintf("h = %.2f", h_selected), 
     col = "red", pos = 4)

#+ save-plot-bandwidth, include=FALSE
png(file.path(output_dir, "07_bandwidth_selection.png"), width = 800, height = 400)
par(mar = c(4, 4, 3, 1))
plot(h_candidates, h_variances, type = "b", pch = 19, col = "darkblue",
     xlab = "Bandwidth h", ylab = "Var(p_hat_h)",
     main = "Bandwidth Selection: Maximize Variance of Pseudo-Density")
abline(v = h_selected, col = "red", lwd = 2, lty = 2)
text(h_selected, max(h_variances) * 0.9, sprintf("h = %.2f", h_selected), col = "red", pos = 4)
dev.off()

#' **Selected bandwidth:** $h$ = `r round(h_selected, 2)`
#'
#' **Interpretation:** The bandwidth $h$ controls how "local" the density estimate is.
#' Small $h$ gives high weight only to very close neighbors (sensitive to local structure),
#' while large $h$ averages over a wider neighborhood (smoother but may miss details).

#+ sec4-compute-pseudo-density
# Compute pseudo-density at selected bandwidth
p_hat <- compute_pseudo_density(D_matrix, h_selected)

#' **Pseudo-density validation:**
#'
#' We verify the pseudo-density values make sense:
#'
#' - **Range (0, 1]:** Since we use $K(z) = e^{-z^2/2}$ and average over $n$ curves,
#'   the maximum is 1 (when a curve equals many others) and minimum approaches 0
#'   (when a curve is far from all others)
#' - **Variation:** A good bandwidth should produce varied values — if all curves 
#'   have similar pseudo-density, we can't distinguish typical from atypical

#+ sec4-pdensity-validation
pdensity_validation <- data.frame(
  Property = c("Min p_hat", "Max p_hat", "Mean p_hat", "SD p_hat", "All in (0, 1]"),
  Value = c(round(min(p_hat), 4), round(max(p_hat), 4), 
            round(mean(p_hat), 4), round(sd(p_hat), 4),
            all(p_hat > 0 & p_hat <= 1))
)
knitr::kable(pdensity_validation)

#'
#' ## 4.3 Conformal Set Approximation (Paper Eq. 11)
#'
#' The conformal prediction set based on pseudo-density is approximated as:
#' $$C_{n,\alpha}^+ = \{f : \hat{p}_h(f) \geq \lambda - n^{-1}K(0)\}$$
#'
#' where $\lambda = \hat{p}_h(X_{(n\alpha)})$ is the $\alpha$-quantile of pseudo-density values.
#'
#' **Lemma 4.1 (from paper):** The set $C_{n,\alpha}^+$ contains the exact conformal set 
#' $C_{n,\alpha}$, so it has coverage at least $(1-\alpha)$. The sample approximation is:
#' $$\hat{C}_{n,\alpha} = C_{n,\alpha}^+ \cap \{X_1, \ldots, X_n\}$$

#+ sec4-conformal-approx

# Conformal threshold computation (Eq. 11)
Cn_plus_threshold <- function(alpha_level, p_hat_vals, n_obs) {
  # alpha_level: miscoverage level
  # p_hat_vals: pseudo-density values for all observations
  # n_obs: number of observations
  # Returns: threshold for C^+_{n,alpha}
  
  # Sort pseudo-densities
  p_sorted <- sort(p_hat_vals)
  
  # lambda = p_hat(X_{(n*alpha)}) - handle non-integer with floor
  rank_idx <- max(1, floor(n_obs * alpha_level))
  lambda <- p_sorted[rank_idx]
  
  # Threshold for C^+: lambda - (1/n)*K(0)
  threshold <- lambda - (1/n_obs) * K0
  
  return(list(lambda = lambda, threshold = threshold, rank_idx = rank_idx))
}

# Indices of curves in C_hat (sample approximation)
C_hat_indices <- function(alpha_level, p_hat_vals, n_obs) {
  thresh_info <- Cn_plus_threshold(alpha_level, p_hat_vals, n_obs)
  which(p_hat_vals >= thresh_info$threshold)
}

#'
#' ### Threshold Index Verification (Section 4)
#'
#' **Indexing convention:** We use $\lambda = \hat{p}_h(X_{(\lfloor n\alpha \rfloor)})$ (floor).
#' Some papers use $\lceil n\alpha \rceil$ (ceiling) for a more conservative set. We compare:

#+ sec4-threshold-verification
# Compare floor vs ceiling for Section 4 threshold
alpha_test <- 0.10  # Use same alpha as Section 3 for comparison
p_sorted_sec4 <- sort(p_hat)

idx_floor <- max(1, floor(n * alpha_test))
idx_ceiling <- max(1, ceiling(n * alpha_test))

lambda_floor <- p_sorted_sec4[idx_floor]
lambda_ceiling <- p_sorted_sec4[idx_ceiling]

# Compute thresholds with both
thresh_floor <- lambda_floor - (1/n) * K0
thresh_ceiling <- lambda_ceiling - (1/n) * K0

size_floor <- sum(p_hat >= thresh_floor)
size_ceiling <- sum(p_hat >= thresh_ceiling)

sec4_comparison <- data.frame(
  Convention = c("⌊nα⌋ (our choice)", "⌈nα⌉ (alternative)"),
  Rank_Index = c(idx_floor, idx_ceiling),
  Lambda = sprintf("%.6f", c(lambda_floor, lambda_ceiling)),
  C_hat_Size = c(size_floor, size_ceiling),
  Pct_of_n = sprintf("%.1f%%", 100 * c(size_floor/n, size_ceiling/n))
)
knitr::kable(sec4_comparison, caption = "Comparison of floor vs ceiling indexing for Section 4")

#' **Result:** We use `floor(n*alpha)` as it matches the paper's definition in Eq.(11). The difference
#' between floor and ceiling is at most one rank position and yields similar set sizes.

#' **Verification:** $|\hat{C}_{n,\alpha}|$ should decrease as $\alpha$ increases (more exclusive set).
#'
#' **How to read this table:**
#'
#' The parameter $\alpha$ controls **conformal exclusiveness**, not a guaranteed retained fraction.
#' Per Eq.(11), the threshold $\lambda$ is set as the $\alpha$-quantile of pseudo-density values,
#' so approximately $\lfloor n\alpha \rfloor$ calibration curves fall below it. However, the 
#' actual size of $\hat{C}_{n,\alpha}$ depends on the threshold adjustment ($-K(0)/n$) and 
#' the density distribution.
#'
#' The **conformal guarantee** says: if a new curve is exchangeable with our data, it falls 
#' in $\hat{C}_{n,\alpha}$ with probability at least $(1-\alpha)$. Larger $\alpha$ yields
#' a smaller, more exclusive set with weaker coverage guarantee.

#+ sec4-conformal-verify
alpha_test_grid <- c(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
C_hat_sizes <- sapply(alpha_test_grid, function(a) length(C_hat_indices(a, p_hat, n)))

conformal_size_df <- data.frame(
  Alpha = alpha_test_grid,
  C_hat_size = C_hat_sizes,
  Pct_of_n = sprintf("%.1f%%", 100 * C_hat_sizes / n)
)
knitr::kable(conformal_size_df)

#' **Observation:** As $\alpha$ increases (requiring higher density), $|\hat{C}_{n,\alpha}|$
#' decreases, correctly reflecting the nested structure of conformal sets. This nesting is
#' crucial: $\hat{C}_{n,0.90} \subset \hat{C}_{n,0.50} \subset \hat{C}_{n,0.10}$.

#'
#' ## 4.4 Anomalies, Median, and High-Density Subsets (Figure 4 style)
#'
#' **Practical application:** One of the most useful things we can do with pseudo-density
#' is rank curves from "most typical" to "most unusual." This has immediate practical uses:
#'
#' - **Quality control:** Flag unusual sensor readings for manual review
#' - **Data cleaning:** Identify recording errors or mislabeled activities
#' - **Feature discovery:** Understand what makes some curves different
#'
#' Following the paper's Figure 4, we define subsets using the **conformal set** 
#' $\hat{C}_{n,\alpha}$ at different $\alpha$ levels:
#'
#' - **Anomalies:** Curves *outside* $\hat{C}_{n,0.05}$ — the ~5% with lowest pseudo-density
#' - **Median/Typical:** Curves *inside* $\hat{C}_{n,0.50}$ — the higher-density portion
#' - **High-Density:** Curves *inside* $\hat{C}_{n,0.05}$ — excludes only extreme outliers
#'
#' Note: The actual subset sizes depend on the threshold computation (Eq. 11), so they
#' may not be exactly 5%/50%/95% of the data.

#+ sec4-subsets

# Rank curves by pseudo-density (1 = lowest, n = highest)
p_hat_ranks <- rank(p_hat)

# Define subsets using conformal sets at chosen alpha levels (consistent with Eq. 11)
# Anomalies: curves NOT in C_hat at alpha=0.05 (i.e., lowest ~5% by density)
idx_high_density <- C_hat_indices(0.05, p_hat, n)  # C_hat at alpha=0.05
idx_anomalies <- setdiff(1:n, idx_high_density)     # Complement = anomalies

# Median: curves in C_hat at alpha=0.50
idx_median <- C_hat_indices(0.50, p_hat, n)

#' **Subset sizes** (defined via $\hat{C}_{n,\alpha}$):

#+ sec4-subset-sizes
subset_sizes <- data.frame(
  Subset = c("Anomalies (outside C_hat at α=0.05)", 
             "Median/Typical (in C_hat at α=0.50)", 
             "High-Density (in C_hat at α=0.05)"),
  Alpha_used = c("0.05 (complement)", "0.50", "0.05"),
  N_curves = c(length(idx_anomalies), length(idx_median), length(idx_high_density)),
  Pct = c(sprintf("%.1f%%", 100*length(idx_anomalies)/n),
          sprintf("%.1f%%", 100*length(idx_median)/n),
          sprintf("%.1f%%", 100*length(idx_high_density)/n))
)
knitr::kable(subset_sizes)

#'
#' ### Figure 4-style Visualization

#+ sec4-figure4-plots, fig.width=12, fig.height=10
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

# Panel A: All curves
plot(NULL, xlim = c(0, 1), ylim = range(X),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("(a) All Curves (n=%d)", n))
for (i in 1:n) {
  lines(grid01, X[i, ], col = adjustcolor("gray30", alpha.f = 0.3), lwd = 0.5)
}

# Panel B: Anomalies only
plot(NULL, xlim = c(0, 1), ylim = range(X),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("(b) Anomalies — Bottom 5%% (n=%d)", length(idx_anomalies)))
for (i in idx_anomalies) {
  lines(grid01, X[i, ], col = adjustcolor("red", alpha.f = 0.6), lwd = 1)
}

# Panel C: Median/Typical (top 50%)
plot(NULL, xlim = c(0, 1), ylim = range(X),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("(c) Median Set — Top 50%% (n=%d)", length(idx_median)))
for (i in idx_median) {
  lines(grid01, X[i, ], col = adjustcolor("blue", alpha.f = 0.4), lwd = 0.5)
}

# Panel D: High-Density (top 95%)
plot(NULL, xlim = c(0, 1), ylim = range(X),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("(d) High-Density — Top 95%% (n=%d)", length(idx_high_density)))
for (i in idx_high_density) {
  lines(grid01, X[i, ], col = adjustcolor("forestgreen", alpha.f = 0.3), lwd = 0.5)
}

# Panel E: Anomalies vs High-Density overlay
plot(NULL, xlim = c(0, 1), ylim = range(X),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = "(e) Anomalies (red) vs High-Density (green)")
for (i in idx_high_density) {
  lines(grid01, X[i, ], col = adjustcolor("forestgreen", alpha.f = 0.2), lwd = 0.5)
}
for (i in idx_anomalies) {
  lines(grid01, X[i, ], col = adjustcolor("red", alpha.f = 0.8), lwd = 1.5)
}
legend("topright", c("Anomalies", "High-Density"), 
       col = c("red", "forestgreen"), lwd = c(2, 1), cex = 0.8, bty = "n")

# Panel F: Reserved for prototypes (next section)
plot(NULL, xlim = c(0, 1), ylim = range(X),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = "(f) Prototypes — see next section")
text(0.5, 0, "Computed in Section 4.5", cex = 1.2)

#+ save-plot-figure4, include=FALSE
png(file.path(output_dir, "08_pseudodensity_subsets.png"), width = 1200, height = 800)
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
plot(NULL, xlim = c(0, 1), ylim = range(X), xlab = "Normalized time", 
     ylab = "Centered acceleration (m/s²)", main = sprintf("(a) All Curves (n=%d)", n))
for (i in 1:n) lines(grid01, X[i, ], col = adjustcolor("gray30", alpha.f = 0.3), lwd = 0.5)
plot(NULL, xlim = c(0, 1), ylim = range(X), xlab = "Normalized time", 
     ylab = "Centered acceleration (m/s²)", main = sprintf("(b) Anomalies — Bottom 5%% (n=%d)", length(idx_anomalies)))
for (i in idx_anomalies) lines(grid01, X[i, ], col = adjustcolor("red", alpha.f = 0.6), lwd = 1)
plot(NULL, xlim = c(0, 1), ylim = range(X), xlab = "Normalized time", 
     ylab = "Centered acceleration (m/s²)", main = sprintf("(c) Median Set — Top 50%% (n=%d)", length(idx_median)))
for (i in idx_median) lines(grid01, X[i, ], col = adjustcolor("blue", alpha.f = 0.4), lwd = 0.5)
plot(NULL, xlim = c(0, 1), ylim = range(X), xlab = "Normalized time", 
     ylab = "Centered acceleration (m/s²)", main = sprintf("(d) High-Density — Top 95%% (n=%d)", length(idx_high_density)))
for (i in idx_high_density) lines(grid01, X[i, ], col = adjustcolor("forestgreen", alpha.f = 0.3), lwd = 0.5)
plot(NULL, xlim = c(0, 1), ylim = range(X), xlab = "Normalized time", 
     ylab = "Centered acceleration (m/s²)", main = "(e) Anomalies (red) vs High-Density (green)")
for (i in idx_high_density) lines(grid01, X[i, ], col = adjustcolor("forestgreen", alpha.f = 0.2), lwd = 0.5)
for (i in idx_anomalies) lines(grid01, X[i, ], col = adjustcolor("red", alpha.f = 0.8), lwd = 1.5)
legend("topright", c("Anomalies", "High-Density"), col = c("red", "forestgreen"), lwd = c(2, 1), cex = 0.8, bty = "n")
plot.new()
text(0.5, 0.5, "See Section 4.5 for prototypes", cex = 1.2)
dev.off()

#' **Observation:** The anomalies (red curves in panel b, e) tend to have unusual shapes 
#' compared to the bulk of the data. The high-density set captures the "typical" behavior
#' while allowing for natural variability.

#'
#' ## 4.5 Mean-Shift Prototypes (Modes)
#'
#' **What are prototypes?** A prototype is a "most representative" curve for a cluster —
#' the curve that would be at the peak of a density hill if we visualized pseudo-density
#' as a mountain range. Each activity type (standing, walking, fast walking) should 
#' have its own prototype.
#'
#' **How mean-shift works:** Imagine placing a ball on a density landscape. The ball 
#' rolls uphill until it reaches a local peak (mode). Mean-shift does this mathematically:
#'
#' 1. Start from a curve (initial position)
#' 2. Compute weighted average of all curves, with nearby curves weighted more
#' 3. Move to this weighted average (shift "uphill")
#' 4. Repeat until convergence (the curve stops moving)
#'
#' The converged curve is a prototype — a local maximum of pseudo-density.
#'
#' Following Cheng (1995) and the paper's description, we use the **mean-shift algorithm**
#' to find modes (local maxima) of the pseudo-density. Starting from initial curves, we
#' iteratively shift toward the weighted mean:
#'
#' $$u^{(s+1)}(t) = \frac{\sum_i w_i X_i(t)}{\sum_i w_i}, \quad w_i = K\left(\frac{d(u^{(s)}, X_i)}{h}\right)$$

#+ sec4-mean-shift

## --- Mean-shift algorithm for functional data ---

mean_shift_functional <- function(u_init, X_mat, grid, h, tol = 1e-4, max_iter = 100) {
  # u_init: initial curve (length M vector)
  # X_mat: n x M matrix of all curves
  # grid: time grid
  # h: bandwidth
  # tol: convergence tolerance (L2 distance)
  # max_iter: maximum iterations
  # Returns: converged prototype curve and convergence info
  
  u_current <- u_init
  n_local <- nrow(X_mat)
  
  for (iter in 1:max_iter) {
    # Compute weights: w_i = K(d(u, X_i)/h)
    distances <- sapply(1:n_local, function(i) l2_distance(u_current, X_mat[i, ], grid))
    weights <- K_gaussian(distances / h)
    
    # Weighted mean update
    if (sum(weights) < 1e-10) {
      # All weights essentially zero - return current
      return(list(prototype = u_current, converged = FALSE, iterations = iter, 
                  final_shift = NA))
    }
    
    u_new <- colSums(weights * X_mat) / sum(weights)
    
    # Check convergence
    shift_dist <- l2_distance(u_new, u_current, grid)
    
    if (shift_dist < tol) {
      return(list(prototype = u_new, converged = TRUE, iterations = iter, 
                  final_shift = shift_dist))
    }
    
    u_current <- u_new
  }
  
  return(list(prototype = u_current, converged = FALSE, iterations = max_iter, 
              final_shift = shift_dist))
}

# Initialize STRATIFIED across activities (not just top density)
# This ensures we find prototypes for each activity type, not just the dominant mode
init_idx <- c()
for (act in c("Stand", "Walk", "Fast_Walk")) {
  act_idx <- which(meta$activity_code == act)
  if (length(act_idx) > 0) {
    # Pick top 2-3 by pseudo-density within each activity
    act_p_hat <- p_hat[act_idx]
    top_in_act <- act_idx[order(act_p_hat, decreasing = TRUE)[1:min(3, length(act_idx))]]
    init_idx <- c(init_idx, top_in_act)
  }
}
n_inits <- length(init_idx)

cat("Running mean-shift from", n_inits, "stratified initializations...\n")

# Run mean-shift from each initialization
prototypes_raw <- list()
for (i in seq_along(init_idx)) {
  result <- mean_shift_functional(X[init_idx[i], ], X, grid01, h_selected)
  prototypes_raw[[i]] <- list(
    prototype = result$prototype,
    converged = result$converged,
    iterations = result$iterations,
    init_idx = init_idx[i],
    init_activity = meta$activity_code[init_idx[i]]
  )
}

# Deduplicate converged modes with TIGHTER threshold to preserve distinct modes
dedup_threshold <- h_selected / 4  # Tighter threshold (was h/2)

unique_prototypes <- list()
for (proto in prototypes_raw) {
  if (!proto$converged) next
  
  # Check if this prototype is new (different from existing ones)
  is_new <- TRUE
  for (existing in unique_prototypes) {
    d_proto <- l2_distance(proto$prototype, existing$prototype, grid01)
    if (d_proto < dedup_threshold) {
      is_new <- FALSE
      break
    }
  }
  
  if (is_new) {
    unique_prototypes[[length(unique_prototypes) + 1]] <- proto
  }
}

n_prototypes <- length(unique_prototypes)
cat("Found", n_prototypes, "unique prototypes.\n")

# Assign each prototype to its nearest activity by finding the closest curve
for (i in seq_along(unique_prototypes)) {
  proto_curve <- unique_prototypes[[i]]$prototype
  # Find nearest data curve
  dists_to_data <- sapply(1:n, function(j) l2_distance(proto_curve, X[j, ], grid01))
  nearest_idx <- which.min(dists_to_data)
  unique_prototypes[[i]]$nearest_activity <- meta$activity_code[nearest_idx]
  unique_prototypes[[i]]$nearest_activity_label <- meta$activity[nearest_idx]
}

#' **Mean-shift results:**
#'
#' The table below shows each unique prototype found. Key columns:
#'
#' - **Converged:** Did the algorithm settle on a stable curve? (Should be TRUE)
#' - **Iterations:** How many steps to converge — fewer is better
#' - **Nearest_Activity:** The activity label of the closest data curve to this prototype.
#'   This indicates which activity type the prototype represents.

#+ sec4-meanshift-results
if (n_prototypes > 0) {
  proto_df <- data.frame(
    Prototype = 1:n_prototypes,
    Converged = sapply(unique_prototypes, function(p) p$converged),
    Iterations = sapply(unique_prototypes, function(p) p$iterations),
    Nearest_Activity = sapply(unique_prototypes, function(p) p$nearest_activity_label)
  )
  knitr::kable(proto_df)
} else {
  cat("No prototypes found (all initializations failed to converge).\n")
}

#'
#' ### Prototype Visualization (Paper Figure 4f style)

#+ sec4-prototype-plot, fig.width=10, fig.height=6
par(mar = c(4, 4, 3, 1))

# Plot background curves (light gray)
plot(NULL, xlim = c(0, 1), ylim = range(X),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("Mean-Shift Prototypes (n=%d modes, h=%.2f)", n_prototypes, h_selected))

for (i in 1:n) {
  lines(grid01, X[i, ], col = adjustcolor("gray80", alpha.f = 0.3), lwd = 0.3)
}

# Plot prototypes with distinct colors and activity labels
proto_colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33")
if (n_prototypes > 0) {
  proto_legend_labels <- sapply(1:n_prototypes, function(i) {
    sprintf("Proto %d (%s)", i, unique_prototypes[[i]]$nearest_activity_label)
  })
  for (i in 1:n_prototypes) {
    lines(grid01, unique_prototypes[[i]]$prototype, 
          col = proto_colors[(i-1) %% length(proto_colors) + 1], lwd = 3)
  }
  legend("topright", proto_legend_labels,
         col = proto_colors[1:n_prototypes], lwd = 3, cex = 0.8, bty = "n")
}

# Add zero reference line
abline(h = 0, col = "black", lty = 2)

#+ save-plot-prototypes, include=FALSE
png(file.path(output_dir, "09_mean_shift_prototypes.png"), width = 800, height = 500)
par(mar = c(4, 4, 3, 1))
plot(NULL, xlim = c(0, 1), ylim = range(X),
     xlab = "Normalized time", ylab = "Centered acceleration (m/s²)",
     main = sprintf("Mean-Shift Prototypes (n=%d modes, h=%.2f)", n_prototypes, h_selected))
for (i in 1:n) lines(grid01, X[i, ], col = adjustcolor("gray80", alpha.f = 0.3), lwd = 0.3)
if (n_prototypes > 0) {
  proto_legend_labels <- sapply(1:n_prototypes, function(i) {
    sprintf("Proto %d (%s)", i, unique_prototypes[[i]]$nearest_activity_label)
  })
  for (i in 1:n_prototypes) {
    lines(grid01, unique_prototypes[[i]]$prototype, 
          col = proto_colors[(i-1) %% length(proto_colors) + 1], lwd = 3)
  }
  legend("topright", proto_legend_labels,
         col = proto_colors[1:n_prototypes], lwd = 3, cex = 0.8, bty = "n")
}
abline(h = 0, col = "black", lty = 2)
dev.off()

#' **Observation:** The prototypes represent characteristic curve shapes (local modes of
#' pseudo-density). Given our dataset with three activities, we expect prototypes 
#' corresponding to: (1) flat/near-zero curves (Standing), (2) moderate oscillations 
#' (Walking), and (3) larger oscillations (Fast Walking).

#'
#' ## 4.6 Conformal Cluster Tree
#'
#' **What is a cluster tree?** As we make $\alpha$ more stringent (higher), we exclude 
#' more curves from $\hat{C}_{n,\alpha}$. At some point, removing low-density curves 
#' might "break" the connectivity — what was one cluster splits into two separate groups.
#'
#' **Why it's useful:** The tree reveals hierarchical structure in the data:
#'
#' - At low $\alpha$: All curves may be in one connected cluster
#' - At intermediate $\alpha$: Distinct activity types may separate into different clusters
#' - At high $\alpha$: Only the most central curves remain — possibly fragmenting further
#'
#' **The $\epsilon$ parameter:** Two curves are "connected" if their distance is ≤ $\epsilon$.
#' We choose $\epsilon$ based on nearest-neighbor distances to ensure reasonable connectivity.
#'
#' The **conformal cluster tree** visualizes how prediction sets evolve with $\alpha$.
#' 
#' For a given $\epsilon > 0$, we define graph $G_{\alpha,\epsilon}$ where:
#' - Nodes = curves in $\hat{C}_{n,\alpha}$
#' - Edges connect $X_i$ and $X_j$ if $d(X_i, X_j) \leq \epsilon$
#'
#' Clusters at level $\alpha$ are the **connected components** of $G_{\alpha,\epsilon}$.

#+ sec4-cluster-tree-setup

## --- Epsilon selection (Paper Figure 5 caption) ---
## Use epsilon = 0.5 * max nearest-neighbor distance within C_hat at reference alpha

alpha_ref <- 0.5
idx_ref <- C_hat_indices(alpha_ref, p_hat, n)

if (length(idx_ref) > 1) {
  # Compute nearest-neighbor distances within C_hat
  D_sub <- D_matrix[idx_ref, idx_ref]
  diag(D_sub) <- Inf  # Exclude self
  nn_dists <- apply(D_sub, 1, min)
  epsilon_selected <- 0.5 * max(nn_dists)
} else {
  epsilon_selected <- h_selected  # Fallback
}

#' **Selected $\epsilon$:** `r round(epsilon_selected, 2)` (from NN distances at $\alpha$ = `r alpha_ref`)

#+ sec4-build-cluster-tree

## --- Build cluster tree over alpha grid ---

alpha_grid <- seq(0.02, 0.98, by = 0.02)

# Function to find connected components
find_connected_components <- function(adj_matrix) {
  # adj_matrix: n x n adjacency matrix (1 = connected, 0 = not)
  # Returns: vector of component labels
  
  n_nodes <- nrow(adj_matrix)
  if (n_nodes == 0) return(integer(0))
  if (n_nodes == 1) return(1L)
  
  visited <- rep(FALSE, n_nodes)
  component <- rep(0L, n_nodes)
  current_comp <- 0L
  
  for (start in 1:n_nodes) {
    if (visited[start]) next
    
    current_comp <- current_comp + 1L
    # BFS from start
    queue <- start
    while (length(queue) > 0) {
      node <- queue[1]
      queue <- queue[-1]
      
      if (visited[node]) next
      visited[node] <- TRUE
      component[node] <- current_comp
      
      # Add unvisited neighbors
      neighbors <- which(adj_matrix[node, ] == 1 & !visited)
      queue <- c(queue, neighbors)
    }
  }
  
  return(component)
}

# Track clusters at each alpha level
cluster_tree_data <- list()

cat("Building conformal cluster tree...\n")
for (alpha_i in alpha_grid) {
  # Use consistent key formatting (sprintf %.2f) for reliable lookup later
  alpha_key <- sprintf("%.2f", alpha_i)
  
  # Get curves in C_hat at this alpha
  idx_alpha <- C_hat_indices(alpha_i, p_hat, n)
  n_alpha <- length(idx_alpha)
  
  if (n_alpha == 0) {
    cluster_tree_data[[alpha_key]] <- list(
      alpha = alpha_i,
      n_curves = 0,
      n_clusters = 0,
      cluster_sizes = integer(0),
      curve_indices = integer(0),
      cluster_labels = integer(0)
    )
    next
  }
  
  if (n_alpha == 1) {
    cluster_tree_data[[alpha_key]] <- list(
      alpha = alpha_i,
      n_curves = 1,
      n_clusters = 1,
      cluster_sizes = 1L,
      curve_indices = idx_alpha,
      cluster_labels = 1L
    )
    next
  }
  
  # Build adjacency matrix for G_{alpha, epsilon}
  D_sub <- D_matrix[idx_alpha, idx_alpha]
  adj <- (D_sub <= epsilon_selected) * 1L
  diag(adj) <- 0  # No self-loops
  
  # Find connected components
  comp_labels <- find_connected_components(adj)
  n_clusters <- max(comp_labels)
  cluster_sizes <- table(comp_labels)
  
  cluster_tree_data[[alpha_key]] <- list(
    alpha = alpha_i,
    n_curves = n_alpha,
    n_clusters = n_clusters,
    cluster_sizes = as.integer(cluster_sizes),
    curve_indices = idx_alpha,
    cluster_labels = comp_labels
  )
}
cat("Cluster tree built.\n")

#'
#' ### Number of Clusters vs Alpha
#'
#' The following plots show how the cluster structure changes as we vary $\alpha$:
#'
#' - **Left panel:** Number of distinct clusters — jumps indicate structural changes
#' - **Right panel:** Number of curves included — decreases monotonically with $\alpha$
#'
#' **What to look for:** Stable regions (flat segments) indicate robust cluster structure.
#' Rapid changes suggest the data has meaningful sub-populations that emerge or merge.

#+ sec4-clusters-vs-alpha, fig.width=10, fig.height=5
n_clusters_vec <- sapply(cluster_tree_data, function(x) x$n_clusters)
n_curves_vec <- sapply(cluster_tree_data, function(x) x$n_curves)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

plot(alpha_grid, n_clusters_vec, type = "b", pch = 19, col = "darkblue",
     xlab = expression(alpha), ylab = "Number of clusters",
     main = expression("Clusters in " * hat(C)[n*","*alpha] * " vs " * alpha))

plot(alpha_grid, n_curves_vec, type = "b", pch = 19, col = "forestgreen",
     xlab = expression(alpha), ylab = "Number of curves",
     main = expression("|" * hat(C)[n*","*alpha] * "| vs " * alpha))

#+ save-plot-clusters-alpha, include=FALSE
png(file.path(output_dir, "10_clusters_vs_alpha.png"), width = 900, height = 400)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(alpha_grid, n_clusters_vec, type = "b", pch = 19, col = "darkblue",
     xlab = expression(alpha), ylab = "Number of clusters",
     main = expression("Clusters in " * hat(C)[n*","*alpha] * " vs " * alpha))
plot(alpha_grid, n_curves_vec, type = "b", pch = 19, col = "forestgreen",
     xlab = expression(alpha), ylab = "Number of curves",
     main = expression("|" * hat(C)[n*","*alpha] * "| vs " * alpha))
dev.off()

#'
#' ### Conformal Cluster Tree Visualization
#'
#' **Reading the tree plot:** This is a "horizontal slice" view of the cluster tree:
#'
#' - **Y-axis:** The $\alpha$ level (higher = more exclusive)
#' - **X-axis:** Cluster index (just for separation — position has no meaning)
#' - **Point size:** Proportional to cluster size (log scale)
#' - **Colors:** Distinguish different clusters at each level
#' - **Annotations:** Show (n=curves, k=clusters) at each sampled $\alpha$
#'
#' **Interpretation:** As you scan from bottom (low $\alpha$) to top (high $\alpha$):
#'
#' - Large points shrink as outliers are removed
#' - A single cluster may split into multiple smaller ones
#' - Eventually only the most central, typical curves remain
#'
#' The tree shows how clusters split as $\alpha$ increases. At low $\alpha$ (bottom),
#' all curves are in one component. As $\alpha$ increases, curves with low pseudo-density
#' are removed, potentially causing the graph to disconnect into multiple clusters.

#+ sec4-tree-visualization, fig.width=12, fig.height=8

# Build a simplified tree structure
# Track parent-child relationships by set inclusion

# Find alpha levels where number of clusters changes
cluster_change_alphas <- alpha_grid[1]
for (i in 2:length(alpha_grid)) {
  if (n_clusters_vec[i] != n_clusters_vec[i-1]) {
    cluster_change_alphas <- c(cluster_change_alphas, alpha_grid[i])
  }
}

par(mar = c(4, 4, 3, 8))

# Create a dendrogram-like plot
plot(NULL, xlim = c(0, max(3, max(n_clusters_vec) + 1)), ylim = c(0, 1),
     xlab = "Cluster index", ylab = expression(alpha),
     main = sprintf("Conformal Cluster Tree (h=%.2f, ε=%.2f)", h_selected, epsilon_selected),
     xaxt = "n")

# Plot horizontal lines at each alpha level with cluster counts
alpha_sample <- seq(0.1, 0.9, by = 0.1)
for (a in alpha_sample) {
  a_str <- sprintf("%.2f", a)
  if (a_str %in% names(cluster_tree_data)) {
    info <- cluster_tree_data[[a_str]]
    if (info$n_clusters > 0) {
      # Draw cluster boxes
      x_positions <- seq(0.5, info$n_clusters - 0.5, length.out = info$n_clusters)
      for (k in 1:info$n_clusters) {
        points(x_positions[k], a, pch = 15, cex = 1.5 + log(info$cluster_sizes[k])/2,
               col = proto_colors[(k-1) %% length(proto_colors) + 1])
      }
      text(max(x_positions) + 0.5, a, sprintf("n=%d, k=%d", info$n_curves, info$n_clusters),
           pos = 4, cex = 0.7)
    }
  }
}

# Add annotation
mtext(expression("Larger points = larger clusters; Color distinguishes components"), 
      side = 1, line = 2.5, cex = 0.8)

#+ save-plot-tree, include=FALSE
png(file.path(output_dir, "11_conformal_cluster_tree.png"), width = 1000, height = 700)
par(mar = c(4, 4, 3, 8))
plot(NULL, xlim = c(0, max(3, max(n_clusters_vec) + 1)), ylim = c(0, 1),
     xlab = "Cluster index", ylab = expression(alpha),
     main = sprintf("Conformal Cluster Tree (h=%.2f, ε=%.2f)", h_selected, epsilon_selected),
     xaxt = "n")
alpha_sample <- seq(0.1, 0.9, by = 0.1)
for (a in alpha_sample) {
  a_str <- sprintf("%.2f", a)
  if (a_str %in% names(cluster_tree_data)) {
    info <- cluster_tree_data[[a_str]]
    if (info$n_clusters > 0) {
      x_positions <- seq(0.5, info$n_clusters - 0.5, length.out = info$n_clusters)
      for (k in 1:info$n_clusters) {
        points(x_positions[k], a, pch = 15, cex = 1.5 + log(info$cluster_sizes[k])/2,
               col = proto_colors[(k-1) %% length(proto_colors) + 1])
      }
      text(max(x_positions) + 0.5, a, sprintf("n=%d, k=%d", info$n_curves, info$n_clusters),
           pos = 4, cex = 0.7)
    }
  }
}
mtext(expression("Larger points = larger clusters; Color distinguishes components"), 
      side = 1, line = 2.5, cex = 0.8)
dev.off()

#'
#' ### Alternative: Two Trees Under Different Parameters (Paper Figure 5 style)
#'
#' To show sensitivity to tuning parameters, we build a second tree with different $h$ and $\epsilon$.

#+ sec4-alternative-tree, fig.width=12, fig.height=6

# Alternative parameters
h_alt <- quantile(offdiag_dists, 0.25)  # Smaller bandwidth
epsilon_alt <- h_alt / 2

# Recompute pseudo-density with alternative h
p_hat_alt <- compute_pseudo_density(D_matrix, h_alt)

# Build alternative cluster tree (use consistent key formatting)
cluster_tree_alt <- list()
for (alpha_i in alpha_grid) {
  alpha_key <- sprintf("%.2f", alpha_i)  # Consistent with main tree
  idx_alpha <- C_hat_indices(alpha_i, p_hat_alt, n)
  n_alpha <- length(idx_alpha)
  
  if (n_alpha <= 1) {
    cluster_tree_alt[[alpha_key]] <- list(n_clusters = max(1, n_alpha), n_curves = n_alpha)
    next
  }
  
  D_sub <- D_matrix[idx_alpha, idx_alpha]
  adj <- (D_sub <= epsilon_alt) * 1L
  diag(adj) <- 0
  comp_labels <- find_connected_components(adj)
  
  cluster_tree_alt[[alpha_key]] <- list(
    n_clusters = max(comp_labels),
    n_curves = n_alpha
  )
}

n_clusters_alt <- sapply(cluster_tree_alt, function(x) x$n_clusters)

# Side-by-side comparison
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

plot(alpha_grid, n_clusters_vec, type = "b", pch = 19, col = "darkblue",
     xlab = expression(alpha), ylab = "Number of clusters",
     main = sprintf("(a) h=%.1f, ε=%.1f", h_selected, epsilon_selected),
     ylim = c(0, max(c(n_clusters_vec, n_clusters_alt)) + 1))

plot(alpha_grid, n_clusters_alt, type = "b", pch = 19, col = "darkred",
     xlab = expression(alpha), ylab = "Number of clusters",
     main = sprintf("(b) h=%.1f, ε=%.1f (smaller)", h_alt, epsilon_alt),
     ylim = c(0, max(c(n_clusters_vec, n_clusters_alt)) + 1))

#+ save-plot-two-trees, include=FALSE
png(file.path(output_dir, "12_two_cluster_trees.png"), width = 900, height = 400)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(alpha_grid, n_clusters_vec, type = "b", pch = 19, col = "darkblue",
     xlab = expression(alpha), ylab = "Number of clusters",
     main = sprintf("(a) h=%.1f, ε=%.1f", h_selected, epsilon_selected),
     ylim = c(0, max(c(n_clusters_vec, n_clusters_alt)) + 1))
plot(alpha_grid, n_clusters_alt, type = "b", pch = 19, col = "darkred",
     xlab = expression(alpha), ylab = "Number of clusters",
     main = sprintf("(b) h=%.1f, ε=%.1f (smaller)", h_alt, epsilon_alt),
     ylim = c(0, max(c(n_clusters_vec, n_clusters_alt)) + 1))
dev.off()

#' **Observation:** The choice of $h$ and $\epsilon$ affects the tree structure:
#' - Smaller $h$ creates more localized pseudo-density (more modes, potentially more clusters)
#' - Smaller $\epsilon$ requires curves to be closer to be connected (more clusters)
#'
#' The tree is **nested** by construction: at any two levels $\alpha_1 < \alpha_2$, 
#' $\hat{C}_{n,\alpha_2} \subseteq \hat{C}_{n,\alpha_1}$, so clusters at higher $\alpha$
#' are subsets of clusters at lower $\alpha$.

#'
#' ## 4.7 Section 4 Summary
#'
#' We have implemented all key components from Section 4 of the paper:
#'
#' | Component | Paper Reference | Status |
#' |-----------|-----------------|--------|
#' | Pseudo-density estimator | Eq. (10) | Implemented |
#' | Conformal set approximation $C^+_{n,\alpha}$ | Eq. (11) | Implemented |
#' | Sample approximation $\hat{C}_{n,\alpha}$ | Below Eq. (11) | Implemented |
#' | Anomalies / Median / High-density | Figure 4 | Implemented |
#' | Mean-shift prototypes | Section 4, Cheng (1995) | Implemented |
#' | Conformal cluster tree | Section 4, $G_{\alpha,\epsilon}$ | Implemented |
#' | Bandwidth selection | Figure 5 caption | Implemented |
#' | Epsilon selection | Figure 5 caption | Implemented |

#+ save-results-sec4, include=FALSE
## Save Section 4 results (after all Section 4 objects are created)
section4_results <- list(
  distance_matrix = D_matrix,
  pseudo_density = list(
    h_selected = h_selected,
    h_candidates = h_candidates,
    h_variances = h_variances,
    p_hat = p_hat
  ),
  prototypes = lapply(unique_prototypes, function(proto) proto$prototype),
  cluster_tree = cluster_tree_data,
  parameters = list(
    epsilon_selected = epsilon_selected,
    h_alt = h_alt,
    epsilon_alt = epsilon_alt
  )
)
saveRDS(section4_results, file.path(output_dir, "section4_results.rds"))

## Save combined summary (now that all objects exist)
summary_df <- data.frame(
  Parameter = c("n (total curves)", "n1 (training)", "n2 (calibration)",
                "M (grid points)", "p (basis dimension)", "K (GMM components)",
                "alpha (miscoverage)", "lambda (threshold)", "Calibration inclusion rate",
                "Out-of-sample coverage", "h (Section 4 bandwidth)", "epsilon (Section 4)",
                "Number of prototypes"),
  Value = c(n, n1, n2, M, p, K, alpha, sprintf("%.6e", lambda), 
            sprintf("%.1f%%", 100 * inclusion_rate),
            sprintf("%.1f%%", 100 * mean_coverage),
            sprintf("%.2f", h_selected),
            sprintf("%.2f", epsilon_selected),
            n_prototypes)
)
write.csv(summary_df, file.path(output_dir, "summary.csv"), row.names = FALSE)

#'
#' # Discussion and Conclusions
#' 
#' ## Summary of Results
#' 
#' We successfully implemented **conformal prediction methods for functional data** from 
#' both Section 3.1 (Gaussian Mixture bands) and Section 4 (pseudo-density methods) of 
#' Lei, Rinaldo, and Wasserman.
#' 
#' **Key numerical results:**
#' 
#' | Metric | Value |
#' |--------|-------|
#' | Total curves analyzed | `r n` |
#' | Training / Calibration split | `r n1` / `r n2` |
#' | Basis dimension ($p$) | `r p` |
#' | GMM components ($K$) | `r K` |
#' | Target coverage $(1-\alpha)$ | `r 100*(1-alpha)`% |
#' | Empirical out-of-sample coverage | `r sprintf("%.1f%%", 100 * mean_coverage)` |
#' | Conformal threshold ($\lambda$) | `r sprintf("%.4e", lambda)` |
#' | Selected bandwidth (Section 4) | `r sprintf("%.2f", h_selected)` |
#' | Mean-shift prototypes found | `r n_prototypes` |
#' 
#' ## Interpretation of Results
#' 
#' ### Section 3: Prediction Bands
#' 
#' The conformal prediction band represents a **set-valued prediction** for future 
#' acceleration curves. With approximately `r 100*(1-alpha)`% probability, a new curve 
#' drawn from the same distribution will fall entirely within this band.
#' 
#' **Key observations:**
#' 
#' 1. **Band structure**: The band $B_n(t) = \bigcup_k [\ell_k(t), u_k(t)]$ can have 
#'    disconnected slices when GMM components are well-separated. Our diagnostics show
#'    that slices vary between `r min(n_intervals)` and `r max(n_intervals)` intervals.
#'    
#' 2. **Multi-component structure**: With $K=3$ components (matching our 3 activity types), 
#'    the band can accommodate the different acceleration patterns of standing, walking, 
#'    and fast walking.
#'    
#' 3. **Coverage guarantee**: Conformal prediction provides a finite-sample coverage guarantee 
#'    of at least $(1-\alpha)$ under the exchangeability assumption. Non-overlapping windows
#'    reduce temporal dependence, making exchangeability a reasonable approximation. Our 
#'    empirical coverage of `r sprintf("%.1f%%", 100 * mean_coverage)` is close to the 
#'    target `r 100*(1-alpha)`%.
#'
#' ### Section 4: Pseudo-Density Methods
#'
#' The pseudo-density approach provides complementary insights:
#'
#' 1. **Anomaly detection**: The bottom 5% of curves by pseudo-density clearly identify
#'    unusual acceleration patterns that deviate from typical behavior.
#'
#' 2. **Prototypes**: Mean-shift finds `r n_prototypes` distinct modes, representing
#'    characteristic curve shapes in the data. These correspond roughly to our activity types.
#'
#' 3. **Cluster tree**: As $\alpha$ increases, the conformal sets shrink and may split
#'    into disconnected components, revealing the multi-modal structure of the data.
#' 
#' ## What We Implemented vs. Paper (Full Checklist)
#' 
#' ### Section 3 (Gaussian Mixture Bands)
#' 
#' | Paper Reference | Description | Implemented | Notes |
#' |-----------------|-------------|-------------|-------|
#' | Algorithm 1 | Inductive conformal predictor | Yes | Split-sample approach |
#' | Algorithm 2 | Functional conformal bands | Yes | Cosine basis projection |
#' | Eq.(6) | Outer bound via ellipsoids: $T_n \subseteq \bigcup_k T_{n,k}$ | Yes | Threshold $\tau_k = \lambda/(K\pi_k)$ |
#' | Projection lemma | $u_k(t), \ell_k(t)$ closed-form | Yes | From course notes |
#' | Band construction | $B_n(t) = \bigcup_k [\ell_k(t), u_k(t)]$ | Yes | With interval merging |
#' 
#' ### Section 4 (Pseudo-Density Methods)
#' 
#' | Paper Reference | Description | Implemented | Notes |
#' |-----------------|-------------|-------------|-------|
#' | Eq.(10) | Pseudo-density: $\hat{p}_h(u) = \frac{1}{n}\sum_i K(d(u,X_i)/h)$ | Yes | Gaussian kernel |
#' | Eq.(11) | Conformal approximation: $C^+_{n,\alpha}$ | Yes | Lemma 4.1 coverage guarantee |
#' | Below Eq.(11) | Sample approximation: $\hat{C}_{n,\alpha}$ | Yes | Intersection with data |
#' | Figure 4 | Anomalies, median, high-density subsets | Yes | Based on pseudo-density ranks |
#' | Section 4 text | Mean-shift prototypes (Cheng 1995) | Yes | Weighted mean iteration |
#' | Section 4 text | Conformal cluster tree: $G_{\alpha,\epsilon}$ | Yes | Connected components |
#' | Figure 5 caption | Bandwidth selection: max variance of $\hat{p}_h$ | Yes | Grid search |
#' | Figure 5 caption | Epsilon selection: based on NN distances | Yes | $0.5 \times \max$ NN dist |
#'
#' ### Key Mathematical Formulas Verified
#' 
#' | Formula | Expression | Status |
#' |---------|------------|--------|
#' | $L_2$ distance | $d(f,g) = [\int(f-g)^2]^{1/2}$ | Trapezoidal rule |
#' | Gaussian kernel | $K(z) = \exp(-z^2/2)$, $K(0) = 1$ | Satisfies $K(z) \le K(0)$ |
#' | Ellipsoid radius | $r^2 = -2\log(\tau) - p\log(2\pi) - \log|\Sigma|$ | Derived from density threshold |
#' | Mean-shift update | $u^{(s+1)} = \sum_i w_i X_i / \sum_i w_i$ | Convergence verified |
#' | Graph connectivity | Edge if $d(X_i, X_j) \le \epsilon$ | BFS for components |
#' 
#' ## Limitations and Future Work
#' 
#' 1. **Outer bound approximation**: We use the union of ellipsoids (Eq. 6), which is an 
#'    outer bound to the exact level set. This may result in slightly conservative bands.
#'    The refinement in Eq.(7) could be implemented for tighter bounds.
#'    
#' 2. **Fixed $K$ and $p$**: We chose $K=3$ to match known activity types and $p=5$ for 
#'    computational tractability. Cross-validation could optimize these choices.
#'    
#' 3. **Exchangeability assumption**: Conformal prediction requires exchangeability of the
#'    calibration and test data. Our data collection uses non-overlapping 10-second windows
#'    from a few longer recordings, which reduces temporal dependence compared to overlapping
#'    windows. However, windows from the same recording session may still exhibit residual
#'    correlation (e.g., similar gait patterns within a session). Our empirical coverage
#'    results (`r sprintf("%.1f%%", 100 * mean_coverage)` vs target `r 100*(1-alpha)`%) 
#'    suggest the exchangeability approximation is reasonable, but the strict finite-sample
#'    coverage guarantee assumes perfect exchangeability. For stronger guarantees, one could
#'    randomly sample windows across different sessions or use session-level splitting.
#'
#' 4. **Tuning parameter sensitivity**: The cluster tree structure depends on $h$ and $\epsilon$.
#'    We demonstrated this sensitivity but did not fully optimize these parameters.
#'
#' 5. **Analytic distance**: The paper mentions an "analytic distance" (weighted by basis 
#'    coefficients) that could produce different results. We used the standard $L_2$ distance.
#'
#' ## Implementation Notes and Debugging
#'
#' This section documents practical implementation details and decisions made during development:
#'
#' **GMM fitting:**
#' - The `Mclust` function with `modelNames = "VVV"` (unconstrained covariance) sometimes fails
#'   on small training sets. We use a fallback to default model selection if VVV fails.
#' - Covariance matrices occasionally become near-singular. We add a geometric ridge (starting
#'   at `1e-8`, increasing by 10×) until Cholesky decomposition succeeds.
#'
#' **Basis choice:**
#' - We chose $p=5$ basis functions as a balance: $p=3$ loses too much signal detail, $p=10$
#'   adds noise and increases computational cost. The first 5 cosine basis functions capture
#'   the dominant low-frequency patterns in acceleration signals.
#'
#' **Windowing:**
#' - Non-overlapping 10-second windows reduce temporal dependence compared to overlapping windows.
#'   This improves exchangeability approximation but reduces the number of curves from each
#'   recording session.
#'
#' **Mean-shift initialization:**
#' - Initializing only from top-density curves (original approach) biased toward finding a single
#'   dominant mode. We switched to stratified initialization across activities to find multiple
#'   prototypes corresponding to different activity types.
#'
#' **Deduplication threshold:**
#' - The original threshold `h/2` was too loose, merging distinct prototypes. We tightened to
#'   `h/4` to preserve separate modes while still removing duplicates from nearby starting points.
#'
#' **Observed patterns:**
#' - Walking and Fast Walking curves show substantial overlap in coefficient space — they share
#'   similar oscillatory structure, differing mainly in amplitude. Standing curves are well-separated.
#' - The GMM components align roughly with activity types, but not perfectly — some walking curves
#'   fall into the Fast Walking component and vice versa.
#'
#' ## What We Tried and Rejected
#'
#' **FPCA basis:** We initially considered implementing empirical eigenfunctions (FPCA) as in the
#' paper, but chose fixed cosine basis for simplicity and computational stability. FPCA requires
#' careful handling of small eigenvalues and can be sensitive to training set composition.
#'
#' **Overlapping windows:** We tested overlapping windows (5-second step size) but found they
#' introduced strong temporal dependence that violated exchangeability. Non-overlapping windows
#' trade off sample size for better statistical properties.
#'
#' ## Conclusions
#' 
#' This implementation demonstrates that conformal prediction can be effectively applied to 
#' functional data, providing meaningful prediction bands with guaranteed coverage. The 
#' accelerometer dataset shows clear activity-dependent structure that both the GMM (Section 3)
#' and pseudo-density (Section 4) methods capture effectively.
#'
#' **Section 3** provides explicit prediction bands that can be visualized and used for 
#' anomaly detection by checking if new curves fall outside the band.
#'
#' **Section 4** provides complementary tools for exploratory analysis: identifying outliers,
#' finding representative prototypes, and visualizing the hierarchical structure via cluster trees.
#' 
#' The methodology is general and could be applied to other functional datasets such as 
#' GPS trajectories, physiological signals, or economic time series.
#' 
#' # Appendix: Session Info
#' 
#' R session information for reproducibility.

#+ session-info, class.source = 'fold-hide'
sessionInfo()
#' ---
#' 
#' *Report generated on `r Sys.Date()` using R version `r getRversion()`*

