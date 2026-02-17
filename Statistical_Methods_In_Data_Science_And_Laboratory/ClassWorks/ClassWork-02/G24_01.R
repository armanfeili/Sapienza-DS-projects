# Classwork-02
# Team: G24
# Team members: Arman Feili, Sohrab Seyyedi Parsa, Milad Torabi

# Q1)

options(stringsAsFactors = FALSE)


# 0) Install + load packages

need <- c("gsl", "copula")

for (p in need) {
  if (!requireNamespace(p, quietly = TRUE)) {
    try(install.packages(p, type = "binary"), silent = TRUE)
    if (!requireNamespace(p, quietly = TRUE)) {
      try(install.packages(p, type = "source"), silent = TRUE)
    }
  }
}

if (!requireNamespace("gsl", quietly = TRUE)) {
  stop("Missing 'gsl'. On macOS we may need: brew install gsl (then reinstall gsl from source).")
}
if (!requireNamespace("copula", quietly = TRUE)) {
  stop("Missing 'copula'. Install it after 'gsl' is working.")
}

# In knitr/spin we want the package attached, so functions are visible.
suppressPackageStartupMessages(suppressWarnings(library(copula)))
cat("Engine selected: copula\n")


# 1) Choose ANY TWO models here

# Supported IDs: "gaussian", "t", "clayton", "gumbel", "frank", "joe"
MODEL_1 <- list(id = "gaussian", name = "Gaussian",  t_df_start = 6)
MODEL_2 <- list(id = "t",        name = "Student-t", t_df_start = 6)

# Starting dependence value for ML (mainly for Gaussian/t)
START_RHO <- 0.2


# 2) Load data

crsp <- read.csv("CRSPday.csv")

# If year/month/day exist, we build a Date column (not needed for fitting, but nice to have)
if (all(c("year", "month", "day") %in% names(crsp))) {
  crsp$date <- as.Date(sprintf("%04d-%02d-%02d", crsp$year, crsp$month, crsp$day))
}

needed_cols <- c("ge", "ibm", "mobil", "crsp")
miss <- setdiff(needed_cols, names(crsp))
if (length(miss) > 0) stop("Missing columns in CRSPday.csv: ", paste(miss, collapse = ", "))


# 3) Pseudo-observations (ranks)

# We do this so each margin becomes approximately Uniform(0,1).
pobs_simple <- function(z) {
  n <- length(z)
  r <- rank(z, ties.method = "average", na.last = "keep")
  r / (n + 1)
}

# Convert a pair (x,y) into (u,v) in (0,1)^2
# If there are many ties, we jitter a tiny bit to avoid flat ranks.
to_uv <- function(x, y, jitter_if_ties = TRUE, jitter_scale = 1e-10, seed = 123) {
  d <- data.frame(x = x, y = y)
  d <- d[complete.cases(d), , drop = FALSE]
  
  if (nrow(d) < 30) stop("Too few complete observations: n = ", nrow(d))
  if (sd(d$x) == 0 || sd(d$y) == 0) stop("One series is constant (sd=0), copula fit is meaningless.")
  
  if (jitter_if_ties) {
    tie_rate_x <- 1 - length(unique(d$x)) / length(d$x)
    tie_rate_y <- 1 - length(unique(d$y)) / length(d$y)
    set.seed(seed)
    if (tie_rate_x > 0.05) d$x <- d$x + rnorm(nrow(d), 0, jitter_scale)
    if (tie_rate_y > 0.05) d$y <- d$y + rnorm(nrow(d), 0, jitter_scale)
  }
  
  uv <- cbind(pobs_simple(d$x), pobs_simple(d$y))
  colnames(uv) <- c("u", "v")
  uv
}


# 4) Build copula objects

make_cop <- function(model, start_rho = 0.2) {
  id <- tolower(model$id)
  
  if (id == "gaussian") {
    return(normalCopula(param = start_rho, dim = 2, dispstr = "un"))
  }
  if (id == "t") {
    df0 <- if (!is.null(model$t_df_start)) model$t_df_start else 6
    return(tCopula(param = start_rho, dim = 2, df = df0, df.fixed = FALSE, dispstr = "un"))
  }
  if (id == "clayton") {
    return(claytonCopula(param = 1, dim = 2))
  }
  if (id == "gumbel") {
    return(gumbelCopula(param = 1.2, dim = 2))
  }
  if (id == "frank") {
    return(frankCopula(param = 2, dim = 2))
  }
  if (id == "joe") {
    return(joeCopula(param = 1.2, dim = 2))
  }
  
  stop("Unknown model id: ", model$id, " (use gaussian, t, clayton, gumbel, frank, joe)")
}


# 5) Fit two models and compare AIC/BIC

# As can be seen, AIC/BIC come from logLik and number of parameters:
#   AIC = -2*logLik + 2*k
#   BIC = -2*logLik + log(n)*k
compare_two <- function(uv, model1 = MODEL_1, model2 = MODEL_2, start_rho = START_RHO) {
  n <- nrow(uv)
  
  cop1 <- make_cop(model1, start_rho)
  fit1 <- tryCatch(
    fitCopula(cop1, data = uv, method = "ml"),
    error = function(e) stop("Fit failed for ", model1$name, ": ", e$message)
  )
  
  cop2 <- make_cop(model2, start_rho)
  fit2 <- tryCatch(
    fitCopula(cop2, data = uv, method = "ml"),
    error = function(e) stop("Fit failed for ", model2$name, ": ", e$message)
  )
  
  ll1 <- as.numeric(logLik(fit1)); k1 <- length(coef(fit1))
  ll2 <- as.numeric(logLik(fit2)); k2 <- length(coef(fit2))
  
  out <- data.frame(
    model = c(model1$name, model2$name),
    logLik = c(ll1, ll2),
    k = c(k1, k2),
    AIC = c(-2 * ll1 + 2 * k1, -2 * ll2 + 2 * k2),
    BIC = c(-2 * ll1 + log(n) * k1, -2 * ll2 + log(n) * k2)
  )
  
  list(
    summary = out,
    winner_AIC = out$model[which.min(out$AIC)],
    winner_BIC = out$model[which.min(out$BIC)],
    fit1 = fit1,
    fit2 = fit2
  )
}


# 6) The 6 pairs (no loops) — compute uv inline


# (1) GE–IBM
res_ge_ibm <- compare_two(to_uv(crsp$ge, crsp$ibm), MODEL_1, MODEL_2)
cat("\n=== GE vs IBM ===\n"); print(res_ge_ibm$summary)
cat("Winner by AIC:", res_ge_ibm$winner_AIC, "\n")
cat("Winner by BIC:", res_ge_ibm$winner_BIC, "\n")

# (2) GE–MOBIL
res_ge_mobil <- compare_two(to_uv(crsp$ge, crsp$mobil), MODEL_1, MODEL_2)
cat("\n=== GE vs MOBIL ===\n"); print(res_ge_mobil$summary)
cat("Winner by AIC:", res_ge_mobil$winner_AIC, "\n")
cat("Winner by BIC:", res_ge_mobil$winner_BIC, "\n")

# (3) GE–CRSP
res_ge_crsp <- compare_two(to_uv(crsp$ge, crsp$crsp), MODEL_1, MODEL_2)
cat("\n=== GE vs CRSP ===\n"); print(res_ge_crsp$summary)
cat("Winner by AIC:", res_ge_crsp$winner_AIC, "\n")
cat("Winner by BIC:", res_ge_crsp$winner_BIC, "\n")

# (4) IBM–MOBIL
res_ibm_mobil <- compare_two(to_uv(crsp$ibm, crsp$mobil), MODEL_1, MODEL_2)
cat("\n=== IBM vs MOBIL ===\n"); print(res_ibm_mobil$summary)
cat("Winner by AIC:", res_ibm_mobil$winner_AIC, "\n")
cat("Winner by BIC:", res_ibm_mobil$winner_BIC, "\n")

# (5) IBM–CRSP
res_ibm_crsp <- compare_two(to_uv(crsp$ibm, crsp$crsp), MODEL_1, MODEL_2)
cat("\n=== IBM vs CRSP ===\n"); print(res_ibm_crsp$summary)
cat("Winner by AIC:", res_ibm_crsp$winner_AIC, "\n")
cat("Winner by BIC:", res_ibm_crsp$winner_BIC, "\n")

# (6) MOBIL–CRSP
res_mobil_crsp <- compare_two(to_uv(crsp$mobil, crsp$crsp), MODEL_1, MODEL_2)
cat("\n=== MOBIL vs CRSP ===\n"); print(res_mobil_crsp$summary)
cat("Winner by AIC:", res_mobil_crsp$winner_AIC, "\n")
cat("Winner by BIC:", res_mobil_crsp$winner_BIC, "\n")





# ============================================================
# Q2) Robust procedures: Median-of-Means (MoM) + MoM histogram
# (simple / student-style, but same correct logic)
# ============================================================

options(stringsAsFactors = FALSE)
set.seed(123)


# 1) Small helper functions


# We randomly shuffle indices 1..n and then split them into K (almost) equal parts.
make_blocks <- function(n, K, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  n <- as.integer(n)
  K <- as.integer(K)
  
  if (n < 1) stop("n must be >= 1")
  if (K < 1) stop("K must be >= 1")
  K <- min(K, n)
  
  idx <- sample.int(n)                    # random permutation of indices
  grp <- rep(1:K, length.out = n)         # assigns indices to groups 1..K
  split(idx, grp)
}

# In the theory, K is often chosen as ceil(8 log(1/alpha)).
k_star <- function(alpha) {
  if (!is.finite(alpha) || alpha <= 0 || alpha >= 1) stop("alpha must be in (0,1)")
  ceiling(8 * log(1 / alpha))
}

# MoM mean: split into blocks, take block means, then take the median of those means.
mom_mean <- function(x, K, seed = 1) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 5) stop("Too few observations for MoM mean.")
  
  K <- as.integer(K)
  K <- max(1, min(K, floor(n / 2)))  # as can be seen, we force block size >= 2
  
  blocks <- make_blocks(n, K, seed = seed)
  block_means <- sapply(blocks, function(id) mean(x[id]))
  median(block_means)
}


# 2) Part A: mean estimation under outliers


# Pareto(xm, shape): X = xm / U^(1/shape), U~Unif(0,1)
rpareto <- function(n, xm = 1, shape = 2.5) {
  if (!is.finite(xm) || xm <= 0) stop("xm must be > 0")
  if (!is.finite(shape) || shape <= 0) stop("shape must be > 0")
  u <- runif(n)
  xm / (u^(1 / shape))
}

# We contaminate eps*n points by replacing them with big spikes (+/- outlier_value).
contaminate <- function(x, eps = 0.05, outlier_value = 50, seed = 99) {
  if (!is.finite(eps) || eps < 0 || eps >= 1) stop("eps must be in [0,1)")
  n <- length(x)
  m <- floor(eps * n)
  if (m == 0) return(x)
  
  set.seed(seed)
  id <- sample.int(n, m)
  x[id] <- sample(c(-1, 1), m, replace = TRUE) * outlier_value
  x
}

run_mean_experiment <- function(n = 500, alpha = 0.05, eps = 0.05, R = 200) {
  if (n < 10) stop("n must be >= 10")
  if (R < 1) stop("R must be >= 1")
  
  K <- k_star(alpha)
  
  # We compare the usual mean vs MoM mean for a few distributions.
  dists <- list(
    Normal_0_1 = list(sample = function(nn) rnorm(nn, 0, 1), mu = 0),
    t_df3      = list(sample = function(nn) rt(nn, df = 3),  mu = 0),
    Pareto_2.5 = list(
      sample = function(nn) rpareto(nn, xm = 1, shape = 2.5),
      mu     = 2.5 / (2.5 - 1) # mean of Pareto(xm=1, shape=2.5)
    )
  )
  
  out <- lapply(names(dists), function(name) {
    dd <- dists[[name]]
    err_mean <- numeric(R)
    err_mom  <- numeric(R)
    
    for (r in 1:R) {
      x <- dd$sample(n)
      x <- contaminate(x, eps = eps, outlier_value = 50, seed = 1000 + r)
      
      # We do this, we do that: compute both estimates and store absolute errors.
      err_mean[r] <- abs(mean(x) - dd$mu)
      err_mom[r]  <- abs(mom_mean(x, K = K, seed = 2000 + r) - dd$mu)
    }
    
    data.frame(
      distribution = name,
      n = n, eps = eps, alpha = alpha,
      K = max(1, min(as.integer(K), floor(n / 2))),
      mean_abs_error_mean   = mean(err_mean),
      mean_abs_error_MoM    = mean(err_mom),
      median_abs_error_mean = median(err_mean),
      median_abs_error_MoM  = median(err_mom)
    )
  })
  
  do.call(rbind, out)
}

cat("\n--- Q2A) MoM mean vs sample mean (with outliers) ---\n")
mean_results <- run_mean_experiment(n = 500, alpha = 0.05, eps = 0.05, R = 200)
print(mean_results, row.names = FALSE)


# 3) Part B: MoM histogram on [0,1] + L2 loss


# True density (so we can compute an L2 error against the truth)
f_true <- function(x) dbeta(x, 2, 5)

# We clip to avoid exact 0 and 1; this makes binning and findInterval stable.
clip01 <- function(x, eps = 1e-12) pmin(1 - eps, pmax(0 + eps, x))

# Contamination inside [0,1]: half goes near 0.99, half becomes uniform noise.
contaminate_unit <- function(x, eps = 0.10, seed = 7) {
  if (!is.finite(eps) || eps < 0 || eps >= 1) stop("eps must be in [0,1)")
  x <- clip01(x)
  
  n <- length(x)
  m <- floor(eps * n)
  if (m == 0) return(x)
  
  set.seed(seed)
  id <- sample.int(n, m)
  
  m1 <- floor(m / 2)
  if (m1 > 0) x[id[1:m1]] <- 0.99 + rnorm(m1, 0, 0.002)
  if (m - m1 > 0) x[id[(m1 + 1):m]] <- runif(m - m1)
  
  clip01(x)
}

# Standard histogram density from hist().
# As can be seen, hist() already returns densities, but we optionally rescale to integrate to 1.
hist_density <- function(x, breaks, rescale_to_1 = TRUE) {
  x <- clip01(x[is.finite(x)])
  
  h <- hist(x, breaks = breaks, plot = FALSE, right = TRUE, include.lowest = TRUE)
  dens <- h$density
  
  if (rescale_to_1) {
    widths <- diff(h$breaks)
    mass <- sum(dens * widths)
    if (mass > 0) dens <- dens / mass
  }
  
  list(density = dens, breaks = h$breaks)
}

# MoM histogram (robust):
# 1) split data into K blocks
# 2) for each block compute bin probabilities (counts / block_size)
# 3) take pointwise median across blocks
# 4) convert probabilities to density by dividing by bin width
mom_hist_density <- function(x, breaks, K, seed = 1, rescale_to_1 = TRUE) {
  x <- clip01(x[is.finite(x)])
  n <- length(x)
  if (n < 10) stop("Too few observations for MoM histogram.")
  
  K <- as.integer(K)
  K <- max(1, min(K, floor(n / 5)))  # practical: blocks not too tiny
  
  blocks <- make_blocks(n, K, seed = seed)
  
  J <- length(breaks) - 1
  widths <- diff(breaks)
  
  # p_mat is J x K: each column is the vector of bin probabilities for one block
  p_mat <- vapply(blocks, function(id) {
    xi <- x[id]
    hi <- hist(xi, breaks = breaks, plot = FALSE, right = TRUE, include.lowest = TRUE)
    hi$counts / length(xi)
  }, numeric(J))
  
  p_med <- apply(p_mat, 1, median)
  dens  <- p_med / widths
  
  # Median step can slightly break normalization, so we fix it if requested.
  if (rescale_to_1) {
    mass <- sum(dens * widths)  # same as sum(p_med)
    if (mass > 0) dens <- dens / mass
  }
  
  list(density = dens, breaks = breaks, K = K)
}

# Evaluate a piecewise-constant histogram at a grid of points
eval_hist_pc <- function(xgrid, breaks, dens) {
  xgrid <- clip01(xgrid)
  bin <- findInterval(xgrid, breaks, rightmost.closed = TRUE, all.inside = TRUE)
  dens[bin]
}

# Approximate L2 loss by a fine grid Riemann sum
L2_loss_on_grid <- function(fhat_fun, ftrue_fun, m = 5000, eps = 1e-12) {
  xg <- seq(0 + eps, 1 - eps, length.out = m)
  dx <- xg[2] - xg[1]
  sum((fhat_fun(xg) - ftrue_fun(xg))^2) * dx
}

run_hist_experiment <- function(n = 1000, h = 0.05, alpha = 0.05, eps = 0.10, R = 150) {
  breaks <- seq(0, 1, by = h)
  if (tail(breaks, 1) < 1) breaks <- c(breaks, 1)  # small guard for rounding
  
  K <- k_star(alpha)
  
  loss_std <- numeric(R)
  loss_mom <- numeric(R)
  
  for (r in 1:R) {
    x <- rbeta(n, 2, 5)
    x <- contaminate_unit(x, eps = eps, seed = 3000 + r)
    
    std <- hist_density(x, breaks)
    mh  <- mom_hist_density(x, breaks, K = K, seed = 4000 + r)
    
    f_std <- function(t) eval_hist_pc(t, std$breaks, std$density)
    f_mom <- function(t) eval_hist_pc(t, mh$breaks,  mh$density)
    
    loss_std[r] <- L2_loss_on_grid(f_std, f_true)
    loss_mom[r] <- L2_loss_on_grid(f_mom, f_true)
  }
  
  data.frame(
    n = n, h = h, eps = eps, alpha = alpha,
    K = max(1, min(as.integer(K), floor(n / 5))),
    mean_L2_standard   = mean(loss_std),
    mean_L2_MoM        = mean(loss_mom),
    median_L2_standard = median(loss_std),
    median_L2_MoM      = median(loss_mom)
  )
}

cat("\n--- Q2B) Standard histogram vs MoM histogram (L2 loss, with contamination) ---\n")
hist_results <- run_hist_experiment(n = 1000, h = 0.05, alpha = 0.05, eps = 0.10, R = 150)
print(hist_results, row.names = FALSE)


# 4) One example plot


plot_one_example <- function(n = 1000, h = 0.05, alpha = 0.05, eps = 0.10) {
  breaks <- seq(0, 1, by = h)
  if (tail(breaks, 1) < 1) breaks <- c(breaks, 1)
  
  K <- k_star(alpha)
  
  x <- rbeta(n, 2, 5)
  x <- contaminate_unit(x, eps = eps, seed = 777)
  
  std <- hist_density(x, breaks)
  mh  <- mom_hist_density(x, breaks, K = K, seed = 888)
  
  mids <- (breaks[-1] + breaks[-length(breaks)]) / 2
  
  plot(mids, std$density, type = "h", lwd = 2,
       main = "Histogram density: standard vs MoM–H",
       xlab = "x", ylab = "density")
  lines(mids, mh$density, type = "h", lwd = 2)
  curve(f_true(x), from = 0, to = 1, add = TRUE, lwd = 2)
  
  legend("topright",
         legend = c("standard hist", "MoM–H", "true density"),
         lwd = 2, bty = "n")
}

plot_one_example()




# BONUS Question)

# We Pick the histogram bin B that contains x0 (width h).
# Then count C = #{Xi in B}. Then C ~ Bin(n, p_B), where p_B = P(X in B).
# Histogram at x0 is: fhat(x0) = (C/n) / h.
# So we build a (1-alpha) CI for p_B (exact or approx), then divide by h.
# If x0 is exactly on a bin edge, there is a silly ambiguity.
# As can be seen, hist() uses right-closed bins (a, b] (with include.lowest for the first bin),
# so we nudge x0 a tiny bit to the left if it sits on an edge.

# 1) Breaks builder: fixed bin width h
make_fixed_breaks <- function(x, x0, h) {
  x <- x[is.finite(x)]
  if (length(x) < 2) stop("Need at least 2 finite observations.")
  if (!is.finite(x0)) stop("x0 must be finite.")
  if (!is.finite(h) || h <= 0) stop("h must be > 0.")
  
  # We align breaks on a simple grid of step h.
  lo <- floor(min(c(x, x0)) / h) * h
  hi <- ceiling(max(c(x, x0)) / h) * h
  if (hi <= lo) hi <- lo + h
  
  br <- seq(lo, hi, by = h)
  br <- sort(unique(br))
  if (length(br) < 2) stop("Failed to build valid breaks.")
  br
}

# 2) Tiny nudge: avoid "x0 exactly equals a break"
nudge_x0_left_if_on_edge <- function(x0, breaks, h) {
  tol <- max(1e-12, 10 * .Machine$double.eps * max(1, abs(h)))
  
  # If x0 is on an edge (up to tol), we shift it slightly left.
  # We do this, we do that, and now x0 is safely inside one bin.
  if (any(abs(x0 - breaks) <= tol)) x0 <- x0 - tol
  
  x0
}

# 3) Main: pointwise CI for f(x0)
hist_ci_density_point <- function(x, x0, h, alpha = 0.05,
                                  method = c("exact", "prop"),
                                  nudge_x0 = TRUE,
                                  breaks = NULL) {
  method <- match.arg(method)
  
  x <- x[is.finite(x)]
  n <- length(x)
  
  if (n < 5) stop("Too few observations.")
  if (!is.finite(x0)) stop("x0 must be finite.")
  if (!is.finite(h) || h <= 0) stop("h must be > 0.")
  if (!is.finite(alpha) || alpha <= 0 || alpha >= 1) stop("alpha must be in (0,1).")
  
  # Breaks: either user supplies them, or we build fixed-width breaks.
  if (is.null(breaks)) {
    breaks <- make_fixed_breaks(x, x0, h)
  } else {
    breaks <- sort(unique(breaks))
    if (length(breaks) < 2) stop("breaks must have length >= 2.")
    # If we pass breaks, we infer h from them (as can be seen, we need h to scale p -> density).
    h <- median(diff(breaks))
    if (!is.finite(h) || h <= 0) stop("Invalid breaks (non-positive bin width).")
  }
  
  # Optional: nudge x0 if it's on an edge
  x0_used <- x0
  if (isTRUE(nudge_x0)) x0_used <- nudge_x0_left_if_on_edge(x0_used, breaks, h)
  
  # Which bin contains x0?
  # We match hist(..., right=TRUE) behavior using findInterval with rightmost.closed=TRUE.
  j0 <- findInterval(x0_used, breaks, rightmost.closed = TRUE, all.inside = TRUE)
  
  # Count how many Xi fall in that same bin
  jx <- findInterval(x, breaks, rightmost.closed = TRUE, all.inside = TRUE)
  C  <- sum(jx == j0)
  
  # Plug-in histogram estimate at x0
  p_hat <- C / n
  f_hat <- p_hat / h
  
  # CI for p_B, then divide by h to get CI for f(x0)
  conf_level <- 1 - alpha
  
  if (method == "exact") {
    # Exact Clopper–Pearson for Binomial(n, p_B)
    p_ci <- as.numeric(binom.test(C, n, conf.level = conf_level)$conf.int)
  } else {
    # Approx (score/Wilson-ish) interval via prop.test, no continuity correction
    p_ci <- as.numeric(suppressWarnings(
      prop.test(C, n, conf.level = conf_level, correct = FALSE)$conf.int
    ))
  }
  
  f_ci <- p_ci / h
  
  # Return a clean little list
  list(
    x0 = x0,
    x0_used = x0_used,
    h = h,
    n = n,
    count_in_bin = C,
    bin_left = breaks[j0],
    bin_right = breaks[j0 + 1],
    f_hat = f_hat,
    f_ci = f_ci,
    p_hat = p_hat,
    p_ci = p_ci,
    conf_level = conf_level,
    method = method,
    breaks = breaks
  )
}

# ============================================================
# Example run (we keep it simple)
# ============================================================

set.seed(1)
x  <- rbeta(1000, 2, 5)
x0 <- 0.30
h  <- 0.05

ci_exact <- hist_ci_density_point(x, x0, h, alpha = 0.05, method = "exact")
ci_prop  <- hist_ci_density_point(x, x0, h, alpha = 0.05, method = "prop")

print(ci_exact)
print(ci_prop)

cat(sprintf("Estimated f(x0)=%.4f; 95%% CI [%.4f, %.4f]\n",
            ci_exact$f_hat, ci_exact$f_ci[1], ci_exact$f_ci[2]))

# Optional quick visual check:
hist(x, breaks = ci_exact$breaks, freq = FALSE,
     main = "Histogram; target bin for pointwise CI",
     xlab = "x")
abline(v = c(ci_exact$bin_left, ci_exact$bin_right), lty = 2)
points(ci_exact$x0, ci_exact$f_hat, pch = 19)


