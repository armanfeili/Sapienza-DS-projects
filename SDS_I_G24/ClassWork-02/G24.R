# Classwork-02
# Team: G24
# Team members: Arman Feili, Sohrab Seyyedi Parsa, Milad Torabi

# Q1)

options(stringsAsFactors = FALSE)

# 0) Install + load what we need
pkgs <- c("gsl", "copula")

for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, quiet = TRUE)
  }
}

if (!requireNamespace("gsl", quietly = TRUE)) {
  stop("Package 'gsl' is missing. Please install it first, then re-run the script.")
}
if (!requireNamespace("copula", quietly = TRUE)) {
  stop("Package 'copula' is missing. Please install it, then re-run the script.")
}

suppressPackageStartupMessages(library(copula))
cat("Engine selected: copula\n")

# 1) Pick the two models you want to compare
# Allowed: gaussian, t, clayton, gumbel, frank, joe
MODEL_1 <- list(id = "gaussian", name = "Gaussian",  t_df_start = 6)
MODEL_2 <- list(id = "t",        name = "Student-t", t_df_start = 6)

START_RHO <- 0.2  # starting value for ML (mainly for Gaussian / t)

# 2) Load data
crsp <- read.csv("CRSPday.csv")

# optional date column (nice to have)
if (all(c("year","month","day") %in% names(crsp))) {
  crsp$date <- as.Date(sprintf("%04d-%02d-%02d", crsp$year, crsp$month, crsp$day))
}

needed_cols <- c("ge","ibm","mobil","crsp")
missing_cols <- setdiff(needed_cols, names(crsp))
if (length(missing_cols) > 0) {
  stop("Missing columns in CRSPday.csv: ", paste(missing_cols, collapse = ", "))
}

# 3) Pseudo-observations (turn data into (0,1) using ranks)
pobs_simple <- function(z) {
  n <- length(z)
  rank(z, ties.method = "average", na.last = "keep") / (n + 1)
}

to_uv <- function(x, y, jitter_if_ties = TRUE, jitter_sd = 1e-10, seed = 123) {
  d <- data.frame(x = x, y = y)
  d <- d[complete.cases(d), , drop = FALSE]
  
  if (nrow(d) < 30) stop("Too few complete observations: n = ", nrow(d))
  if (sd(d$x) == 0 || sd(d$y) == 0) stop("One series is constant (sd=0), cannot fit a copula.")
  
  # If there are many ties, add a tiny jitter so ranks are not too flat.
  if (jitter_if_ties) {
    tie_x <- 1 - length(unique(d$x)) / length(d$x)
    tie_y <- 1 - length(unique(d$y)) / length(d$y)
    
    if (tie_x > 0.05 || tie_y > 0.05) {
      set.seed(seed)
      if (tie_x > 0.05) d$x <- d$x + rnorm(nrow(d), 0, jitter_sd)
      if (tie_y > 0.05) d$y <- d$y + rnorm(nrow(d), 0, jitter_sd)
    }
  }
  
  uv <- cbind(pobs_simple(d$x), pobs_simple(d$y))
  colnames(uv) <- c("u","v")
  uv
}

# 4) Build the right copula object from the chosen ID
make_copula <- function(model, start_rho = 0.2) {
  id <- tolower(model$id)
  
  if (id == "gaussian") {
    return(normalCopula(start_rho, dim = 2, dispstr = "un"))
  }
  
  if (id == "t") {
    df0 <- ifelse(is.null(model$t_df_start), 6, model$t_df_start)
    return(tCopula(start_rho, dim = 2, df = df0, df.fixed = FALSE, dispstr = "un"))
  }
  
  if (id == "clayton") return(claytonCopula(1,   dim = 2))
  if (id == "gumbel")  return(gumbelCopula(1.2,  dim = 2))
  if (id == "frank")   return(frankCopula(2,     dim = 2))
  if (id == "joe")     return(joeCopula(1.2,     dim = 2))
  
  stop("Unknown model id: ", model$id)
}

# 5) Fit both models and compare with AIC / BIC
# AIC = -2*logLik + 2*k
# BIC = -2*logLik + log(n)*k
compare_two_models <- function(uv, model1 = MODEL_1, model2 = MODEL_2, start_rho = START_RHO) {
  n <- nrow(uv)
  
  cop1 <- make_copula(model1, start_rho)
  fit1 <- tryCatch(fitCopula(cop1, uv, method = "ml"),
                   error = function(e) stop("Fit failed for ", model1$name, ": ", e$message))
  
  cop2 <- make_copula(model2, start_rho)
  fit2 <- tryCatch(fitCopula(cop2, uv, method = "ml"),
                   error = function(e) stop("Fit failed for ", model2$name, ": ", e$message))
  
  ll1 <- as.numeric(logLik(fit1)); k1 <- length(coef(fit1))
  ll2 <- as.numeric(logLik(fit2)); k2 <- length(coef(fit2))
  
  tab <- data.frame(
    model = c(model1$name, model2$name),
    logLik = c(ll1, ll2),
    k = c(k1, k2),
    AIC = c(-2*ll1 + 2*k1,         -2*ll2 + 2*k2),
    BIC = c(-2*ll1 + log(n)*k1,    -2*ll2 + log(n)*k2)
  )
  
  list(
    summary = tab,
    winner_AIC = tab$model[which.min(tab$AIC)],
    winner_BIC = tab$model[which.min(tab$BIC)],
    fit1 = fit1,
    fit2 = fit2
  )
}

# small helper to print results nicely
print_result <- function(title, res) {
  cat("\n===", title, "===\n")
  print(res$summary)
  cat("Winner by AIC:", res$winner_AIC, "\n")
  cat("Winner by BIC:", res$winner_BIC, "\n")
}

# 6) Run the 6 pairs (no loops, as requested)

res_ge_ibm   <- compare_two_models(to_uv(crsp$ge,    crsp$ibm))
print_result("GE vs IBM", res_ge_ibm)

res_ge_mobil <- compare_two_models(to_uv(crsp$ge,    crsp$mobil))
print_result("GE vs MOBIL", res_ge_mobil)

res_ge_crsp  <- compare_two_models(to_uv(crsp$ge,    crsp$crsp))
print_result("GE vs CRSP", res_ge_crsp)

res_ibm_mobil <- compare_two_models(to_uv(crsp$ibm,  crsp$mobil))
print_result("IBM vs MOBIL", res_ibm_mobil)

res_ibm_crsp <- compare_two_models(to_uv(crsp$ibm,   crsp$crsp))
print_result("IBM vs CRSP", res_ibm_crsp)

res_mobil_crsp <- compare_two_models(to_uv(crsp$mobil, crsp$crsp))
print_result("MOBIL vs CRSP", res_mobil_crsp)




# Q2)

options(stringsAsFactors = FALSE)
set.seed(123)

# Some helper functions

# Split indices 1..n into K random blocks (almost equal size)
make_blocks <- function(n, K, seed = NULL) {
  n <- as.integer(n); K <- as.integer(K)
  if (n < 1) stop("n must be >= 1")
  if (K < 1) stop("K must be >= 1")
  K <- min(K, n)
  
  if (!is.null(seed)) set.seed(seed)
  
  idx <- sample.int(n)  # random shuffle
  split(idx, rep(1:K, length.out = n))
}

# Theory choice: K = ceil(8 log(1/alpha))
k_star <- function(alpha) {
  if (!is.finite(alpha) || alpha <= 0 || alpha >= 1) stop("alpha must be in (0,1)")
  ceiling(8 * log(1 / alpha))
}

# MoM mean = median of block means
mom_mean <- function(x, K, seed = 1) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 5) stop("Too few observations for MoM mean.")
  
  # keep blocks not too small: block size >= 2
  K <- as.integer(K)
  K <- max(1, min(K, floor(n / 2)))
  
  blocks <- make_blocks(n, K, seed)
  block_means <- sapply(blocks, function(ii) mean(x[ii]))
  median(block_means)
}


# Part A: mean under outliers

# Pareto sampler: X = xm / U^(1/shape), U~Unif(0,1)
rpareto <- function(n, xm = 1, shape = 2.5) {
  if (!is.finite(xm) || xm <= 0) stop("xm must be > 0")
  if (!is.finite(shape) || shape <= 0) stop("shape must be > 0")
  u <- runif(n)
  xm / (u^(1 / shape))
}

# Replace about eps*n points with big spikes (+/- outlier_value)
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
  
  dists <- list(
    Normal_0_1 = list(r = function(nn) rnorm(nn, 0, 1), mu = 0),
    t_df3      = list(r = function(nn) rt(nn, df = 3),  mu = 0),
    Pareto_2.5 = list(
      r  = function(nn) rpareto(nn, xm = 1, shape = 2.5),
      mu = 2.5 / (2.5 - 1)   # mean exists since shape > 1
    )
  )
  
  out <- lapply(names(dists), function(name) {
    dd <- dists[[name]]
    err_mean <- numeric(R)
    err_mom  <- numeric(R)
    
    for (r in 1:R) {
      x <- dd$r(n)
      x <- contaminate(x, eps = eps, outlier_value = 50, seed = 1000 + r)
      
      err_mean[r] <- abs(mean(x) - dd$mu)
      err_mom[r]  <- abs(mom_mean(x, K, seed = 2000 + r) - dd$mu)
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



# Part B: MoM histogram on [0,1] + L2 loss


# True density (for L2 loss)
f_true <- function(x) dbeta(x, 2, 5)

# keep points strictly inside (0,1) so binning is stable
clip01 <- function(x, eps = 1e-12) pmin(1 - eps, pmax(0 + eps, x))

# Contamination in [0,1]: some go near 0.99, others are random uniform noise
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

# Standard histogram density (and optionally re-normalize)
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

# MoM histogram:
# build hist on each block, take median of bin probabilities, then / binwidth
mom_hist_density <- function(x, breaks, K, seed = 1, rescale_to_1 = TRUE) {
  x <- clip01(x[is.finite(x)])
  n <- length(x)
  if (n < 10) stop("Too few observations for MoM histogram.")
  
  # avoid tiny blocks
  K <- as.integer(K)
  K <- max(1, min(K, floor(n / 5)))
  
  blocks <- make_blocks(n, K, seed)
  widths <- diff(breaks)
  J <- length(widths)
  
  # each column = bin probs for one block
  p_mat <- vapply(blocks, function(ii) {
    hi <- hist(x[ii], breaks = breaks, plot = FALSE, right = TRUE, include.lowest = TRUE)
    hi$counts / length(ii)
  }, numeric(J))
  
  p_med <- apply(p_mat, 1, median)
  dens  <- p_med / widths
  
  if (rescale_to_1) {
    mass <- sum(dens * widths)   # same as sum(p_med)
    if (mass > 0) dens <- dens / mass
  }
  
  list(density = dens, breaks = breaks, K = K)
}

# Evaluate the piecewise-constant histogram on a grid
eval_hist <- function(xgrid, breaks, dens) {
  xgrid <- clip01(xgrid)
  j <- findInterval(xgrid, breaks, rightmost.closed = TRUE, all.inside = TRUE)
  dens[j]
}

# L2 loss via a fine grid
L2_loss <- function(fhat, ftrue, m = 5000, eps = 1e-12) {
  xg <- seq(0 + eps, 1 - eps, length.out = m)
  dx <- xg[2] - xg[1]
  sum((fhat(xg) - ftrue(xg))^2) * dx
}

run_hist_experiment <- function(n = 1000, h = 0.05, alpha = 0.05, eps = 0.10, R = 150) {
  breaks <- seq(0, 1, by = h)
  if (tail(breaks, 1) < 1) breaks <- c(breaks, 1)
  
  K <- k_star(alpha)
  
  loss_std <- numeric(R)
  loss_mom <- numeric(R)
  
  for (r in 1:R) {
    x <- rbeta(n, 2, 5)
    x <- contaminate_unit(x, eps = eps, seed = 3000 + r)
    
    std <- hist_density(x, breaks)
    mh  <- mom_hist_density(x, breaks, K, seed = 4000 + r)
    
    f_std <- function(t) eval_hist(t, std$breaks, std$density)
    f_mom <- function(t) eval_hist(t, mh$breaks,  mh$density)
    
    loss_std[r] <- L2_loss(f_std, f_true)
    loss_mom[r] <- L2_loss(f_mom, f_true)
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


# One example plot

plot_one_example <- function(n = 1000, h = 0.05, alpha = 0.05, eps = 0.10) {
  breaks <- seq(0, 1, by = h)
  if (tail(breaks, 1) < 1) breaks <- c(breaks, 1)
  
  K <- k_star(alpha)
  
  x <- rbeta(n, 2, 5)
  x <- contaminate_unit(x, eps = eps, seed = 777)
  
  std <- hist_density(x, breaks)
  mh  <- mom_hist_density(x, breaks, K, seed = 888)
  
  mids <- (breaks[-1] + breaks[-length(breaks)]) / 2
  
  plot(mids, std$density, type = "h", lwd = 2, lty = 1,
       main = "Histogram density: standard vs MoM–H",
       xlab = "x", ylab = "density")
  
  lines(mids, mh$density, type = "h", lwd = 2, lty = 2)
  curve(f_true(x), from = 0, to = 1, add = TRUE, lwd = 2, lty = 3)
  
  legend("topright",
         legend = c("standard hist", "MoM–H", "true density"),
         lwd = 2,
         lty = c(1, 2, 3),
         bty = "n")
  
  # Short interpretation (2 sentences)
  cat("\nPlot interpretation: The standard histogram is more affected by the contaminated points (especially near 1), so some bins get inflated compared to the true density.\n")
  cat("The MoM–H version is more robust: it stays closer to the true density curve because the median across blocks down-weights the outlier-heavy blocks.\n")
}

plot_one_example()





# BONUS Q)
# We pick the bin that contains x0, and count how many data points fall in it,
# then treat that count as Binomial(n, p_bin). Finally convert p_bin to density by /h.

options(stringsAsFactors = FALSE)

# Make fixed-width breaks so x0 is covered
make_breaks <- function(x, x0, h) {
  x <- x[is.finite(x)]
  if (length(x) < 2) stop("Need at least 2 finite observations.")
  if (!is.finite(x0)) stop("x0 must be finite.")
  if (!is.finite(h) || h <= 0) stop("h must be > 0.")
  
  lo <- floor(min(c(x, x0)) / h) * h
  hi <- ceiling(max(c(x, x0)) / h) * h
  if (hi <= lo) hi <- lo + h
  
  br <- seq(lo, hi, by = h)
  br <- sort(unique(br))
  if (length(br) < 2) stop("Failed to build valid breaks.")
  br
}

# If x0 lands exactly on a bin edge, push it a tiny bit left
nudge_left <- function(x0, breaks, h) {
  tol <- max(1e-12, 10 * .Machine$double.eps * max(1, abs(h)))
  if (any(abs(x0 - breaks) <= tol)) x0 <- x0 - tol
  x0
}

# Main function: CI for f(x0)
hist_point_CI <- function(x, x0, h, alpha = 0.05,
                          method = c("exact", "prop"),
                          nudge = TRUE,
                          breaks = NULL) {
  method <- match.arg(method)
  
  x <- x[is.finite(x)]
  n <- length(x)
  
  if (n < 5) stop("Too few observations.")
  if (!is.finite(x0)) stop("x0 must be finite.")
  if (!is.finite(h) || h <= 0) stop("h must be > 0.")
  if (!is.finite(alpha) || alpha <= 0 || alpha >= 1) stop("alpha must be in (0,1).")
  
  # choose breaks
  if (is.null(breaks)) {
    breaks <- make_breaks(x, x0, h)
  } else {
    breaks <- sort(unique(breaks))
    if (length(breaks) < 2) stop("breaks must have length >= 2.")
    h <- median(diff(breaks))  # infer bin width from breaks
    if (!is.finite(h) || h <= 0) stop("Invalid breaks (non-positive bin width).")
  }
  
  # optional nudge
  x0_used <- x0
  if (isTRUE(nudge)) x0_used <- nudge_left(x0_used, breaks, h)
  
  # which bin is x0 in? (match hist(..., right=TRUE): bins are (a,b])
  bin_x0 <- findInterval(x0_used, breaks,
                         left.open = TRUE, rightmost.closed = TRUE, all.inside = TRUE)
  
  # count how many x fall in the same bin (same bin rule as above)
  bin_x <- findInterval(x, breaks,
                        left.open = TRUE, rightmost.closed = TRUE, all.inside = TRUE)
  C <- sum(bin_x == bin_x0)
  
  # histogram estimate at x0: fhat = (C/n)/h
  p_hat <- C / n
  f_hat <- p_hat / h
  
  # CI for p_bin, then divide by h
  conf_level <- 1 - alpha
  if (method == "exact") {
    p_ci <- as.numeric(binom.test(C, n, conf.level = conf_level)$conf.int)
  } else {
    p_ci <- as.numeric(suppressWarnings(
      prop.test(C, n, conf.level = conf_level, correct = FALSE)$conf.int
    ))
  }
  f_ci <- p_ci / h
  
  list(
    x0 = x0,
    x0_used = x0_used,
    n = n,
    h = h,
    bin_left = breaks[bin_x0],
    bin_right = breaks[bin_x0 + 1],
    count_in_bin = C,
    p_hat = p_hat,
    p_ci = p_ci,
    f_hat = f_hat,
    f_ci = f_ci,
    conf_level = conf_level,
    method = method,
    breaks = breaks
  )
}

# Example run

set.seed(1)
x  <- rbeta(1000, 2, 5)
x0 <- 0.30
h  <- 0.05

ci_exact <- hist_point_CI(x, x0, h, alpha = 0.05, method = "exact")
ci_prop  <- hist_point_CI(x, x0, h, alpha = 0.05, method = "prop")

print(ci_exact)
print(ci_prop)

cat(sprintf("Estimated f(x0)=%.4f; 95%% CI [%.4f, %.4f]\n",
            ci_exact$f_hat, ci_exact$f_ci[1], ci_exact$f_ci[2]))

# Quick visual check:
hist(x, breaks = ci_exact$breaks, freq = FALSE,
     main = "Histogram; target bin for pointwise CI",
     xlab = "x")
abline(v = c(ci_exact$bin_left, ci_exact$bin_right), lty = 2)
points(ci_exact$x0, ci_exact$f_hat, pch = 19)


# Interpretation:
cat("As can be seen, the dashed lines show the bin that contains x0, and the black dot is the histogram estimate f_hat(x0) from that bin.\n")
cat("More points in this bin means a higher estimate and a tighter CI; fewer points means a lower estimate and a wider CI.\n")


