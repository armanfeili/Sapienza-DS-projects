###############################################
# Classwork-01 – Q1)  (all pairs 2-by-2)
###############################################

## 1. Import dataset
grsp <- read.csv("./CRSPday.csv", header = TRUE)

## Quick look at the data structure and variable names
names(grsp)
str(grsp)
head(grsp)

## (Optional) construct a Date variable if year/month/day are present
if (all(c("year", "month", "day") %in% names(grsp))) {
  grsp$Date <- as.Date(
    with(grsp, paste(year, month, day, sep = "-")),
    format = "%Y-%m-%d"
  )
}

########################################################
# PAIR 1: ge vs ibm
########################################################

stock_name <- "ge"
index_name <- "ibm"

x <- grsp[[stock_name]]
y <- grsp[[index_name]]

pair_data <- data.frame(
  stock = x,
  index = y
)

pair_data <- pair_data[complete.cases(pair_data), ]

## Quick check
cat("========= Pair:", stock_name, "vs", index_name, "=========\n")
dim(pair_data)
head(pair_data)

## Univariate summaries
summary(pair_data$stock)
sd(pair_data$stock, na.rm = TRUE)

summary(pair_data$index)
sd(pair_data$index, na.rm = TRUE)

## Joint summaries
cor(pair_data$stock, pair_data$index)
cor(pair_data$stock, pair_data$index, method = "spearman")

mu_hat <- colMeans(pair_data)
Sigma_hat <- var(pair_data)

mu_hat
Sigma_hat

## Plots
par(mfrow = c(1, 2))
hist(pair_data$stock,
     main = paste("Histogram of", stock_name, "returns"),
     xlab = "Return")
hist(pair_data$index,
     main = paste("Histogram of", index_name, "returns"),
     xlab = "Return")

par(mfrow = c(1, 1))
plot(pair_data$stock, pair_data$index,
     main = paste(stock_name, "vs", index_name, "returns"),
     xlab = paste(stock_name, "return"),
     ylab = paste(index_name, "return"))
abline(lm(pair_data$index ~ pair_data$stock), col = "red")

par(mfrow = c(1, 2))
qqnorm(pair_data$stock,
       main = paste("QQ-plot of", stock_name, "returns"))
qqline(pair_data$stock)
qqnorm(pair_data$index,
       main = paste("QQ-plot of", index_name, "returns"))
qqline(pair_data$index)

par(mfrow = c(1, 1))  # reset layout

# Description of the output:
# **Histograms (GE vs IBM returns)**

# * Both are single-peaked and roughly symmetric around 0 → returns mostly small, no strong skew.
# * IBM histogram is wider → IBM is **more volatile** (higher standard deviation) than GE.
# * Visible few bars far in the tails → occasional **large positive/negative returns** for both.

# * **Scatterplot (GE return vs IBM return)**
#   
#   * Cloud of points roughly elliptical with a **positively sloped** regression line.
# * Points are quite dispersed around the line → **moderate, not strong**, positive dependence.
# * When GE goes up/down, IBM tends to move in the same direction, but with substantial noise.

# * **QQ-plots (GE and IBM separately)**
#   
#   * Central points lie close to the straight line → **bulk of returns ≈ normal**.
# * Tails deviate from the line (both ends) → **heavier tails** than a Gaussian (more extremes than normal).

# * **Qualitative joint (bivariate) probabilistic model**
#   
#   * Each marginal: approximately **zero-mean, symmetric, near-Gaussian** with heavy tails (IBM more volatile).
# * Jointly: returns form an **elliptical cloud with correlation ≈ 0.3** (moderate positive correlation).
# * A reasonable approximation: **bivariate normal distribution** with small positive means and the estimated covariance matrix, but with the caveat of **fatter-than-normal tails**.


########################################################
# PAIR 2: ge vs mobil
########################################################

stock_name <- "ge"
index_name <- "mobil"

x <- grsp[[stock_name]]
y <- grsp[[index_name]]

pair_data <- data.frame(
  stock = x,
  index = y
)

pair_data <- pair_data[complete.cases(pair_data), ]

cat("========= Pair:", stock_name, "vs", index_name, "=========\n")
dim(pair_data)
head(pair_data)

summary(pair_data$stock)
sd(pair_data$stock, na.rm = TRUE)

summary(pair_data$index)
sd(pair_data$index, na.rm = TRUE)

cor(pair_data$stock, pair_data$index)
cor(pair_data$stock, pair_data$index, method = "spearman")

mu_hat <- colMeans(pair_data)
Sigma_hat <- var(pair_data)

mu_hat
Sigma_hat

par(mfrow = c(1, 2))
hist(pair_data$stock,
     main = paste("Histogram of", stock_name, "returns"),
     xlab = "Return")
hist(pair_data$index,
     main = paste("Histogram of", index_name, "returns"),
     xlab = "Return")

par(mfrow = c(1, 1))
plot(pair_data$stock, pair_data$index,
     main = paste(stock_name, "vs", index_name, "returns"),
     xlab = paste(stock_name, "return"),
     ylab = paste(index_name, "return"))
abline(lm(pair_data$index ~ pair_data$stock), col = "red")

par(mfrow = c(1, 2))
qqnorm(pair_data$stock,
       main = paste("QQ-plot of", stock_name, "returns"))
qqline(pair_data$stock)
qqnorm(pair_data$index,
       main = paste("QQ-plot of", index_name, "returns"))
qqline(pair_data$index)

par(mfrow = c(1, 1))


# Description of the output:
# 
# * **Histograms (GE vs Mobil returns)**
#   
#   * Both distributions are **single-peaked and roughly symmetric** around 0.
# * Mean returns are small and positive (≈ 0.11% GE, ≈ 0.08% Mobil) → very mild upward drift.
# * Volatilities are similar (sd ≈ 0.0137 GE, 0.0129 Mobil) → **comparable risk level**, with a few large positive/negative days (visible tails).
# 
# * **Scatterplot (GE return vs Mobil return)**
#   
#   * Point cloud is roughly **elliptical** with a **positively sloped** regression line.
# * Correlations (Pearson ≈ 0.30, Spearman ≈ 0.27) → **moderate positive dependence**: when GE goes up, Mobil tends to go up, but with substantial noise.
# 
# * **QQ-plots (GE and Mobil separately)**
#   
#   * Central points lie close to the straight line → **bulk of returns ≈ normal**.
# * Tails deviate from the line on both sides → **heavier tails than Gaussian**, i.e. more extreme returns than a pure normal model would predict.
# 
# * **Qualitative joint (bivariate) model**
#   
#   * Each stock: **approximately zero-mean, symmetric, near-Gaussian** daily returns with similar variance and moderately heavy tails.
# * Jointly: returns form an **approximately elliptical distribution with moderate positive correlation (~0.3)**.
# * A reasonable approximation: **bivariate normal** with mean vector
# ((\mu_{\text{GE}}, \mu_{\text{Mobil}}) ≈ (0.0011, 0.00078))
# and covariance matrix
# (\Sigma ≈ \begin{pmatrix}1.88!\times!10^{-4} & 5.27!\times!10^{-5} \ 5.27!\times!10^{-5} & 1.67!\times!10^{-4}\end{pmatrix}),
# keeping in mind the empirical **fat tails**.


########################################################
# PAIR 3: ge vs crsp
########################################################

stock_name <- "ge"
index_name <- "crsp"

x <- grsp[[stock_name]]
y <- grsp[[index_name]]

pair_data <- data.frame(
  stock = x,
  index = y
)

pair_data <- pair_data[complete.cases(pair_data), ]

cat("========= Pair:", stock_name, "vs", index_name, "=========\n")
dim(pair_data)
head(pair_data)

summary(pair_data$stock)
sd(pair_data$stock, na.rm = TRUE)

summary(pair_data$index)
sd(pair_data$index, na.rm = TRUE)

cor(pair_data$stock, pair_data$index)
cor(pair_data$stock, pair_data$index, method = "spearman")

mu_hat <- colMeans(pair_data)
Sigma_hat <- var(pair_data)

mu_hat
Sigma_hat

par(mfrow = c(1, 2))
hist(pair_data$stock,
     main = paste("Histogram of", stock_name, "returns"),
     xlab = "Return")
hist(pair_data$index,
     main = paste("Histogram of", index_name, "returns"),
     xlab = "Return")

par(mfrow = c(1, 1))
plot(pair_data$stock, pair_data$index,
     main = paste(stock_name, "vs", index_name, "returns"),
     xlab = paste(stock_name, "return"),
     ylab = paste(index_name, "return"))
abline(lm(pair_data$index ~ pair_data$stock), col = "red")

par(mfrow = c(1, 2))
qqnorm(pair_data$stock,
       main = paste("QQ-plot of", stock_name, "returns"))
qqline(pair_data$stock)
qqnorm(pair_data$index,
       main = paste("QQ-plot of", index_name, "returns"))
qqline(pair_data$index)

par(mfrow = c(1, 1))


# * **Histograms (GE vs CRSP returns)**
#   
#   * Both are **single-peaked and roughly symmetric** around 0 → returns usually small, no strong skew.
# * GE has wider support than CRSP → **GE slightly more volatile** (sd ≈ 0.0137 vs 0.0078).
# * Tails show a few large positive/negative days for each series.
# 
# * **Scatterplot (GE return vs CRSP return)**
#   
#   * Points form a **tight, elongated ellipse** along an increasing line.
# * Regression line has strong positive slope; Pearson correlation ≈ **0.71**, Spearman ≈ **0.64** → **strong positive dependence**: market (CRSP) and GE move together closely.
# 
# * **QQ-plots (GE and CRSP separately)**
#   
#   * Centre of both plots lies near the straight line → **bulk of returns roughly normal**.
# * Tails deviate from the line (especially extremes) → **heavier tails than Gaussian**, with more extreme returns than a pure normal model.
# 
# * **Qualitative joint (bivariate) probabilistic model**
#   
#   * Each marginal: **approximately symmetric, near-Gaussian daily returns with small positive mean**, GE more volatile than CRSP.
# * Jointly: **strongly positively correlated elliptical distribution**, well approximated by a **bivariate normal** with mean vector
# ((\mu_{\text{GE}}, \mu_{\text{CRSP}}) ≈ (0.00107, 0.00068))
# and covariance matrix
# (\Sigma ≈ \begin{pmatrix}1.88!\times!10^{-4} & 7.61!\times!10^{-5}\ 7.61!\times!10^{-5} & 6.02!\times!10^{-5}\end{pmatrix}),
# noting that real data show **fat tails** compared to an ideal Gaussian.

########################################################
# PAIR 4: ibm vs mobil
########################################################

stock_name <- "ibm"
index_name <- "mobil"

x <- grsp[[stock_name]]
y <- grsp[[index_name]]

pair_data <- data.frame(
  stock = x,
  index = y
)

pair_data <- pair_data[complete.cases(pair_data), ]

cat("========= Pair:", stock_name, "vs", index_name, "=========\n")
dim(pair_data)
head(pair_data)

summary(pair_data$stock)
sd(pair_data$stock, na.rm = TRUE)

summary(pair_data$index)
sd(pair_data$index, na.rm = TRUE)

cor(pair_data$stock, pair_data$index)
cor(pair_data$stock, pair_data$index, method = "spearman")

mu_hat <- colMeans(pair_data)
Sigma_hat <- var(pair_data)

mu_hat
Sigma_hat

par(mfrow = c(1, 2))
hist(pair_data$stock,
     main = paste("Histogram of", stock_name, "returns"),
     xlab = "Return")
hist(pair_data$index,
     main = paste("Histogram of", index_name, "returns"),
     xlab = "Return")

par(mfrow = c(1, 1))
plot(pair_data$stock, pair_data$index,
     main = paste(stock_name, "vs", index_name, "returns"),
     xlab = paste(stock_name, "return"),
     ylab = paste(index_name, "return"))
abline(lm(pair_data$index ~ pair_data$stock), col = "red")

par(mfrow = c(1, 2))
qqnorm(pair_data$stock,
       main = paste("QQ-plot of", stock_name, "returns"))
qqline(pair_data$stock)
qqnorm(pair_data$index,
       main = paste("QQ-plot of", index_name, "returns"))
qqline(pair_data$index)

par(mfrow = c(1, 1))

# * **Histograms (IBM vs Mobil returns)**
#   
#   * Both series are **single-peaked and roughly symmetric** around 0 → daily returns mostly small, no strong skew.
# * IBM has wider support and higher sd (≈ 1.75% vs 1.29%) → **IBM is more volatile** than Mobil, with more extreme returns in both tails.
# 
# * **Scatterplot (IBM return vs Mobil return)**
#   
#   * Point cloud is fairly round and diffuse, with a **slightly upward-sloping** regression line.
# * Pearson ≈ **0.16**, Spearman ≈ **0.17** → only a **weak positive dependence**: when IBM moves, Mobil tends to move in the same direction but the effect is small and noisy.
# 
# * **QQ-plots (IBM and Mobil separately)**
#   
#   * Central points lie close to the straight line → **bulk of returns ≈ normal**.
# * Tails deviate clearly (especially for IBM) → **heavy tails**: more extreme positive and negative returns than a Gaussian.
# 
# * **Qualitative joint (bivariate) probabilistic model**
#   
#   * Each marginal: **approximately symmetric, near-Gaussian daily returns with small positive mean**, IBM more volatile and more heavy-tailed.
# * Jointly: **weakly positively correlated** bivariate distribution, roughly elliptical but quite round due to low correlation.
# * A simple approximation: **bivariate normal** with mean vector
# ((\mu_{\text{IBM}}, \mu_{\text{Mobil}}) ≈ (0.0007, 0.00078))
# and covariance matrix
# (\Sigma ≈ \begin{pmatrix}3.06!\times!10^{-4} & 3.59!\times!10^{-5}\ 3.59!\times!10^{-5} & 1.67!\times!10^{-4}\end{pmatrix}),
# remembering that real data show **fatter tails** than the Gaussian ideal.
# 

########################################################
# PAIR 5: ibm vs crsp
########################################################

stock_name <- "ibm"
index_name <- "crsp"

x <- grsp[[stock_name]]
y <- grsp[[index_name]]

pair_data <- data.frame(
  stock = x,
  index = y
)

pair_data <- pair_data[complete.cases(pair_data), ]

cat("========= Pair:", stock_name, "vs", index_name, "=========\n")
dim(pair_data)
head(pair_data)

summary(pair_data$stock)
sd(pair_data$stock, na.rm = TRUE)

summary(pair_data$index)
sd(pair_data$index, na.rm = TRUE)

cor(pair_data$stock, pair_data$index)
cor(pair_data$stock, pair_data$index, method = "spearman")

mu_hat <- colMeans(pair_data)
Sigma_hat <- var(pair_data)

mu_hat
Sigma_hat

par(mfrow = c(1, 2))
hist(pair_data$stock,
     main = paste("Histogram of", stock_name, "returns"),
     xlab = "Return")
hist(pair_data$index,
     main = paste("Histogram of", index_name, "returns"),
     xlab = "Return")

par(mfrow = c(1, 1))
plot(pair_data$stock, pair_data$index,
     main = paste(stock_name, "vs", index_name, "returns"),
     xlab = paste(stock_name, "return"),
     ylab = paste(index_name, "return"))
abline(lm(pair_data$index ~ pair_data$stock), col = "red")

par(mfrow = c(1, 2))
qqnorm(pair_data$stock,
       main = paste("QQ-plot of", stock_name, "returns"))
qqline(pair_data$stock)
qqnorm(pair_data$index,
       main = paste("QQ-plot of", index_name, "returns"))
qqline(pair_data$index)

par(mfrow = c(1, 1))


# * **Univariate behaviour (histograms)**
#   
#   * Both IBM and CRSP daily returns are tightly concentrated around 0 with small positive means (~0.0007 and ~0.0007).
# * Shapes are roughly bell-shaped and symmetric, suggesting approximately normal marginal distributions.
# * IBM’s histogram is wider than CRSP’s → IBM is more volatile (sd ≈ 0.0175 vs 0.0078).
# 
# * **Normality checks (QQ-plots)**
#   
#   * Points lie close to the straight line in the middle → central part of both distributions is close to Gaussian.
# * Clear deviations in the tails (both left and right) → heavier tails / more extreme returns than a perfect normal, especially for IBM.
# * CRSP also shows tail deviations but slightly less pronounced.
# 
# * **Dependence structure (scatterplot)**
#   
#   * Cloud is elongated along an upward-sloping line → clear positive linear relationship.
# * Correlations: Pearson ≈ 0.49, Spearman ≈ 0.47 → moderately strong, mainly linear and monotone dependence.
# * Some dispersion around the line → IBM does not move one-for-one with the market; there is idiosyncratic noise.
# 
# * **Qualitative joint (bivariate) probabilistic model**
#   
#   * Treat ((R_{IBM}, R_{CRSP})) as **approximately bivariate normal**, with small positive means and covariance matrix
# (\Sigma \approx \begin{pmatrix} 3.06\cdot10^{-4} & 6.6\cdot10^{-5} \ 6.6\cdot10^{-5} & 6.0\cdot10^{-5} \end{pmatrix}).
# * Conditional view:
#   
#   * (R_{IBM} \approx \alpha + \beta,R_{CRSP} + \varepsilon), with (\beta \approx \text{cov}/\text{var} \approx 1.1) and (\varepsilon) ≈ normal noise.
# * Interpretation: IBM behaves like a stock with slightly **above-market beta**, more volatile than the index but strongly driven by it.
# * Model is reasonable in the center of the distribution but underestimates extreme co-movements because of the observed heavy tails.


########################################################
# PAIR 6: mobil vs crsp
########################################################

stock_name <- "mobil"
index_name <- "crsp"

x <- grsp[[stock_name]]
y <- grsp[[index_name]]

pair_data <- data.frame(
  stock = x,
  index = y
)

pair_data <- pair_data[complete.cases(pair_data), ]

cat("========= Pair:", stock_name, "vs", index_name, "=========\n")
dim(pair_data)
head(pair_data)

summary(pair_data$stock)
sd(pair_data$stock, na.rm = TRUE)

summary(pair_data$index)
sd(pair_data$index, na.rm = TRUE)

cor(pair_data$stock, pair_data$index)
cor(pair_data$stock, pair_data$index, method = "spearman")

mu_hat <- colMeans(pair_data)
Sigma_hat <- var(pair_data)

mu_hat
Sigma_hat

par(mfrow = c(1, 2))
hist(pair_data$stock,
     main = paste("Histogram of", stock_name, "returns"),
     xlab = "Return")
hist(pair_data$index,
     main = paste("Histogram of", index_name, "returns"),
     xlab = "Return")

par(mfrow = c(1, 1))
plot(pair_data$stock, pair_data$index,
     main = paste(stock_name, "vs", index_name, "returns"),
     xlab = paste(stock_name, "return"),
     ylab = paste(index_name, "return"))
abline(lm(pair_data$index ~ pair_data$stock), col = "red")

par(mfrow = c(1, 2))
qqnorm(pair_data$stock,
       main = paste("QQ-plot of", stock_name, "returns"))
qqline(pair_data$stock)
qqnorm(pair_data$index,
       main = paste("QQ-plot of", index_name, "returns"))
qqline(pair_data$index)

par(mfrow = c(1, 1))  # final reset


# * **Histograms (mobil vs crsp)**
#   
#   * Both return series are tightly centered around 0, with small daily movements and few large jumps → typical daily stock/index returns.
# * Shapes are roughly bell–shaped but with slightly heavy tails (more extreme values than a perfect Normal).
# * mobil has a bit wider spread than crsp (sd ≈ 0.0129 vs 0.0078), so mobil is more volatile than the market index.
# 
# * **Scatterplot (mobil on y, crsp on x)**
#   
#   * Cloud of points is elongated along an upward–sloping line → **positive linear dependence**.
# * Pearson correlation ≈ 0.43, Spearman ≈ 0.40 → moderate positive association: when the market (crsp) is up, mobil tends to be up, but with noticeable noise.
# * Regression line has positive slope but many points scattered around it → index explains part, but not all, of mobil’s variability.
# 
# * **QQ-plots (marginal distributions)**
#   
#   * For both mobil and crsp, middle quantiles lie close to the straight line → central part of the distribution is close to Normal.
# * Tails curve away from the line (especially on extremes) → heavier tails / outliers compared to a Gaussian, i.e. occasional large gains/losses.
# 
# * **Qualitative joint (bivariate) probabilistic model**
#   
#   * A reasonable first model:
#   
#   * ((\text{mobil}, \text{crsp})) are **approximately bivariate Normal**, with means near 0, covariance matrix (\Sigma \approx \begin{pmatrix}1.67\cdot10^{-4} & 4.3\cdot10^{-5}\ 4.3\cdot10^{-5} & 6.0\cdot10^{-5}\end{pmatrix}).
# * Correlation around 0.4 → moderate market–beta: mobil moves with the market but retains substantial idiosyncratic risk.
# * In words: *daily mobil and crsp returns look roughly like a noisy linear relationship between two almost-Gaussian variables with moderate positive correlation and slightly heavier-than-Normal tails.*


###############################################
# Classwork-01 – Question 2
# Cramér–Wold idea via random projections
###############################################

set.seed(123)   # for reproducibility

## 1. Choose dimension p and sample size n
p <- 2          # small dimension (as suggested)
n <- 1000       # number of observations for each vector

## 2. Define two different multivariate distributions F_X and F_Y
##    Example:
##    X ~ N_p(0, I_p)    (mean 0, independent components)
##    Y ~ N_p(mu, I_p)   (shifted mean, independent components)

# Data from F_X
X <- matrix(rnorm(n * p, mean = 0, sd = 1), nrow = n, ncol = p)

# Data from F_Y (different mean)
mu_Y <- c(1, 1)   # for p = 2
Y <- matrix(NA, nrow = n, ncol = p)
for (j in 1:p) {
  Y[, j] <- rnorm(n, mean = mu_Y[j], sd = 1)
}

colnames(X) <- paste0("X", 1:p)
colnames(Y) <- paste0("Y", 1:p)

## 3. Function to generate a random unit vector gamma ~ something like Unif on [0,1]^p
gen_gamma <- function(p) {
  g <- runif(p, min = 0, max = 1)
  g / sqrt(sum(g^2))   # normalize to have norm 1
}

## 4. Choose how many random directions gamma to try
n_gamma <- 3

## 5. For each gamma: project X and Y, compare distributions
for (k in 1:n_gamma) {
  cat("=====================================\n")
  cat("Projection", k, "\n")
  
  # 5.1 Generate a random unit vector gamma
  gamma <- gen_gamma(p)
  cat("gamma =", gamma, "\n\n")
  
  # 5.2 Compute projections gamma^T X_i and gamma^T Y_i
  proj_X <- as.vector(X %*% gamma)
  proj_Y <- as.vector(Y %*% gamma)
  
  # 5.3 Numerical summaries of projections
  cat("Summary of gamma^T X:\n")
  print(summary(proj_X))
  cat("sd(gamma^T X) =", sd(proj_X), "\n\n")
  
  cat("Summary of gamma^T Y:\n")
  print(summary(proj_Y))
  cat("sd(gamma^T Y) =", sd(proj_Y), "\n\n")
  
  # 5.4 Numerical distance between the two projected distributions
  #     Here I use the two-sample Kolmogorov–Smirnov test
  ks_res <- ks.test(proj_X, proj_Y)
  cat("KS statistic:", ks_res$statistic, "\n")
  cat("KS p-value :", ks_res$p.value, "\n\n")
  
  # 5.5 Visual comparison via histograms
  par(mfrow = c(1, 2))
  
  hist(proj_X,
       main = paste("Projection", k, ": gamma^T X"),
       xlab = expression(gamma^T * X),
       col = "lightblue", border = "white")
  
  hist(proj_Y,
       main = paste("Projection", k, ": gamma^T Y"),
       xlab = expression(gamma^T * Y),
       col = "lightgreen", border = "white")
  
  par(mfrow = c(1, 1))
}


###############################################

# ---
#   
#   ### Results for Question 2 – Cramér–Wold device (simulation)
#   
#   * We generated two bivariate samples of size (n=1000):
#   
#   * (X \sim F_X = \mathcal{N}_2((0,0), I_2))
# * (Y \sim F_Y = \mathcal{N}_2((1,1), I_2))
# So the two distributions differ only in the mean vector.
# 
# * For each replication we drew a random unit vector (\gamma) in (\mathbb{R}^2) and looked at the one–dimensional projections
# (\gamma^\top X_i) and (\gamma^\top Y_i), (i = 1,\dots,n).
# 
# #### Projection 1
# 
# * Direction: (\gamma \approx (0.76, 0.65)).
# * Summaries:
#   
#   * (\gamma^\top X): mean ≈ 0.04, sd ≈ 1.04.
# * (\gamma^\top Y): mean ≈ 1.39, sd ≈ 1.01.
# * The two histograms are bell–shaped with very similar spread but the distribution of (\gamma^\top Y) is clearly shifted to the right by about 1 unit.
# * KS test: statistic ≈ 0.52, p-value ≈ 0 → we strongly reject that the two projected distributions are equal.
# 
# #### Projection 2
# 
# * Direction: (\gamma \approx (0.96, 0.29)).
# * Summaries:
#   
#   * (\gamma^\top X): mean ≈ 0.03, sd ≈ 1.02.
# * (\gamma^\top Y): mean ≈ 1.23, sd ≈ 0.99.
# * Again, both histograms look approximately normal with similar variability, but the distribution of (\gamma^\top Y) is shifted to larger values.
# * KS test: statistic ≈ 0.46, p-value ≈ 0 → projected distributions differ also along this direction.
# 
# #### Projection 3
# 
# * Direction: (\gamma \approx (0.67, 0.74)).
# * Summaries:
#   
#   * (\gamma^\top X): mean ≈ 0.04, sd ≈ 1.04.
# * (\gamma^\top Y): mean ≈ 1.39, sd ≈ 1.01.
# * The histograms (shown in the figure) confirm the same pattern: both projections are roughly symmetric and bell–shaped, but (\gamma^\top Y) is centered around ≈1.4 while (\gamma^\top X) is centered around 0.
# * KS test: statistic ≈ 0.52, p-value ≈ 0 → strong evidence that the one–dimensional distributions are not the same.
# 
# ### Interpretation (link with Cramér–Wold)
# 
# * In all three randomly chosen directions (\gamma), the distributions of (\gamma^\top X) and (\gamma^\top Y) look different: they have almost the same spread but clearly different centers.
# * The Kolmogorov–Smirnov tests always give very large statistics and p-values essentially equal to zero, confirming that the projected distributions are different.
# * This empirical result is exactly what the “reverse” use of the Cramér–Wold device suggests: since the underlying multivariate distributions (F_X) and (F_Y) are different (they have different means), then with probability 1 there exist directions (\gamma) for which the 1D projections have different distributions.
# * Our simulation shows that even with a few random directions in low dimension (p = 2) these differences are easy to detect both visually (histograms) and numerically (KS test).
# 
###############################################


###############################################
# Classwork-01 – Question 3 (Bonus: MIC)
# Explore MIC(X, Y) for the construction:
# X ~ Unif(-1, 1)
# Z ~ Unif(-1, 1) independent of X
# Y = Z  if X and Z have the same sign
# Y = -Z otherwise
###############################################

# install.packages("minerva")


set.seed(123)   # for reproducibility

## 1. Sample size
n <- 5000

## 2. Generate X and Z ~ Unif(-1, 1), independent
X <- runif(n, min = -1, max = 1)
Z <- runif(n, min = -1, max = 1)

## 3. Build Y according to the rule in the text
##    Y = Z if sign(X) == sign(Z), else Y = -Z
same_sign <- (X >= 0 & Z >= 0) | (X < 0 & Z < 0)

Y <- ifelse(same_sign, Z, -Z)

## Quick check
head(cbind(X, Z, Y))

###############################################
# 4. Basic summaries and classical correlations
###############################################

summary(X)
summary(Y)

## Pearson correlation (linear dependence)
cor_pearson <- cor(X, Y)
cor_pearson

## Spearman correlation (rank-based, monotone)
cor_spearman <- cor(X, Y, method = "spearman")
cor_spearman

###############################################
# 5. Plots to see the structure of (X, Y)
###############################################

## 5.1 Scatterplot of X vs Y
plot(X, Y,
     main = "Scatterplot of X vs Y",
     xlab = "X",
     ylab = "Y",
     pch = 19, cex = 0.5)

## 5.2 Histograms of X and Y
par(mfrow = c(1, 2))

hist(X,
     main = "Histogram of X ~ Unif(-1, 1)",
     xlab = "X",
     col = "lightblue", border = "white")

hist(Y,
     main = "Histogram of Y (constructed from X, Z)",
     xlab = "Y",
     col = "lightgreen", border = "white")

par(mfrow = c(1, 1))  # reset layout

###############################################
# 6. MIC(X, Y) using the minerva package
###############################################

## If you want MIC, you need the 'minerva' package.
## Install it once with:
## install.packages("minerva")

if (!requireNamespace("minerva", quietly = TRUE)) {
  mic_value <- NA
  warning("Package 'minerva' is not installed. Run install.packages('minerva') to compute MIC.")
} else {
  library(minerva)
  
  ## mine() can take two vectors X and Y
  mic_res <- mine(X, Y)
  
  ## The MIC value is stored in mic_res$MIC
  mic_value <- mic_res$MIC
}

###############################################
# 7. Print all measures together
###############################################

cat("Pearson correlation:", cor_pearson, "\n")
cat("Spearman correlation:", cor_spearman, "\n")
cat("MIC(X, Y):", mic_value, "\n")

###############################################

#  
#  ---
#  
#  ### Description of the simulation
#  
#  * We generated
#
#* (X \sim \text{Unif}(-1,1))
#* (Z \sim \text{Unif}(-1,1)), independent of (X).
#* We then defined
#
#* (Y = Z) if (X) and (Z) have the **same sign**,
#* (Y = -Z) otherwise.
#
#So (X) and (Y) are not independent: they are constructed to have the **same sign** almost surely.
#
#---
#  
#  ### Marginal behaviour (histograms of X and Y)
#  
#  * Both histograms are almost flat over ([-1,1]), confirming that **marginally**
#  
#  * (X \approx \text{Unif}(-1,1))
#* (Y \approx \text{Unif}(-1,1)).
#* Means are very close to 0 and the ranges almost identical, so looking only at marginals one might (wrongly) think the variables are independent.
#
#---
#  
#  ### Joint behaviour (scatterplot of X vs Y)
#  
#  * The scatterplot shows **two filled rectangles**:
#  
#  * one in the bottom–left quadrant ((X<0, Y<0)),
#* one in the top–right quadrant ((X>0, Y>0)),
#* with **no points** in the top–left and bottom–right quadrants.
#* This reflects the construction: (X) and (Y) always share the **same sign**, but within each quadrant the points look roughly uniform.
#* The dependence is therefore **strong but non-linear**: there is no straight line or smooth monotone curve; the support is “L-shaped / block-shaped” rather than elliptical.
#
#---
#  
#  ### Numerical measures of dependence
#  
#  * **Pearson correlation**: (\rho \approx 0.74)
#→ indicates a **strong positive linear association**, but still < 1 because the relation is not purely linear.
#* **Spearman correlation**: (\rho_S \approx 0.75)
#→ confirms a strong **monotone** association in ranks, again not perfect.
#* **MIC(X, Y) = 1**
#  → the Maximal Information Coefficient detects that the pair ((X,Y)) has an almost **perfect functional-type dependence** (the joint pattern is highly structured), even though it is not linear.
#
#---
#  
#  ### Qualitative probabilistic model for (X, Y)
#  
#  * A simple way to describe the joint law is:
#  
#  * (X \sim \text{Unif}(-1,1));
#* conditional on the sign of (X),
#[
#  Y \mid X = x \sim \text{Unif}(0,1)\cdot \text{sign}(x),
#]
#i.e. (Y) is uniform on ((0,1)) if (X>0) and uniform on ((-1,0)) if (X<0).
#
#* So:
#  
#  * **marginals**: both approximately Uniform((-1,1));
#* **joint**: concentrated only on the two diagonal quadrants, with strong, nonlinear positive dependence.
#
#* This example shows that:
#  
#  * classical correlations (Pearson/Spearman) see a strong but not perfect association;
#* MIC correctly reports **maximal dependence** (value 1), capturing the complex, non-linear structure in the scatterplot.
#

###############################################

