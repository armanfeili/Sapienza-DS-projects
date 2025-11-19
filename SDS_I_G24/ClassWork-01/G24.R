# Classwork-01 
# Team: G24
# Team members: Arman Feili, Sohrab Seyyedi Parsa, Milad Torabi


# -----

# Q1)

# Load the data
grsp <- read.csv("CRSPday.csv")

# Pick two variables: GE and IBM returns
x <- grsp$ge
y <- grsp$ibm

# Create a clean data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

# Look at summaries
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plot histograms
hist(data$stock, main = "GE returns", xlab = "GE")
hist(data$index, main = "IBM returns", xlab = "IBM")

# The histograms show that both GE and IBM returns are centered around 0 and look pretty symmetric. IBM's returns are more spread out, so it's more volatile.

# Scatterplot with regression line
plot(data$stock, data$index,
     main = "GE vs IBM", xlab = "GE", ylab = "IBM")
abline(lm(index ~ stock, data = data), col = "red")

# The scatterplot shows a cloud of points going upwards, meaning GE and IBM tend to move in the same direction — but the points are scattered, so the correlation is only moderate.

# QQ-plots to check normality
qqnorm(data$stock); qqline(data$stock)
qqnorm(data$index); qqline(data$index)

# The QQ-plots look normal in the center but curve off in the tails, suggesting more extreme returns than a perfect normal distribution.

# Both return series are roughly normal with heavier tails, and they’re moderately positively correlated. A bivariate normal model fits okay for the bulk, but not the extremes.

# ---

# ge vs mobil

# Load the data
grsp <- read.csv("CRSPday.csv")

# Pick two variables: GE and Mobil returns
x <- grsp$ge
y <- grsp$mobil

# Create a clean data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

# Look at summaries
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plot histograms
hist(data$stock, main = "GE returns", xlab = "GE")
hist(data$index, main = "Mobil returns", xlab = "Mobil")

# The histograms show that both GE and Mobil returns are centered around 0 and look pretty symmetric. Their spreads are fairly similar, so the volatilities are comparable, with a few larger positive and negative days in the tails.

# Scatterplot with regression line
plot(data$stock, data$index,
     main = "GE vs Mobil", xlab = "GE", ylab = "Mobil")
abline(lm(index ~ stock, data = data), col = "red")

# The scatterplot shows a cloud of points going upwards, meaning GE and Mobil tend to move in the same direction — but the points are quite scattered, so the correlation is only moderate.

# QQ-plots to check normality
qqnorm(data$stock); qqline(data$stock)
qqnorm(data$index); qqline(data$index)

# The QQ-plots look normal in the center but curve off in the tails, suggesting more extreme returns than a perfect normal distribution.

# Both return series are roughly normal with heavier tails, and they’re moderately positively correlated. A bivariate normal model fits okay for the bulk, but not the extremes.

# ---

# ge vs crsp

# Pick two variables: GE and CRSP index returns
x <- grsp$ge
y <- grsp$crsp

# Create a clean data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

# Look at summaries
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plot histograms
hist(data$stock, main = "GE returns", xlab = "GE")
hist(data$index, main = "CRSP returns", xlab = "CRSP")

# The histograms show that both GE and CRSP returns are centered around 0 and look pretty symmetric. GE’s histogram is more spread out, so GE is more volatile than the market index.

# Scatterplot with regression line
plot(data$stock, data$index,
     main = "GE vs CRSP", xlab = "GE", ylab = "CRSP")
abline(lm(index ~ stock, data = data), col = "red")

# The scatterplot shows a tight cloud of points going upwards, meaning GE and the market index move strongly in the same direction, with a fairly high positive correlation.

# QQ-plots to check normality
qqnorm(data$stock); qqline(data$stock)
qqnorm(data$index); qqline(data$index)

# The QQ-plots look normal in the center but bend away in the tails, again suggesting heavier tails and more extreme returns than a perfect normal.

# Both return series are roughly normal with heavier tails, and they’re strongly positively correlated. A bivariate normal model is a reasonable description for day-to-day movements, but it underestimates the extremes.

# ---

# ibm vs mobil

# Pick two variables: IBM and Mobil returns
x <- grsp$ibm
y <- grsp$mobil

# Create a clean data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

# Look at summaries
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plot histograms
hist(data$stock, main = "IBM returns", xlab = "IBM")
hist(data$index, main = "Mobil returns", xlab = "Mobil")

# The histograms show that both IBM and Mobil returns are centered around 0 and look roughly symmetric. IBM’s returns are more spread out, so IBM is more volatile than Mobil.

# Scatterplot with regression line
plot(data$stock, data$index,
     main = "IBM vs Mobil", xlab = "IBM", ylab = "Mobil")
abline(lm(index ~ stock, data = data), col = "red")

# The scatterplot shows a fairly round cloud of points with a slight upward trend, so IBM and Mobil tend to move in the same direction, but the correlation is quite weak.

# QQ-plots to check normality
qqnorm(data$stock); qqline(data$stock)
qqnorm(data$index); qqline(data$index)

# The QQ-plots look close to normal in the center but deviate in the tails, especially for IBM, indicating heavier tails and more extreme returns than a normal model.

# Both return series are roughly normal with heavier tails, and they’re only weakly positively correlated. A bivariate normal model gives a rough description, but it does not capture the rare large moves well.

# ---

# ibm vs crsp

# Pick two variables: IBM and CRSP index returns
x <- grsp$ibm
y <- grsp$crsp

# Create a clean data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

# Look at summaries
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plot histograms
hist(data$stock, main = "IBM returns", xlab = "IBM")
hist(data$index, main = "CRSP returns", xlab = "CRSP")

# The histograms show that both IBM and CRSP returns are centered around 0 and look pretty symmetric. IBM’s returns are more spread out, so IBM is more volatile than the overall market.

# Scatterplot with regression line
plot(data$stock, data$index,
     main = "IBM vs CRSP", xlab = "IBM", ylab = "CRSP")
abline(lm(index ~ stock, data = data), col = "red")

# The scatterplot shows an elongated cloud of points going upwards, meaning IBM and the market index move together with a clear positive relationship, though there is still noticeable scatter.

# QQ-plots to check normality
qqnorm(data$stock); qqline(data$stock)
qqnorm(data$index); qqline(data$index)

# The QQ-plots look approximately normal in the middle but bend away in the tails, which points to heavier tails and more extreme events than a perfect Gaussian model.

# Both return series are roughly normal with heavier tails, and they’re moderately to strongly positively correlated. A bivariate normal model is a reasonable first approximation for the joint behaviour, but it misses the tail risk.

# ---

# mobil vs crsp

# Pick two variables: Mobil and CRSP index returns
x <- grsp$mobil
y <- grsp$crsp

# Create a clean data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

# Look at summaries
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plot histograms
hist(data$stock, main = "Mobil returns", xlab = "Mobil")
hist(data$index, main = "CRSP returns", xlab = "CRSP")

# The histograms show that both Mobil and CRSP returns are centered around 0 and look roughly symmetric. Mobil’s histogram is wider, so Mobil is more volatile than the market index.

# Scatterplot with regression line
plot(data$stock, data$index,
     main = "Mobil vs CRSP", xlab = "Mobil", ylab = "CRSP")
abline(lm(index ~ stock, data = data), col = "red")

# The scatterplot shows a cloud of points tilted upwards, meaning Mobil and the market tend to move in the same direction, with a moderate positive correlation and some dispersion around the line.

# QQ-plots to check normality
qqnorm(data$stock); qqline(data$stock)
qqnorm(data$index); qqline(data$index)

# The QQ-plots look normal in the central part but curve away in the tails, indicating more extreme returns than predicted by a perfect normal distribution.

# Both return series are roughly normal with heavier tails, and they’re moderately positively correlated. A bivariate normal model gives a reasonable description for ordinary days, but it underestimates joint extreme moves.




# Q2)

set.seed(123)

# 1) We pick the dimension (2D points) and how many points we simulate
p <- 2
n <- 1000

# 2) We create two different 2D clouds of points:
#    X is centered around (0,0), Y is centered around (1,1)
X1 <- rnorm(n, mean = 0, sd = 1)
X2 <- rnorm(n, mean = 0, sd = 1)
X  <- cbind(X1, X2)

Y1 <- rnorm(n, mean = 1, sd = 1)
Y2 <- rnorm(n, mean = 1, sd = 1)
Y  <- cbind(Y1, Y2)

# 3) We choose a random direction in 2D and rescale it to have length 1
gamma <- runif(p)
gamma <- gamma / sqrt(sum(gamma^2))

gamma

# 4) We project all points of X and Y on this direction
proj_X <- as.vector(X %*% gamma)
proj_Y <- as.vector(Y %*% gamma)

# 5) We look at basic summaries of the two 1D samples
summary(proj_X)
summary(proj_Y)

# Numerical check: test if the two 1D samples look like they come
# from the same distribution
ks.test(proj_X, proj_Y)

# Visual check: compare the two histograms side by side
par(mfrow = c(1, 2))
hist(proj_X, main = "Projection of X", xlab = "values")
hist(proj_Y, main = "Projection of Y", xlab = "values")
par(mfrow = c(1, 1))

# From the summaries and the KS test, we see that the projected values
# for Y are shifted to the right compared to X, and the p-value is
# basically zero. So, even after reducing everything to 1D along this
# random direction, the two samples still look clearly different.
# This is exactly the idea behind Cramér–Wold: if the original
# multivariate distributions are not the same, then their 1D
# projections will also not match along (almost) any direction.


# ---

# Q3)

# install.packages("minerva")

set.seed(123)

# 1) Number of points to simulate
n <- 5000

# 2) generating X and Z as independent uniforms on [-1, 1]
X <- runif(n, min = -1, max = 1)
Z <- runif(n, min = -1, max = 1)

# We build Y:
# if X and Z have the same sign,  Y = Z
# otherwise, Y = -Z
same_sign <- (X >= 0 & Z >= 0) | (X < 0 & Z < 0)
Y <- ifelse(same_sign, Z, -Z)

head(cbind(X, Z, Y))

summary(X)
summary(Y)

# Pearson (usual linear correlation)
cor_pearson <- cor(X, Y)

# Spearman (based on ranks)
cor_spearman <- cor(X, Y, method = "spearman")

cor_pearson
cor_spearman

# plot X vs Y to see the shape of the relationship
plot(X, Y,
     main = "Scatterplot of X vs Y",
     xlab = "X",
     ylab = "Y",
     pch = 19, cex = 0.4)

# histograms of X and Y
par(mfrow = c(1, 2))
hist(X, main = "Histogram of X", xlab = "X")
hist(Y, main = "Histogram of Y", xlab = "Y")
par(mfrow = c(1, 1))

# computing MIC(X, Y) if the minerva package is available, we use mine() from that package
if (!requireNamespace("minerva", quietly = TRUE)) {
  mic_value <- NA
  warning("Package 'minerva' is not installed. Run install.packages('minerva') to get MIC.")
} else {
  library(minerva)
  mic_res  <- mine(X, Y)
  mic_value <- mic_res$MIC
}

# printing all the dependence measures together
cat("Pearson correlation:", cor_pearson, "\n")
cat("Spearman correlation:", cor_spearman, "\n")
cat("MIC(X, Y):", mic_value, "\n")

# From the plots and the numbers we see:
# - X and Y both look roughly uniform on [-1, 1] on their own.
# - In the scatterplot, points only appear in the top-right and
#   bottom-left quadrants → X and Y almost always have the same sign.
# - Pearson and Spearman are clearly positive but < 1.
# - MIC is (almost) 1, which tells us that the dependence between X and Y
#   is very strong and highly structured, even if it is not just a straight line.


