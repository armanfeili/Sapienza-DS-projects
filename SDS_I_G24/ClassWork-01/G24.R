# Classwork-01 
# Team: G24
# Team members: Arman Feili, Sohrab Seyyedi Parsa, Milad Torabi


# -----

# Q1)

# Load the data
grsp <- read.csv("CRSPday.csv")

# Pick two variables: GE and IBM
x <- grsp$ge
y <- grsp$ibm

# Create a data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

 
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plotting
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

# Pick two variables: GE and Mobil
x <- grsp$ge
y <- grsp$mobil

# Create a data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plotting
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

# Pick two variables: GE and CRSP index
x <- grsp$ge
y <- grsp$crsp

# Create a data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

 
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plotting
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

# Pick two variables: IBM and Mobil
x <- grsp$ibm
y <- grsp$mobil

# Create a data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

 
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plotting
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

# Pick two variables: IBM and CRSP index
x <- grsp$ibm
y <- grsp$crsp

# Create a data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

 
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plotting
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

# Pick two variables: Mobil and CRSP index
x <- grsp$mobil
y <- grsp$crsp

# Create a data frame with no missing values
data <- na.omit(data.frame(stock = x, index = y))

 
summary(data)
cor(data$stock, data$index)

# Estimate mean and covariance
colMeans(data)
var(data)

# Plotting
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

# Picking the dimension (2D points) and the number of simulatioin points
p <- 2
n <- 1000

# creating two different 2D clouds of points:
# X is centered around (0,0)
X1 <- rnorm(n, mean = 0, sd = 1)
X2 <- rnorm(n, mean = 0, sd = 1)
X  <- cbind(X1, X2)

# Y is centered around (1,1)
Y1 <- rnorm(n, mean = 1, sd = 1)
Y2 <- rnorm(n, mean = 1, sd = 1)
Y  <- cbind(Y1, Y2)

# We choose a random direction in 2D and rescale it to have length 1
gamma <- runif(p)
gamma <- gamma / sqrt(sum(gamma^2))

gamma

# projecting all points of X and Y on this direction
proj_X <- as.vector(X %*% gamma)
proj_Y <- as.vector(Y %*% gamma)

summary(proj_X)
summary(proj_Y)

# Numerical check: test if the two 1D samples look like they come
# from the same distribution
ks.test(proj_X, proj_Y)

# comparing the two histograms:
par(mfrow = c(1, 2))
hist(proj_X, main = "Projection of X", xlab = "values")
hist(proj_Y, main = "Projection of Y", xlab = "values")
par(mfrow = c(1, 1))

# When we look at the summaries and the KS test, we see that the projected values of Y are clearly shifted to the right compared to X.
# The KS p‑value is basically zero, so the two projected samples definitely do not come from the same distribution.
# Even if we reduce everything to just one dimension, using a random direction, it again happens.
# That's why it is based on the Cramér–Wold idea: if two multivariate distributions are different, then their 1‑D projections will also look different in almost any direction.


# ---

# Q3)

# install.packages("minerva")

set.seed(123)

# 1) Number of points to simulate
n <- 5000

# 2) generating X and Z as independent uniforms on [-1, 1]
X <- runif(n, min = -1, max = 1)
Z <- runif(n, min = -1, max = 1)

# building Y:
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

# library(minerva)
# mic_res  <- mine(X, Y)
# mic_value <- mic_res$MIC

# printing all the dependence measures together
cat("Pearson correlation:", cor_pearson, "\n")
cat("Spearman correlation:", cor_spearman, "\n")
cat("MIC(X, Y):", mic_value, "\n")

# Both X and Y, separately seem that they come from a uniform distribution on the interval [−1,1].
# In the scatterplot, all the points are either in the top-right or bottom-left corner,
# it means X and Y almost always have the same sign.
# The Pearson and Spearman correlations are clearly positive, but still less than 1
# so there's strong connection, but not perfectly linear or monotonic.
# The MIC (Maximal Information Coefficient) is basically 1, 
# showing that there's a very strong dependence between X and Y.

