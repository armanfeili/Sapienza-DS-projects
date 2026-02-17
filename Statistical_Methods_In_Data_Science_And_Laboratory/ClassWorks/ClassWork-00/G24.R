# G21
# Team Leader: Arman Feili
# Team Members: Milad Torabi, Sohrab Seyyedi Parsa 

# Q1


data <- read.csv("./Country-data/Country-data.csv")

x <- data$life_expec

summary(x)

hist(x,
     breaks = 20,
     main = "life Expectancy",
     xlab = "Years"
     )

qqnorm(x)
qqline(x, col="red")

mean_x <- mean(x)
mean_x

sd_x <- sd(x)
sd_x

# Comments: If the distribution was perfectly normal, all dots should have been on the Red line.
# Right now dots are often on the line, so the distribution is closed to the gaussian.
# Whenever they're not on the line, it suggests the tails


# Q2 - Wavelet Coefficients Analysis

# Load libraries
library(jpeg)
library(waveslim)

# Load grayscale image
img <- readJPEG("./grayscale_images/Lenna.jpg")

####
# We asked the chatbot how to convert an RGB image to grayscale and how to pad an image so its size becomes a power of 2 for the wavelet transform.
####

# If image is RGB, convert to grayscale
if (length(dim(img)) == 3) {
  img <- 0.299 * img[,,1] + 0.587 * img[,,2] + 0.114 * img[,,3]
}

# Pad image so both dimensions are powers of 2 (needed for wavelets)
pad_to_pow2 <- function(img) {
  nr <- nrow(img)
  nc <- ncol(img)
  nr2 <- 2^ceiling(log2(nr))
  nc2 <- 2^ceiling(log2(nc))
  padded <- matrix(0, nr2, nc2)
  padded[1:nr, 1:nc] <- img
  return(padded)
}

X <- pad_to_pow2(img)
####
# We asked the chatbot how to apply a 2D Haar wavelet transform in R and how to extract the coefficients (like LH2, HL2, or HH2) from the result.
####
# Apply 2D Haar wavelet transform (3 levels)
dwt <- dwt.2d(X, wf = "haar", J = 3, boundary = "periodic")

# Extract detail coefficients at level 2 (try LH2, else HL2 or HH2)
get_subband <- function(dwt, level = 2) {
  for (band in c("LH", "HL", "HH")) {
    name <- paste0(band, level)
    if (name %in% names(dwt)) {
      return(as.matrix(dwt[[name]]))
    }
  }
  stop("Level 2 detail subband not found.")
}

coef <- get_subband(dwt)


####
# Here we asked gpt that how to turn a matrix of wavelet coefficients to numeric vector and plot the distribution to check if it is Gaussian.
####
# Flatten matrix to vector, remove NaN/Inf
vec_clean <- function(mat) {
  v <- as.numeric(mat)
  v[is.finite(v)]
}

# Plot histogram and QQ-plot of raw coefficients
par(mfrow = c(2,2))
hist(vec_clean(coef), breaks = 50, col = "lightblue", main = "Raw Coeffs")
qqnorm(vec_clean(coef)); qqline(vec_clean(coef), col = "red")

####
# Results:
# Histogram: It shows the distribution of the raw wavelet coefficients from the image. Most of the values are very close to zero showing wavelet detail coefficients. they capture edges and noise, but most areas in an image are smooth with small values.
# Q-Q plot: checks if those coefficients follow a normal (Gaussian) distribution. The points curve away from the red line, especially at the tails, which means the distribution is not normal — it's more peaked in the center and has heavier tails than a Gaussian.
####

####
# We asked gpt to see how to normalize a wavelet coefficient matrix block-by-block using local RMS, like it's done in GSM models.
# Then we wanted to see if the normalization makes the distribution more Gaussian, so we tried to plot a histogram and Q-Q plot after normalization.
####

# Normalize using local RMS per 8x8 block
normalize_blocks <- function(mat, block = 8) {
  out <- mat
  for (i in seq(1, nrow(mat), by = block)) {
    for (j in seq(1, ncol(mat), by = block)) {
      blk <- mat[i:min(i+block-1, nrow(mat)), j:min(j+block-1, ncol(mat))]
      scale <- sqrt(mean(blk^2)) + 1e-6  # avoid dividing by zero
      out[i:min(i+block-1, nrow(mat)), j:min(j+block-1, ncol(mat))] <- blk / scale
    }
  }
  return(out)
}

coef_norm <- normalize_blocks(coef)

# Plot histogram and QQ-plot after normalization
par(mfrow = c(2,2))
hist(vec_clean(coef_norm), breaks = 50, col = "lightgreen", main = "Normalized Coeffs")
qqnorm(vec_clean(coef_norm)); qqline(vec_clean(coef_norm), col = "red")

###
# Result: After normalization (like in GSM models), the distribution is sharp at zero but now has a wider range (from -6 to 6). This means the normalization shows local contrast and make the data not look Gaussian.
# The Q-Q plot still bends strongly at the ends (tails), so the distribution is still not Gaussian. In fact, the normalization made the tails even heavier — this is common when normalizing wavelet coefficients block by block.
####
# What we asked from chatbot was that how to check if there’s correlation between wavelet coefficients and their neighbors, before and after normalization.
####

# Scatter plots: original vs. neighbor to the right
par(mfrow = c(1,2))
if (ncol(coef) > 1) {
  plot(coef[, -ncol(coef)], coef[, -1], pch = 16, cex = 0.3,
       xlab = "coef(i,j)", ylab = "coef(i,j+1)", main = "Neighbor (raw)")
}
if (ncol(coef_norm) > 1) {
  plot(coef_norm[, -ncol(coef_norm)], coef_norm[, -1], pch = 16, cex = 0.3,
       xlab = "nu(i,j)", ylab = "nu(i,j+1)", main = "Neighbor (normalized)")
}

# Each point shows a pair of neighboring wavelet coefficients (one pixel and the one to its right). The points form a vertical ellipse, meaning there is a positive correlation between neighbors: if one coefficient is large, the next one is likely large too — but there's still a lot of noise.
# After block-wise RMS normalization, the correlation between neighboring coefficients is still there, but the values are now more spread out, especially vertically. This suggests the normalization increased variability while keeping some dependency between neighbors.






?dunif

set.seed(3435)
M <- 1000

Xsamp <- runif(M, min = -1, max = +1)
Ysamp = (Xsamp)^2

# density function for Y = X^2, X ~ U(-1,1)
dtrans <- function(y) 1 / (2 * sqrt(y))

par(mfrow = c(2,1))

# Plot for X
hist(Xsamp, prob=TRUE, border="white", col="orange2",
     main = "Samples from Unif(-1,+1)", xlab="x", breaks = 25)
curve(dunif(x, -1,+1), add = TRUE, lwd = 4, col = "red3")
rug(Xsamp, col = rgb(0,0,0,.3))

# Plot for Y
hist(Ysamp, prob=TRUE, border="white", col="blue",
     main = "Samples from Y = X^2", xlab="y", breaks = 25)
curve(dtrans(x), from=0.001, to=1, add = TRUE, lwd = 4, col = "red3")  # Avoid sqrt(0)
rug(Ysamp, col = rgb(0,0,0,.3))



# Student-t

# Stochastic representation:

# T = Z / sqrt(W/nu)

# Where Z ~ N(0,1)
#       W ~ Chi2(nu)
#       z ind W

# Pick nu
nu <- 2

?rnorm
?rchisq

set.seed(4321)
M <- 5000
Zsamp <- rnorm(M, 0, 1)
Wsamp <- rchisq(M, df = nu)
Tsamp <- Zsamp/sqrt(Wsamp/nu)

# Plot 
Tsamp_sel <- Tsamp[abs(Tsamp) < 20]
hist(Tsamp_sel, prob=TRUE, border="white", col="navy",
     main = "Samples from a Student-t?",
     sub = paste("Degrees of Freedom = ", nu),
     xlab="y", breaks = 50)
curve(dt(x, df = nu), add = T, lwd = 4, col = "cyan3")  # Avoid sqrt(0)
# rug(Tsamp, col = rgb(0,0,0,.3))
rug(Tsamp)
box()
