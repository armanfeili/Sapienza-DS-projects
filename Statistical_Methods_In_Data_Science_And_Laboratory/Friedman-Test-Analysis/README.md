# Statistical Methods: Friedman Test & Bayes Classification

This project is a comprehensive analysis performed as part of the **Statistical Methods in Data Science** course ("Stat4DS"). It covers fundamental classification concepts, the derivation of optimal decision rules, and the implementation of non-parametric two-sample tests for multivariate distributions.

## Project Overview

The analysis is divided into three main parts, all contained within `Friedman-Test-Analysis.Rmd`:

### Part 1: Bayes Classification vs. Logistic Regression
-   **Theoretical Derivation**: Calculation of the optimal **Bayes Classification Rule** $\eta^\star(x)$ for a specific 1D data generating process involving uniform distributions.
-   **Simulation Study**: Generation of synthetic data to validate the theoretical Bayes rule.
-   **Comparison**: Implementation of a **Logistic Regression** classifier to compare its performance against the optimal Bayes benchmark.
-   **Error Analysis**: Monte Carlo simulation (10,000 runs) to estimate the true misclassification rates of both classifiers.

### Part 2 & 3: Friedman’s Two-Sample Test
-   **Concept**: Implementation of **Friedman’s Two-Sample Test**, a method to test the hypothesis $H_0: F_X = F_Y$ vs $H_1: F_X \neq F_Y$ for multivariate distributions without assuming normality.
-   **Methodology**:
    1.  Train a binary classifier (e.g., Logistic Regression) to distinguish between Sample X and Sample Y.
    2.  Compute the classifier's "scores" (predicted probabilities) for the held-out/training data.
    3.  Perform a univariate two-sample test (Mann-Whitney U or Kolmogorov-Smirnov) on these scores to detect distributional differences.
-   **Power Analysis**: A simulation study to analyze the **size** (Type I error) and **power** (Type II error) of the test under various scenarios, tweaking the distance between the two multivariate distributions.

## Files

-   **`Friedman-Test-Analysis.Rmd`**: The main R Markdown file containing all code, derivations, simulations, and explanatory text.
-   **`Friedman-Test-Analysis.html`**: (Generated) The compiled HTML report of the analysis.

## Key Technologies

-   **R**
-   **Key Libraries**: `ggplot2`, `dplyr`, `tidyr`, `gridExtra`, `knitr`.

## Author

**Arman Feili**
-   Student ID: 2101835
-   Email: feili.2101835@studenti.uniroma1.it
