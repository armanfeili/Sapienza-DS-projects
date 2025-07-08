# Linear Diffusion Models with KernelPCA, PCA, and Conformal Prediction

**Statistical Learning – Homework Project**
**Professor:** Pierpaolo Brutti
**Student:** Arman Feili (Matricola: 2101835, [feili.2101835@studenti.uniroma1.it](mailto:feili.2101835@studenti.uniroma1.it))

---

## Overview

This Colab notebook implements and tests linear and nonlinear diffusion models for MNIST digit generation, combining PCA, KernelPCA (RBF, polynomial, sigmoid), and conformal prediction for uncertainty quantification. The project explores how different kernels, latent sizes, noise, and diffusion schedules affect classification accuracy and reliability, with a focus on logistic regression.

---

## 1. Environment Setup

* Installs specific versions of `numpy`, `scipy`, and `scikit-learn` for compatibility.
* Patches numpy internals to fix missing modules in Colab.
* Mounts Google Drive for persistent storage.
* Clones the `linear-diffusion` repository if not already present.
* Sets up all configuration variables for experiment control.

---

## 2. Data Preparation

* Loads the MNIST dataset.
* Splits data into train, calibration, and test sets using stratified sampling for balanced digit representation.
* Flattens images and prepares balanced test sets for fair evaluation.

---

## 3. Model and Experiment Configuration

* Experiments sweep over:

  * **Latent sizes:** \[8, 10, 12]
  * **Noise levels (`std`):** \[1.0, 1.5, 2.0]
  * **Diffusion steps (`T`):** \[400, 1000]
  * **Kernels:** PCA, KernelPCA (RBF with several gammas, poly with degree/coef0, sigmoid)
* **Logistic Regression** is the main classifier; SVM and Random Forest are available but usually not enabled due to lower performance.

---

## 4. Diffusion Model

* `LinearDiffusionNonlinear` supports both PCA and KernelPCA as latent encoders, including label interaction features and noise injection.
* Models are trained using Ridge regression to map features to latent codes.
* Image generation supports both deterministic (single-step) and diffusion-style (multi-step, cosine or strided noise schedule) sampling.

---

## 5. Helper Functions

* Utilities for plotting generated digits, comparing real vs generated images, sampling balanced digits, and evaluating classifier and conformal prediction performance.
* Efficient caching for encoders/decoders to save RAM and computation time.

---

## 6. Evaluation Pipeline

* For each configuration:

  * Trains the diffusion model (PCA or KernelPCA).
  * Generates digits and classifies them using the selected classifier.
  * Computes accuracy, precision, recall, F1-score, and confusion matrices.
  * Applies conformal prediction to measure coverage (fraction of true labels within prediction sets) and average set size (confidence).
  * Plots and saves results for visual inspection.

---

## 7. Results and Key Findings

* **Polynomial KernelPCA (degree=2, coef0=1, std=1.0, latent=10/12):**

  * Achieves the highest accuracy (\~0.99), coverage (\~0.97), and lowest set size (close to 1).
* **RBF KernelPCA:**

  * Decent accuracy (0.86–0.93) but poor coverage (around 0.53) due to overconfidence.
* **PCA:**

  * Middle performance; lower accuracy and coverage compared to polynomial kernels.
* **Latent Size:** 10 is optimal for polynomial kernels; larger sizes add little.
* **Noise:** Higher noise sharply reduces accuracy and coverage.
* **Diffusion Steps:** Raising T from 400 to 1000 has little effect.
* **All models generate digits in under 2 seconds per configuration.**
* **Best results:** High accuracy, high coverage, low set size, clear digits.
* **Worst results:** Low coverage (especially for RBF and sigmoid), blurred digits at high noise, or overconfident predictions.

---

## 8. Conformal Prediction Summary

* Coverage close to 1.0 indicates reliable uncertainty estimates.
* Average set size near 1.0 is ideal if coverage remains high.
* Per-digit results and tables allow detailed analysis of where models perform well or fail.

---

## 9. How GPT Helped

* Debugged shape and feature size mismatches.
* Advised on memory management and modular coding.
* Suggested experiment tuning (kernel/gamma/latent).
* Explained results and metrics for effective presentation.
* Helped optimize the workflow for efficiency and flexibility.

---

## 10. Self-Assessment

* The model achieves strong performance with the right kernel settings.
* Conformal prediction is well integrated, offering robust, interpretable intervals.
* Results are fast, accurate, and reliable, with code ready for further research or deployment.

---

## How to Run

1. Open the Colab notebook.
2. Run the setup and package installation cells.
3. Follow the experiment configuration to adjust latent sizes, kernels, or noise as desired.
4. Run all cells to reproduce the full sweep of experiments.
5. Inspect results, summary tables, and generated digit plots.

---

## Citation

Project by Arman Feili, MSc. Data Science
Department of Statistical Sciences, Sapienza University of Rome
Academic Year 2024/2025
Email: [feili.2101835@studenti.uniroma1.it](mailto:feili.2101835@studenti.uniroma1.it)
