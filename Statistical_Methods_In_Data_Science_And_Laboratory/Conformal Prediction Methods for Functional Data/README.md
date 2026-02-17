# Conformal Prediction Methods for Functional Data

This project implements **Conformal Prediction** methods for functional data analysis, specifically applied to accelerometer data collected via smartphone. The methodology follows the approach described by **Lei, Rinaldo, and Wasserman** in *"A Conformal Prediction Approach to Explore Functional Data"*.

## Project Overview

The goal is to construct prediction bands for functional data that guarantee a specified coverage probability (e.g., 90%) without making strong distributional assumptions.

The analysis is performed in two main stages:
1.  **Data Processing (`dataset_creation.R`)**: Raw accelerometer signals are trimmed, windowed into 10-second segments, resampled to a common grid, and centered.
2.  **Conformal Analysis (`main.R`)**:
    -   **Gaussian Mixture Approximation**: Projects curves onto a cosine basis and fits a Gaussian Mixture Model (GMM) to the coefficients.
    -   **Pseudo-Density Methods**: Uses kernel density estimation to identify functional modes (prototypes) and detect anomalies.
    -   **Construction of Prediction Bands**: Generates conformal prediction bands based on GMM density level sets.

## Dataset

The dataset consists of tri-axial accelerometer recordings (magnitude) collected from a **Samsung Galaxy A70** using the **phyphox** app.

-   **Activities**: Standing, Walking, Fast Walking.
-   **Sampling Rate**: ~203 Hz.
-   **Preprocessing**: Windows of 10 seconds, resampled to 200 points.

## Project Structure

-   `dataset_creation.R`: Script to process raw data and generate `accel_fda_dataset.rds`.
-   `main.R`: Main analysis script implementing conformal prediction algorithms.
-   `accel_fda_dataset.rds`: The processed functional dataset.
-   `data/`: Directory containing raw data files.
-   `outputs/`: Directory for generated plots and results.
-   `Feili_SeyyediParsa_Torabi_HW.Rproj`: R Project file.

## Requirements

The project requires **R** and the following packages:

```r
install.packages(c("knitr", "mclust", "mvtnorm"))
```

## Usage

1.  **Generate the Dataset**:
    Run `dataset_creation.R` to process the raw recordings.
    ```r
    source("dataset_creation.R")
    ```

2.  **Run the Analysis**:
    Run `main.R` to perform the conformal prediction and generate the report.
    ```r
    source("main.R")
    ```
    Alternatively, you can knit `main.R` to HTML in RStudio to produce a full report.

## Authors

-   Arman Feili
-   SeyyediParsa
-   Torabi
