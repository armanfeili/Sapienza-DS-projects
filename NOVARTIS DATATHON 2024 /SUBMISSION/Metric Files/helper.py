# -*- coding: utf-8 -*-
"""Compute metrics for Novartis Datathon 2024.
   This auxiliar file is intended to be used by participants in case
   you want to test the metric with your own train/validation splits."""

import pandas as pd
from pathlib import Path
from typing import Tuple


def _CYME(df: pd.DataFrame) -> float:
    """ Compute the CYME metric, that is 1/2(median(yearly error) + median(monthly error))"""

    yearly_agg = df.groupby("cluster_nl")[["target", "prediction"]].sum().reset_index()
    yearly_error = abs((yearly_agg["target"] - yearly_agg["prediction"])/yearly_agg["target"]).median()

    monthly_error = abs((df["target"] - df["prediction"])/df["target"]).median()

    return 1/2*(yearly_error + monthly_error)


def _metric(df: pd.DataFrame) -> float:
    """Compute metric of submission.

    :param df: Dataframe with target and 'prediction', and identifiers.
    :return: Performance metric
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Split 0 actuals - rest
    zeros = df[df["zero_actuals"] == 1]
    recent = df[df["zero_actuals"] == 0]

    # weight for each group
    zeros_weight = len(zeros)/len(df)
    recent_weight = 1 - zeros_weight

    # Compute CYME for each group
    return round(recent_weight*_CYME(recent) + zeros_weight*min(1,_CYME(zeros)),8)


def compute_metric(submission: pd.DataFrame) -> Tuple[float, float]:
    """Compute metric.

    :param submission: Prediction. Requires columns: ['cluster_nl', 'date', 'target', 'prediction']
    :return: Performance metric.
    """

    submission["date"] = pd.to_datetime(submission["date"])
    submission = submission[['cluster_nl', 'date', 'target', 'prediction', 'zero_actuals']]

    return _metric(submission)


# Load data
PATH = Path("path/to/data/folder")
train_data = pd.read_csv(PATH / "train_data.csv")

# Split into train and validation set

validation = #proportion of train

# Train your model

# Perform predictions on validation set
validation["prediction"] = #model.predict(validation[features])

# Assign column ["zero_actuals"] in the depending if in your
# split the cluster_nl has already had actuals on train or not

validation["zero_actuals"] = # Boolean assignation

# Optionally check performance
print("Performance:", compute_metric(validation))

# Prepare submission
submission_data = pd.read_parquet(PATH / "submission_data.csv")
submission = pd.read_csv(PATH / "submission_template.csv")

# Fill in 'prediction' values of submission
submission["prediction"] = #model.predict(submission_data[features])

# ...

# Save submission
SAVE_PATH = Path("path/to/save/folder")
ATTEMPT = "attempt_x"
submission.to_csv(SAVE_PATH / f"submission_{ATTEMPT}.csv", sep=",", index=False)
