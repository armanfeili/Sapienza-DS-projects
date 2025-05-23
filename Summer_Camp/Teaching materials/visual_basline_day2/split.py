from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/train.csv")
kf = KFold(5, shuffle=True, random_state=42)

df_fold = pd.DataFrame()
df_fold["fold"] = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    df_fold.loc[val_idx, "fold"] = int(fold)

df_fold["fold"] = df_fold["fold"].astype(int)
df_fold.to_csv("dataset/fold.csv", index=False)
