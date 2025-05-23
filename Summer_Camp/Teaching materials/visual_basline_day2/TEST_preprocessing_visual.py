from my_lib import *
import pandas as pd

config = VisualConfig()
tr_train = get_preprocessing(config, is_training=True)
tr_test = get_preprocessing(config, is_training=False)

df_train = pd.read_csv("dataset/train.csv")
train_ds = DFDataset("dataset/images/train", df_train, transform=tr_train)

print(train_ds[5])

print(tr_train)

print(tr_test)
