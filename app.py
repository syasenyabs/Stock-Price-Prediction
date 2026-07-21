import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df=pd.read_csv("dataset.csv")
df["Date"]=pd.to_datetime(df["Date"])
df=df.sort_values("Date")

feature_cols=["Open","High","Low","Close","Volume"]

data=torch.tensor(df[feature_cols].values,dtype=torch.float32)

train_size=int(len(data)*0.7)
val_size=int(len(data)*0.15)
test_size=len(data)-train_size-val_size

train_data=data[:train_size]
val_data=data[train_size:train_size+val_size]
test_data=data[train_size+val_size:]

train_min=train_data.min(dim=0).values
train_max=train_data.max(dim=0).values

train_scaled=(train_data-train_min)/(train_max-train_min)
val_scaled=(val_data-train_min)/(train_max-train_min)
test_scaled=(test_data-train_min)/(train_max-train_min)

seq_length=30

def create_sequences(data,seq_length):
    xs=[]
    ys=[]
    for i in range(len(data)-seq_length):
        x=data[i:i+seq_length]
        y=data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs),torch.stack(ys)

x_train,y_train=create_sequences(train_scaled,seq_length)
x_val,y_val=create_sequences(val_scaled,seq_length)
x_test,y_test=create_sequences(test_scaled,seq_length)

print(x_train.shape)
print(y_train.shape)
