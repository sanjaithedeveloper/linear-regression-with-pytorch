import torch
import pandas as pd
from config import G_DATA_PATH
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def processDataset():
    df = pd.read_csv(f'{G_DATA_PATH}/raw-data/laptop-price-dataset.csv')
    labelEncoder = LabelEncoder()
    categoricalColumns = ["Company", "Product", "TypeName", "ScreenResolution", 
                           "CPU_Company", "CPU_Type", "Memory", "GPU_Company", "GPU_Type", "OpSys"]
    for col in categoricalColumns:
        df[col] = labelEncoder.fit_transform(df[col].astype(str))
    numericalColumns = ["Inches", "CPU_Frequency (GHz)", "RAM (GB)", "Weight (kg)", "Price (Euro)"]
    scaler = MinMaxScaler()
    df[numericalColumns] = scaler.fit_transform(df[numericalColumns])
    X = df.drop(columns=["Price (Euro)"])
    y = df["Price (Euro)"]
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    combinedTensor = torch.cat((X_tensor, y_tensor), dim=1)
    return combinedTensor

