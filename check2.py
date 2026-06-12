import pandas as pd
df = pd.read_csv(r'C:\Users\jumma\Downloads\GaitPhase\data\processed\features.csv')
print(df.shape)
print(df.isnull().sum())
print(df.describe())