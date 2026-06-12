import pandas as pd
df = pd.read_csv(r'C:\Users\jumma\Downloads\GaitPhase\data\GP10_0.6_force.csv')
print('Rows:', len(df))
print('Null rows:', df.isnull().all(axis=1).sum())
print('FP1_z min/max:', df['FP1_z'].min(), df['FP1_z'].max())
print('FP2_z min/max:', df['FP2_z'].min(), df['FP2_z'].max())