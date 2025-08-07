import pandas as pd

df = pd.read_parquet('yellow_tripdata_2023-03.parquet')
print(len(df))
