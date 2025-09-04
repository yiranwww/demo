import pandas as pd
import os

base_path = os.path.dirname(__file__)  # folder of your script
csv_path = os.path.join(base_path, "Data", "ad_10000records.csv")
data = pd.read_csv(csv_path)
print(data.head())
