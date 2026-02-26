import pandas as pd
data = pd.read_csv("raw_data.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

print("Duplicates:", data.duplicated().sum())

data = data.dropna()
print(data.duplicated().sum())
data = data.drop_duplicates()
data.to_csv("cleaned_data.csv", index=False)
print("cleaning completed successfully")