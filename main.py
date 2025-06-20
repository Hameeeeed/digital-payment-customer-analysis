import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ✅ Load Dataset
df = pd.read_csv("C:/Users/xdham/Downloads/archive (1)/digital_wallet_ltv_dataset.csv")
print("🔍 Step 1: Initial Dataset Shape:", df.shape)

# ✅ Remove Blank Rows and Drop Duplicates
df.dropna(how='all', inplace=True)
df.drop_duplicates(inplace=True)
print("🔍 Step 2: After Removing Empty Rows and Duplicates:", df.shape)

# ✅ Remove Header Row Duplicates
df = df[df['Customer_ID'] != 'Customer_ID']
print("🔍 Step 3: After Removing Header Duplicates:", df.shape)

# ✅ Reset Index
df.reset_index(drop=True, inplace=True)

# ✅ Check Data Types and Missing Values Before Cleaning
print("🔍 Step 4: Column Data Types\n", df.dtypes)
print("🔍 Step 5: Missing Values Count:\n", df.isna().sum())

# ✅ Convert Appropriate Columns to Numeric
non_numeric_cols = ['Customer_ID', 'Location', 'Preferred_Payment_Method']
numeric_columns = [col for col in df.columns if col not in non_numeric_cols]

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
print("🔍 Step 6: After Conversion to Numeric, Missing Values Count:\n", df[numeric_columns].isna().sum())

# ✅ Impute Missing Values
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
print("🔍 Step 7: After Imputation, Missing Values Count:\n", df[numeric_columns].isna().sum())

# ✅ Validate Data Cleaning
print("🔍 Step 8: Final Dataset Shape Before Heatmap:", df.shape)
print(df.head())

# ✅ Normalize Data Using MinMaxScaler
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns)

# ✅ Plot Enhanced Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_normalized.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 8})
plt.title("Enhanced Correlation Heatmap After Cleaning")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
