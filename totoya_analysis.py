import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("toyota.csv")

# Display basic info
print("\n")
print("-----------BASIC INFORMATION OF THE DATASET-----------")
print("Number of rows and columns:", df.shape)
print(df.info())


print("FIRST 5 ROWS OF THE DATASET:")
print(df.head())


print("\n")
print("--------CLEANING THE DATASET-------")
print("Null values per column:\n", df.isnull().sum())

# Drop rows with missing values
df = df.dropna()


# Find duplicates
duplicates = df.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")

# Drop duplicates
df = df.drop_duplicates()


# View data types
print(df.dtypes)

# Ensure correct data types
df['year'] = df['year'].astype(int)
df['price'] = df['price'].astype(float)
df['mileage'] = df['mileage'].astype(float)
df['engine_size'] = df['engine_size'].astype(float)
df['fuel_type'] = df['fuel_type'].astype('category')


# remove outliers using Interquartile Range method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

for col in ['price', 'mileage', 'engine_size']:
    df = remove_outliers_iqr(df, col)

print(f"Remaining data points after outlier removal: {len(df)}")


# Remove cars from the future
df = df[df['year'] <= 2025]

# Remove non-positive values
df = df[df['price'] > 0]
df = df[df['mileage'] > 0]
df = df[df['engine_size'] > 0]

df = df.reset_index(drop=True)

print("\n")
print("-----------CLEANED DATASET-----------")

print(df.info())
print(f"Final number of rows: {len(df)}")


prices = df['price'].values

# NumPy Operations
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)
max_price = np.max(prices)
min_price = np.min(prices)

# Mode of fuel type
values, counts = np.unique(df['fuel_type'], return_counts=True)
ind = np.argmax(counts)
most_common_fuel = values[ind]
most_common_count = counts[ind]

print("\n")
print("--------MOST USED FUEL TYPE (USING NUMPY--------")
print(f"Most common fuel type: {most_common_fuel}, \nCount: {most_common_count}")

print("\n")
print("----------PRIZE ANALYSIS USING NUMPY-----------")
print(f"Mean: {mean_price}, \nMedian: {median_price}, \nStd: {std_price}, \nMax: {max_price}, \nMin: {min_price}")

# scipy operations

corr, p_value1 = pearsonr(df['mileage'], df['price'])
corr2, p_value2 = pearsonr(df['engine_size'], df['price'])



print("\n")
print("--------PEARSON CORRELATION ANALYSIS USING SCIPY--------")
print(f"Mileage vs Price:{corr}, p-value: {p_value1}")
print(f"Engine Size vs Price: {corr2}, p-value: {p_value2}")



# price vs mileage (using statsmodels)
X = sm.add_constant(df['mileage'])
y = df['price']
model = sm.OLS(y, X).fit()

print("\n")
print("----------LINEAR REGRESSION ANALYSIS USING STATSMODELS (PRICE VS. MILEAGE)--------")

print(model.summary())


# Visualizations
# Required visualization 1: Distribution of car prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Required visualization 2: Boxplot of price by fuel type
plt.figure(figsize=(10, 6))
sns.boxplot(x='fuel_type', y='price', data=df)
plt.title("Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.show()

# Required correlation plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='mileage', y='price', data=df, scatter_kws={"alpha":0.5})
plt.title("Correlation between Mileage and Price")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.show()