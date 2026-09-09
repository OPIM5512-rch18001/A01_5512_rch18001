from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Create a boxplot of 'MedInc' (Median Income)
plt.figure(figsize=(8, 6))
df.boxplot(column='MedInc')
plt.title('Boxplot of Median Income')
plt.ylabel('Median Income')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("figs/median_income_boxplot.png", dpi=300)
plt.show()

# Create a boxplot of 'MedHouseVal' (Median House Value)
plt.figure(figsize=(8, 6))
df.boxplot(column='MedHouseVal')
plt.title('Boxplot of Median House Value')
plt.ylabel('Median House Value')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("figs/median_house_value_boxplot.png", dpi=300)
plt.show()