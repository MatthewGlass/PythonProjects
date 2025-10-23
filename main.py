# 🎬 Movie Data Analysis

# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Set plot style
plt.style.use('ggplot')
from matplotlib.pyplot import figure
matplotlib.rcParams['figure.figsize'] = (12,8)

# Read in the data
df = pd.read_csv('movies.csv')

# Changing data type of budget and gross columns to be integers.
# Also making all empty cells 0
df['budget'] = df['budget'].fillna(0).astype('int64')
df['gross'] = df['gross'].fillna(0).astype('int64')
df['votes'] = df['votes'].fillna(0)
df['score'] = df['score'].fillna(0)
df['runtime'] = df['runtime'].fillna(0)

# Data types for columns
print(df.dtypes)

# Checking columns with empty cells
for col in df.columns:
    print('{} - {}% missing'.format(col, np.mean(df[col].isnull())*100))

# Creating correct year column from 'released' column
df['yearcorrect'] = df['released'].str.extract(r'(\d{4})')

# Sorting by gross earnings
df.sort_values(by=['gross'], inplace=False, ascending=False)

# Drop any duplicates
df.drop_duplicates(inplace=True)

# Scatter plot: Budget vs Gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget for Film')
plt.ylabel('Gross Earnings')
plt.show()

# Plot Budget vs Gross using Seaborn with regression line
sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color": "red"}, line_kws={"color":"blue"})
plt.title('Budget vs Gross Earnings (Regression Line)')
plt.show()

# Correlation matrix for numeric features
correlation_matrix = df.corr(numeric_only=True, method='pearson')
print(correlation_matrix)

# Heatmap of correlation matrix
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

# Numerizing object columns to find correlations
df_numerized = df.copy()
for col_name in df_numerized.columns:
    if df_numerized[col_name].dtype == 'object':
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

print(df_numerized)

# Correlation matrix after numerization
correlation_matrix = df_numerized.corr(numeric_only=True, method='pearson')

# Heatmap of numerized correlations
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features (Numerized)')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

# Finding high correlations (>0.5 but not 1)
correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
sorted_pairs = corr_pairs.sort_values()
high_corr = sorted_pairs[((sorted_pairs) > 0.5) & ((sorted_pairs) != 1)]
print("Highly correlated feature pairs:\n", high_corr)
