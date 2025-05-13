# EDA-1: Bike Details Dataset

## Q1: What is the range of selling prices in the dataset?
```python
import pandas as pd

df = pd.read_csv("BIKE DETAILS.csv")
price_range = df['selling_price'].max() - df['selling_price'].min()
price_range
```
**Answer:** ₹1,766,000

---

## Q2: What is the median selling price for bikes in the dataset?
```python
median_price = df['selling_price'].median()
median_price
```
**Answer:** ₹55,000

---

## Q3: What is the most common seller type?
```python
most_common_seller = df['seller_type'].mode()[0]
most_common_seller
```
**Answer:** "Individual"

---

## Q4: How many bikes have driven more than 50,000 kilometers?
```python
count_over_50k = df[df['km_driven'] > 50000].shape[0]
count_over_50k
```
**Answer:** 166 bikes

---

## Q5: What is the average km_driven value for each ownership type?
```python
avg_km_by_owner = df.groupby('owner')['km_driven'].mean()
avg_km_by_owner
```
**Answer:**
- 1st owner: 30,362 km  
- 2nd owner: 37,802 km  
- 3rd owner: 46,533 km  
- 4th owner or more: 63,358 km  
- Test Drive Car: 3,544 km

---

## Q6: What proportion of bikes are from the year 2015 or older?
```python
old_bikes = df[df['year'] <= 2015].shape[0]
proportion_old = old_bikes / df.shape[0]
proportion_old
```
**Answer:** Approximately 36.3%

---

## Q7: What is the trend of missing values across the dataset?
```python
missing_values = df.isnull().sum()
missing_values_percent = df.isnull().mean() * 100
missing_values, missing_values_percent
```
**Answer:** `ex_showroom_price` has 195 missing values (~18.4%)

---

## Q8: What is the highest ex_showroom_price recorded, and for which bike?
```python
highest_price_row = df[df['ex_showroom_price'] == df['ex_showroom_price'].max()]
highest_price_row[['name', 'ex_showroom_price']]
```
**Answer:** ₹2,634,983 for Harley Davidson Street 750

---

## Q9: What is the total number of bikes listed by each seller type?
```python
seller_counts = df['seller_type'].value_counts()
seller_counts
```
**Answer:**
- Individual: 875 bikes  
- Dealer: 186 bikes

---

## Q10: What is the relationship between selling_price and km_driven for first-owner bikes?
```python
import seaborn as sns
import matplotlib.pyplot as plt

first_owner_df = df[df['owner'] == '1st owner']
sns.scatterplot(data=first_owner_df, x='km_driven', y='selling_price')
plt.title("Selling Price vs KM Driven for 1st Owner Bikes")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price")
plt.show()
```
**Answer:** Negative correlation – selling price decreases as km_driven increases.

---

## Q11: Identify and remove outliers in the km_driven column using the IQR method.
```python
Q1 = df['km_driven'].quantile(0.25)
Q3 = df['km_driven'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR
outliers_removed_df = df[df['km_driven'] <= upper_limit]
df.shape[0] - outliers_removed_df.shape[0]  # Number of outliers
```
**Answer:** 47 outliers removed

---

## Q12: Perform a bivariate analysis to visualize the relationship between year and selling_price.
```python
sns.scatterplot(data=outliers_removed_df, x='year', y='selling_price')
plt.title("Selling Price vs Manufacturing Year")
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.show()
```
**Answer:** Positive trend – newer bikes have higher prices.

---

## Q13: What is the average depreciation in selling price based on the bike's age?
```python
df = df.dropna(subset=['ex_showroom_price'])
df['age'] = 2025 - df['year']
df = df[df['age'] > 0]
df['depreciation'] = (df['ex_showroom_price'] - df['selling_price']) / df['age']
avg_depreciation = df['depreciation'].mean()
avg_depreciation
```
**Answer:** Approx. ₹11,062 depreciation per year

---

## Q14: Which bike names are priced significantly above the average price for their manufacturing year?
```python
df['year_avg'] = df.groupby('year')['selling_price'].transform('mean')
above_avg_bikes = df[df['selling_price'] > 1.5 * df['year_avg']]
above_avg_bikes[['name', 'year', 'selling_price']]
```
**Answer:** Includes bikes like Harley Davidson Street 750, KTM RC 390, and RE Interceptor 650

---

## Q15: Develop a correlation matrix for numeric columns and visualize it using a heatmap.
```python
numeric_cols = df[['selling_price', 'km_driven', 'year', 'ex_showroom_price']]
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```
**Answer:**  
- Strong positive: selling_price & ex_showroom_price (+0.83)  
- Moderate positive: selling_price & year (+0.59)  
- Weak negative: selling_price & km_driven (-0.28)