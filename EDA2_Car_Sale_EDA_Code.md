# EDA-2: Car Sale Dataset

## Q1: What is the average selling price of cars for each dealer, and how does it compare across different dealers?
```python
import pandas as pd

df = pd.read_csv("Car Sale.csv")
avg_price_by_dealer = df.groupby("Dealer_Name")["Price ($)"].mean().sort_values(ascending=False)
avg_price_by_dealer
```
**Answer:** AutoNation has the highest average car price; Hertz Car Sales has the lowest.

---

## Q2: Which car brand (Company) has the highest variation in prices, and what does this tell us about the pricing trends?
```python
price_std_by_company = df.groupby("Company")["Price ($)"].std().sort_values(ascending=False)
price_std_by_company.head()
```
**Answer:** BMW has the highest price variation, indicating a wide range from budget to luxury models.

---

## Q3: What is the distribution of car prices for each transmission type, and how do the interquartile ranges compare?
```python
manual_iqr = df[df['Transmission'] == 'Manual']["Price ($)"].quantile([0.25, 0.5, 0.75])
automatic_iqr = df[df['Transmission'] == 'Automatic']["Price ($)"].quantile([0.25, 0.5, 0.75])
manual_iqr, automatic_iqr
```
**Answer:** Automatic cars have higher median price and a wider IQR than manual cars.

---

## Q4: What is the distribution of car prices across different regions?
```python
avg_price_by_region = df.groupby("Dealer_Region")["Price ($)"].mean()
avg_price_by_region
```
**Answer:** West region has the highest average car prices.

---

## Q5: What is the distribution of cars based on body styles?
```python
body_style_counts = df["Body Style"].value_counts()
body_style_counts
```
**Answer:** Sedans are the most common body style, followed by SUVs.

---

## Q6: How does the average selling price of cars vary by customer gender and annual income?
```python
avg_price_income_by_gender = df.groupby("Gender")[["Annual Income", "Price ($)"]].mean()
avg_price_income_by_gender
```
**Answer:** Male customers have higher income and spend more on average compared to female customers.

---

## Q7: What is the distribution of car prices by region, and how does the number of cars sold vary by region?
```python
region_summary = df.groupby("Dealer_Region").agg({"Price ($)": "mean", "Car_id": "count"})
region_summary
```
**Answer:** North has the highest number of sales; West has the highest average prices.

---

## Q8: How does the average car price differ between cars with different engine sizes?
```python
avg_price_by_engine = df.groupby("Engine")["Price ($)"].mean()
avg_price_by_engine
```
**Answer:** Electric and V8 engines have the highest average prices.

---

## Q9: How do car prices vary based on the customer’s annual income bracket?
```python
def income_bracket(income):
    if income < 50000:
        return "Below $50,000"
    elif income <= 100000:
        return "$50,000 - $100,000"
    else:
        return "Above $100,000"

df["Income Bracket"] = df["Annual Income"].apply(income_bracket)
avg_price_by_income_bracket = df.groupby("Income Bracket")["Price ($)"].mean()
avg_price_by_income_bracket
```
**Answer:** Higher income brackets tend to spend more on car purchases.

---

## Q10: What are the top 5 car models with the highest number of sales, and how does their price distribution look?
```python
top_models = df["Model"].value_counts().head(5)
top_models_data = df[df["Model"].isin(top_models.index)]
top_models_summary = top_models_data.groupby("Model")["Price ($)"].describe()
top_models_summary
```
**Answer:** Civic and Corolla are top-selling models, with Civic having the highest average price.

---

## Q11: How does car price vary with engine size across different car colors, and which colors have the highest price variation?
```python
color_variation = df.groupby("Color")["Price ($)"].std().sort_values(ascending=False)
color_variation.head()
```
**Answer:** Black cars have the highest price variation, often linked with high-end models.

---

## Q12: Is there any seasonal trend in car sales based on the date of sale?
```python
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
monthly_sales = df.groupby("Month")["Car_id"].count()
monthly_sales
```
**Answer:** Peak sales occur in March and July; lowest in December.

---

## Q13: How does the car price distribution change when considering different combinations of body style and transmission type?
```python
combo_price = df.groupby(["Body Style", "Transmission"])["Price ($)"].mean()
combo_price
```
**Answer:** Automatic SUVs are the most expensive; Manual Sedans are more affordable.

---

## Q14: What is the correlation between car price, engine size, and annual income of customers, and how do these features interact?
```python
encoded_engine = pd.get_dummies(df["Engine"], drop_first=True)
merged_df = df.join(encoded_engine)
correlation_matrix = merged_df[["Price ($)", "Annual Income"] + list(encoded_engine.columns)].corr()
correlation_matrix["Price ($)"].sort_values(ascending=False)
```
**Answer:** Positive correlations suggest high-income customers prefer cars with larger or electric engines.

---

## Q15: How does the average car price vary across different car models and engine types?
```python
avg_price_model_engine = df.groupby(["Model", "Engine"])["Price ($)"].mean().sort_values(ascending=False)
avg_price_model_engine.head()
```
**Answer:** Mustang (V8) and Prius (Electric) are among the highest-priced combinations.