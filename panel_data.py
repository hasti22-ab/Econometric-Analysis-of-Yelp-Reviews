# -*- coding: utf-8 -*-
"""panel data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qPCMZrkoTlRD7OMCji_jtz8XneKP0fGn
"""

import sqlite3
import pandas as pd

# Path to the database file
db_path = r'C:\Users\asal\Downloads\New folder\YelpCHI\yelpResData.db'

# Connect to the SQLite database
conn = sqlite3.connect(db_path)

try:
    # Query to list all tables in the database
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = conn.execute(query).fetchall()
    print("Tables in the database:", tables)

    # Export 'restaurant' table to CSV
    restaurant_query = "SELECT * FROM restaurant;"
    restaurant_df = pd.read_sql_query(restaurant_query, conn)
    restaurant_df.to_csv("restaurant.csv", index=False)
    print("Table 'restaurant' has been exported to 'restaurant.csv'.")

    # Export 'reviewer' table to CSV
    reviewer_query = "SELECT * FROM reviewer;"
    reviewer_df = pd.read_sql_query(reviewer_query, conn)
    reviewer_df.to_csv("reviewer.csv", index=False)
    print("Table 'reviewer' has been exported to 'reviewer.csv'.")

    # Export 'review' table to CSV using alternative decoding
    review_query = "SELECT * FROM review;"
    try:
        # Try UTF-8 decoding first
        review_df = pd.read_sql_query(review_query, conn)
    except Exception as utf8_error:
        print(f"UTF-8 decoding failed: {utf8_error}")
        try:
            # Fallback to ISO-8859-1 encoding
            conn.text_factory = lambda x: str(x, 'ISO-8859-1', 'ignore')
            review_df = pd.read_sql_query(review_query, conn)
            print("Successfully decoded using ISO-8859-1 encoding.")
        except Exception as iso_error:
            print(f"ISO-8859-1 decoding failed: {iso_error}")
            # Log problematic rows if all decoding attempts fail
            problematic_rows_query = """
            SELECT *
            FROM review
            WHERE reviewContent LIKE '%�%';
            """
            problematic_rows = pd.read_sql_query(problematic_rows_query, conn)
            problematic_rows.to_csv("problematic_rows_review.csv", index=False)
            print("Problematic rows have been logged to 'problematic_rows_review.csv'.")
            raise iso_error

    # Save the successfully decoded 'review' table to a CSV file
    review_df.to_csv("review.csv", index=False)
    print("Table 'review' has been exported to 'review.csv'.")

except Exception as e:
    print(f"Error processing data: {e}")

finally:
    # Close the database connection
    conn.close()

pip install ipywidgets

import pandas as pd
df1 = pd.read_csv(r"C:\Users\asal\Downloads\New folder\YelpCHI\restaurant.csv")
df1 = df1.rename(columns={'name': 'restaurant_name', 'rating': 'rating_restaurant', 'reviewCount':'reviewCount_restaurant' , 'address':'address_restaurant'})
df1.columns

import pandas as pd
df2 = pd.read_csv(r"C:\Users\asal\Downloads\New folder\YelpCHI\review.csv")
df2 = df2.rename(columns={ 'rating': 'rating_review','usefulCount':'usefulCount_review', 'coolCount':'coolCount_review', 'funnyCount':'funnyCount_review'})
df2.columns

import pandas as pd
df3 = pd.read_csv(r"C:\Users\asal\Downloads\New folder\YelpCHI\reviewer.csv")
df3 = df3.rename(columns={ 'reviewCount': 'reviewCount_reviewer','name':'name_reviewer'})
df3.columns

import pandas as pd

try:
    # Create DataFrames (Ensure correct data is loaded)
    restaurant_df = pd.DataFrame(df1)
    review_df = pd.DataFrame(df2)
    reviewer_df = pd.DataFrame(df3)

    # Ensure consistent column names
    restaurant_df.rename(columns={'restaurantID ': 'restaurantID'}, inplace=True)
    review_df.rename(columns={'rest_id': 'restaurantID'}, inplace=True)

    # Ensure consistent data types
    restaurant_df['restaurantID'] = restaurant_df['restaurantID'].astype(str)
    review_df['restaurantID'] = review_df['restaurantID'].astype(str)

    # Step 1: Merge restaurant and review on restaurantID
    merged_df1 = pd.merge(review_df, restaurant_df, on='restaurantID', how='inner')

    # Step 2: Merge the resulting DataFrame with reviewer on reviewerID
    final_merged_df = pd.merge(merged_df1, reviewer_df, on='reviewerID', how='inner')

    # Handle null values (optional)
    final_merged_df = final_merged_df.dropna()  # Drop rows with null values if necessary

    # Print merged data (first 5 rows)
    print("Merged DataFrame (First 5 Rows):")
    print(final_merged_df.head())

    # Save the merged DataFrame to a CSV file
    final_merged_df.to_csv("merged_data.csv", index=False)
    print("Merged data saved to 'merged_data.csv'.")

except Exception as e:
    print(f"Error merging data: {e}")

import pandas as pd
df = pd.read_csv(r"C:\Users\asal\Downloads\New folder\YelpCHI\merged_data.csv")

df.columns

df.shape

df['location_x']

print(df['location_x'].head(10))

df['city_state'] = df['location_x'].str.split('-').str[-1].str.strip()
print(df[['location_x', 'city_state']].head(10))

df['city_state'] = df['location_x'].str.split('-').str[-1].str.strip()

df['state'] = df['city_state'].str.split(',').str[-1].str.strip()
print(df[['location_x', 'city_state', 'state']])

filtered_df = df[df['state']=='IL']
filtered_df.shape

filtered_df['parsed_date'] = pd.to_datetime(filtered_df['date'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0], errors='coerce')

filtered_df['year_month'] = filtered_df['date'].str.extract(r'(\d{4}-\d{2})')[0]

filtered_df['to_drop'] = df.apply(lambda x: 'updated' in x['date'] and pd.notna(x['parsed_date']), axis=1)
filtered_df = filtered_df[~filtered_df['to_drop']]
filtered_df

print(filtered_df['date'].dtype)

filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')

import pandas as pd
import numpy as np

# Ensure 'flagged' is numeric: Convert 'N' to 1 and 'YR' to 0
filtered_df['flagged'] = filtered_df['flagged'].replace({'N': 0, 'YR': 1, 'NR': 0, 'Y': 1}).astype(int)

# Extract month and calculate review length
filtered_df['Month'] = filtered_df['date'].dt.to_period('M').dt.to_timestamp()
filtered_df['review_length'] = filtered_df['reviewContent'].astype(str).str.len()
filtered_df['rating_review'] = filtered_df['rating_review'].astype(float)  # Ensure ratings are numeric

# Aggregate metrics by restaurant and month, including fake reviews
agg_data = filtered_df.groupby(['restaurantID', 'Month']).agg(
    total_reviewers=('reviewerID', 'nunique'),           # Unique reviewers
    total_reviews=('reviewID', 'count'),                # Total reviews
    fake_reviews=('flagged', 'sum'),                    # Sum of fake reviews (flagged as 1)
    avg_review_length=('review_length', 'mean'),         # Average review length
    avg_restaurant_rating=('rating_review', 'mean')      # Average restaurant rating
).reset_index()

# Sort the aggregated data by restaurantID and Month
agg_data = agg_data.sort_values(by=['restaurantID', 'Month'], ascending=[True, True])

# Calculate the cumulative number of reviews
agg_data['cumulative_reviews'] = agg_data.groupby('restaurantID')['total_reviews'].cumsum()

# Calculate cumulative rating as a weighted average
agg_data['cumulative_rating_review'] = (
    agg_data.groupby('restaurantID').apply(
        lambda group: (group['avg_restaurant_rating'] * group['total_reviews']).cumsum() /
                      group['cumulative_reviews']
    ).reset_index(drop=True)
)

# Add columns for total and genuine review ratings
agg_data['total_rating'] = agg_data['total_reviews'] * agg_data['avg_restaurant_rating']
agg_data['fake_review_ratings'] = agg_data['fake_reviews'] * agg_data['avg_restaurant_rating']
agg_data['genuine_review_ratings'] = agg_data['total_rating'] - agg_data['fake_review_ratings']

# Calculate average ratings for fake and genuine reviews
agg_data['avg_fake_review_rating'] = agg_data['fake_review_ratings'] / agg_data['fake_reviews']
agg_data['avg_genuine_review_rating'] = (
    agg_data['genuine_review_ratings'] / (agg_data['total_reviews'] - agg_data['fake_reviews'])
)

# Handle cases where there are no fake or genuine reviews
agg_data['avg_fake_review_rating'] = agg_data['avg_fake_review_rating'].fillna(0)  # If no fake reviews
agg_data['avg_genuine_review_rating'] = agg_data['avg_genuine_review_rating'].fillna(0)  # If no genuine reviews

# Save the sorted data with cumulative reviews to a CSV file
agg_data.to_csv("sorted_monthly_restaurant_metrics_with_cumulative.csv", index=False)

# Display the sorted table (only the first few rows for brevity)
agg_data

filtered_df.shape

# Count the number of unique restaurant IDs
num_unique_restaurant_ids = len(filtered_df['restaurantID'].unique())
print(f"Number of unique restaurant IDs: {num_unique_restaurant_ids}")

filtered_df.shape

pip install linearmodels

import statsmodels.api as sm

# Define dependent and independent variables
X = agg_data[['total_reviewers', 'total_reviews', 'fake_reviews', 'avg_review_length', 'avg_restaurant_rating','avg_fake_review_rating','avg_genuine_review_rating']]
X = sm.add_constant(X)
y = agg_data['total_rating']

# OLS Model
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# Correlation matrix
correlation_matrix = agg_data[['total_reviewers', 'total_reviews', 'fake_reviews', 'avg_review_length', 'avg_restaurant_rating','avg_fake_review_rating','avg_genuine_review_rating']].corr()
print(correlation_matrix)

from statsmodels.api import OLS
from linearmodels.iv import IV2SLS
import pandas as pd

# Step 1: Add Ratio Instruments
agg_data['total_review_ratio'] = agg_data['total_reviews'] / agg_data['total_reviews'].sum()
agg_data['fake_review_ratio'] = agg_data['fake_reviews'] / agg_data['total_reviews']
agg_data['review_length_ratio'] = agg_data['avg_review_length'] / agg_data['total_reviews']
agg_data['restaurant_rating_ratio'] = agg_data['avg_restaurant_rating'] / agg_data['total_reviews']
agg_data['fake_rating_ratio'] = agg_data['avg_fake_review_rating'] / agg_data['total_reviews']
agg_data['genuine_rating_ratio'] = agg_data['avg_genuine_review_rating'] / agg_data['total_reviews']

# Ensure no division by zero or missing values
agg_data = agg_data.replace([float('inf'), float('-inf')], pd.NA).dropna()

# Step 2: Define Instruments for Each Variable
instruments = {
    'total_reviews': 'total_review_ratio',
    'fake_reviews': 'fake_review_ratio',
    'avg_review_length': 'review_length_ratio',
    'avg_restaurant_rating': 'restaurant_rating_ratio',
    'avg_fake_review_rating': 'fake_rating_ratio',
    'avg_genuine_review_rating': 'genuine_rating_ratio'
}

# Step 3: Initialize Results
results = []

# Step 4: Test Endogeneity and IV Relevance
for var, instrument in instruments.items():
    print(f"Running IV regression for: {var}")

    # Step 4a: First-Stage Regression (Relevance Condition)
    formula_first_stage = f'{var} ~ 1 + {" + ".join([v for v in instruments if v != var])} + {instrument}'
    first_stage = OLS.from_formula(formula_first_stage, data=agg_data).fit()

    # Print First-Stage F-Statistic
    f_stat = first_stage.fvalue
    print(f"First-Stage F-Statistic for {var}: {f_stat}")

    # Check relevance condition
    if f_stat < 10:
        print(f"Warning: Weak instrument detected for {var}. F-Statistic is {f_stat}.")

    # Step 4b: Augmented OLS Model with Residuals
    agg_data[f'{var}_residuals'] = first_stage.resid
    formula_augmented = f'total_rating ~ 1 + {" + ".join(instruments.keys())} + {var}_residuals'
    augmented_ols = OLS.from_formula(formula_augmented, data=agg_data).fit()

    # Residual Significance Test (Endogeneity)
    residual_pvalue = augmented_ols.pvalues[f'{var}_residuals']
    print(f"Residual p-value for {var}: {residual_pvalue}")
    endogeneity = "Yes" if residual_pvalue < 0.05 else "No"

    # Step 4c: Run IV Regression
    formula_iv = f'total_rating ~ 1 + {" + ".join([v for v in instruments if v != var])} + [{var} ~ {instrument}]'
    iv_model = IV2SLS.from_formula(formula_iv, data=agg_data).fit()

    # Step 4d: Hansen's J-Test (Exogeneity)
    hansen_stat = None
    hansen_pvalue = None
    if iv_model.model.instruments.shape[1] > iv_model.model.endog.shape[1]:
        # Only perform Hansen's J-Test when there are more instruments than endogenous variables
        hansen_stat = iv_model.j_stat.stat
        hansen_pvalue = iv_model.j_stat.pval
        print(f"Hansen's J-Statistic for {var}: {hansen_stat}")
        print(f"Hansen's J-Test P-Value for {var}: {hansen_pvalue}")
    else:
        print(f"Hansen's J-Test not applicable for {var} (exactly identified).")

    # Step 4e: Collect Results
    results.append({
        'Variable': var,
        'First-Stage F-Statistic': f_stat,
        'Residual P-Value': residual_pvalue,
        'Endogeneity': endogeneity,
        'Hansen J-Statistic': hansen_stat,
        'Hansen P-Value': hansen_pvalue,
        'OLS Coef': augmented_ols.params[var] if var in augmented_ols.params else None,
        'IV Coef': iv_model.params[var] if var in iv_model.params else None,
        'OLS R-Squared': augmented_ols.rsquared,
        'IV R-Squared': iv_model.rsquared
    })

# Step 5: Create Results DataFrame
results_df = pd.DataFrame(results)

# Step 6: Print Results
print("\nEndogeneity and IV Test Results:")
print(results_df)