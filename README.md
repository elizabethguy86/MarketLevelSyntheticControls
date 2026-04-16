## Store-Level Synthetic Control

Fits a separate synthetic control for **each individual test unit** and returns the original dataframe with an `expected` column (or `expected_{outcome}` columns for multiple outcomes) representing each test store's counterfactual trajectory.

Unlike `SyntheticDiD`, which treats all test units as one pooled treatment group, `StoreLevelSyntheticControl` fits independent unit weights per unit, so each unit gets its own tailored synthetic control. This allows for breakouts of various groups of stores that ladder up to the overall effect.

Other test units are **excluded from each units's donor pool** to prevent contamination between treated units.

### Example Usage

```python
# Load your panel data
df = pd.read_csv('your_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Define test stores and add flag columns
test_units = [1, 2, 3, 4, 5,
               6, 7, 8, 9, 10, 11,
               12, 13, 14, 15, 16, 17, 18, 19, 20]

test_date = pd.to_datetime('YYYY-MM-DD')
df['is_test'] = df['locationNum'].isin(test_stores)
df['post_launch'] = df['checkOpenDate'] > test_date

# Single outcome
ulsc = UnitLevelSyntheticControl(
    outcome_col='transactions',
    time_col='Date',
    unit_col='locationNum',
    treat_col='is_test',
    post_col='post_launch',
    regularization_multiplier=3.0,
    tail_periods=30,
)
result_df = ulsc.fit_transform(df)
# result_df now has an 'expected' column for test store rows

# Multiple outcomes
ulsc_multi = UnitLevelSyntheticControl(
    outcome_col=['transactions', 'revenue'],
    time_col='Date',
    unit_col='locationNum',
    treat_col='is_test',
    post_col='post_launch',
    regularization_multiplier=3.0,
    tail_periods=30,
)
result_df_multi = ulsc_multi.fit_transform(df)
# result_df_multi has 'expected_transactions' and 'expected_revenue' columns
```

### Outcomes that are a ratio of two metrics
Defining ratio metrics is necessary to get accurate
estimates of the expected value of a ratio
metric as a function of changes in the numerator /
changes in the denominator

```python

ulsc_ratio = UnitLevelSyntheticControl(
    outcome_col=['transactions', 'revenue'],
    time_col='Date',
    unit_col='locationNum',
    treat_col='is_test',
    post_col='post_launch',
    regularization_multiplier=3.0,
    tail_periods=30,
    ratio_metrics={'avgTransaction': ('revenue', 'transactions')},  # derived ratio
)

# Inspect per-store unit weights
for store_id, weights in ulsc.store_weights_.items():
    print(f"Store {store_id}:")
    print(weights['transactions'].nlargest(5))
```
