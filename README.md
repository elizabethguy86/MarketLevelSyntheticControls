## Market-Level Synthetic Control Package

This functionality creates a separate synthetic control for **each individual test unit** and returns the original dataframe with an `expected` column (or `expected_{outcome}` columns for multiple outcomes) representing each test store's counterfactual trajectory.

Unlike `SyntheticDiD`, which treats all test units as one pooled treatment group, `StoreLevelSyntheticControl` fits independent unit weights per unit, so each unit gets its own tailored synthetic control. This allows for breakouts of various groups of stores that ladder up to the overall effect.

Other test units are **excluded from each units's donor pool** to prevent contamination between treated units.

### Data Requirements
Python pandas dataframe with the structure: 

| Column | Type | Description |
|--------|------|-------------|
| `time_col` | date / datetime | One value per time period (e.g., `Date`) |
| `unit_col` | int / str | Unit identifier (e.g., `locationNum`) |
| `treat_col` | bool | `True` for test units, `False` for control units |
| `post_col` | bool | `True` for post-intervention rows, `False` for pre-period |
| outcome columns | numeric | All columns listed in `outcome_col`; if using `ratio_metrics`, both numerator and denominator columns must be present in your dataframe |

Note that other time series like ambient temperature or macroeconomic factors could be used as long as they are in the time series format and have an entry for that date in the `time_col`. Specify the time series indicator label in the `unit_col` (e.g., "temperature").

### Example Usage

```python
# Load your panel data
df = pd.read_csv('your_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Define test units and add flag columns
test_units = [1, 2, 3, 4, 5,
               6, 7, 8, 9, 10, 11,
               12, 13, 14, 15, 16, 17, 18, 19, 20]

test_date = pd.to_datetime('YYYY-MM-DD')
df['is_test'] = df['locationNum'].isin(test_units)
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
```
### Inspect pre-intervention fit metrics
To assess the quality of your model, MarketLevelSyntheticControls has a `pre_period_fit_metrics` function that outputs the MAPE and R^2 values for your model prior to the intervention time. 

```python
pre_period_fit_metrics(
    result_df,
    outcomes=['transactions', 'revenue'],
    treat_col='is_test_store',
    post_col='post_launch',
    unit_col='locationNum',
    ratio_metrics={'avgTransaction': ('revenue', 'transactions')},
)
```
For more comprehensive k-folds forward chaining validation, use the `sc_forward_chain_cv` function on a fitted unit-level synthetic control object. 
```python
sc_forward_chain_cv(ulsc, df, k=5)
```
Validation outputs will appear in a table and performed on each outcome and ratio metric identified. The output will look something like the following, with rows for each fold x metric combination:
| outcome | fold | n_train_days | n_val_store_days | mape | r2 | train_end | val_start | val_end |
|---|---|---|---|---|---|---|---|---|
| transactions | 1 | 46 | 874 | 10.76% | 0.7626 | 2025-02-14 | 2025-02-15 | 2025-04-01 |
| revenue | 1 | 46 | 874 | 12.37% | 0.7682 | 2025-02-14 | 2025-02-15 | 2025-04-01 |
| avgTransaction | 1 | 46 | 874 | 3.70% | 0.8241 | 2025-02-14 | 2025-02-15 | 2025-04-01 |


### Inspect per-treatment unit synthetic control weights
Examine the weights for time series used in each test unit's synthetic control. 

```python

for store_id, weights in ulsc.store_weights_.items():
    print(f"Store {store_id}:")
    print(weights['transactions'].nlargest(5))
```

### Examine p-values

```python
pval_df = ulsc_ratio.permutation_p_values(df)
# pval_df has rows for: transactions, revenue, avgTransaction
# avgTransaction lift = (Σ actual_revenue / Σ actual_transactions
#                  - Σ expected_revenue/ Σ expected_transactions)
#                 / (Σ expected_revenue / Σ expected_transactions)

# using a subsample of control units to build out the null distribution
pval_df_n_placebos = slsc.permutation_p_values(df, n_placebos=200, seed=42)
```
### Plotting actual vs expected time series based on SC modeling

```python
plotter = SyntheticControlPlotter(
    result_df=result_df_multi,
    time_col='Date',
    unit_col='locationNum',
    treat_col='is_test_store',
    post_col='post_launch',
    intervention_datetime='2025-11-03',
    ratio_metrics={'avgTicket': ('grossSales', 'transactions')},
)

plotter.plot('avgTransaction', yaxis_label='Avg Transaction Size').show()
plotter.plot('transactions').show()
plotter.plot('revenue', yaxis_label='Revenue').show()

```
### Pseudo-Power curves to check how well the model performs when fake lifts are added to a holdout time period
When building these curves, we want to see that the model is capable of detecting lifts applied to a holdout portion of data with 0.80 power. 
We also want to see that there is no lift detected at 0 lift (0% power). Power curves should be symmetric about the y-axis at 0 if the synthetic control is an unbiased estimator of the counterfactual for the treated group.

```python
# Make sure simulation window mirrors the length of the actual post-period 
# Hold out portion of pre-period

lifts = np.arange(-0.10, 0.10, 0.02).tolist()   # -10% to +10% in 2% steps

power_results, power_fig = sc_power_curve(
    slsc,
    df_drop,
    sim_start_date='2025-08-03',   # start of period for testing model
    sim_end_date='2025-09-28',     # last day before the start of the test
    lifts=lifts,
    n_permutations=500,
    alpha=0.05,
    seed=42, #for reproducibility
)

```
