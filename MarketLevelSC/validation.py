def pre_period_fit_metrics(
    result_df,
    outcomes,
    treat_col='is_test_store',
    post_col='post_launch',
    unit_col='locationNum',
    ratio_metrics=None,
):
    """
    Compute pre-period MAPE and R² for each outcome in result_df.

    For standard outcomes, uses the expected_{outcome} columns directly.
    For ratio metrics, derives actual and expected from their components.

    Days with NA values are dropped from the calculations and reflected in the n_store_days column.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of StoreLevelSyntheticControl.fit_transform().
    outcomes : list of str
        Declared outcome columns (matching outcome_col used in the SC).
    treat_col : str
        Boolean column flagging test stores.
    post_col : str
        Boolean column flagging post-intervention rows.
    unit_col : str
        Column identifying stores.
    ratio_metrics : dict or None
        Same dict passed to StoreLevelSyntheticControl, e.g.
        {'avgTicket': ('grossSales', 'transactions')}.

    Returns
    -------
    pd.DataFrame
        Columns: outcome, mape, r2, n_store_days
    """
    ratio_metrics = ratio_metrics or {}

    pre_test = result_df[
        result_df[treat_col].astype(bool) & ~result_df[post_col].astype(bool)
    ].copy()

    rows = []

    # ── Standard outcomes ──────────────────────────────────────────────────
    for outcome in outcomes:
        exp_col = f'expected_{outcome}'
        sub = pre_test[[outcome, exp_col]].dropna()
        actual = sub[outcome].values
        expected = sub[exp_col].values

        nonzero = actual != 0
        mape = np.mean(np.abs((actual[nonzero] - expected[nonzero]) / actual[nonzero]))

        ss_res = np.sum((actual - expected) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        rows.append({'outcome': outcome, 'mape': mape, 'r2': r2, 'n_store_days': len(sub)})

    # ── Ratio metrics: derive actual/expected from components ─────────────
    for ratio, (num_col, denom_col) in ratio_metrics.items():
        exp_num_col = f'expected_{num_col}'
        exp_denom_col = f'expected_{denom_col}'

        sub = pre_test[[num_col, denom_col, exp_num_col, exp_denom_col]].dropna()

        actual = sub[num_col].values / sub[denom_col].values.clip(min=1e-9)
        expected = sub[exp_num_col].values / sub[exp_denom_col].values.clip(min=1e-9)

        nonzero = actual != 0
        mape = np.mean(np.abs((actual[nonzero] - expected[nonzero]) / actual[nonzero]))

        ss_res = np.sum((actual - expected) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        rows.append({'outcome': ratio, 'mape': mape, 'r2': r2, 'n_store_days': len(sub)})

    metrics_df = pd.DataFrame(rows)
    metrics_df['mape'] = metrics_df['mape'].map('{:.2%}'.format)
    metrics_df['r2'] = metrics_df['r2'].map('{:.4f}'.format)
    return metrics_df