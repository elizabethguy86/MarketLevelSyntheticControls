import numpy as np 
import pandas as pd
from tqdm import tqdm
from .unit_level_synthetic_control import fit_unit_weights

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

def sc_forward_chain_cv(slsc, data, k=5):
    """
    Forward-chaining (expanding-window) cross-validation for
    StoreLevelSyntheticControl, evaluated on pre-period data only.

    The pre-period dates are divided into (k+1) consecutive chunks.
    For fold i (1-indexed):
      - Training:   chunks 1 … i   (expanding window)
      - Validation: chunk i+1      (next unseen chunk, strictly in the future)

    For each fold and each test store, SC weights are **re-fitted** on the
    training dates only (using those dates as the "pre-period" inside
    fit_unit_weights). This gives a genuine out-of-sample evaluation of how
    well the synthetic control extrapolates forward in time.

    Using the already-fitted weights from the full pre-period would leak
    future information into the weight-fitting step and overstate fit quality.

    Example with k=5 (6 chunks):
      fold 1: train [1],         validate [2]
      fold 2: train [1 2],       validate [3]
      fold 3: train [1 2 3],     validate [4]
      fold 4: train [1 2 3 4],   validate [5]
      fold 5: train [1 2 3 4 5], validate [6]

    Parameters
    ----------
    slsc : StoreLevelSyntheticControl
        A fitted StoreLevelSyntheticControl instance.
    data : pd.DataFrame
        The same panel data used in slsc.fit_transform().
    k : int, default 5
        Number of folds. Pre-period is split into k+1 chunks.

    Returns
    -------
    pd.DataFrame
        One row per (outcome, fold) with columns:
        outcome, fold, n_train_days, n_val_store_days, mape, r2,
        train_end, val_start, val_end

        Summary rows (fold='mean') with forward-chained mean MAPE and R²
        appended at the bottom for each outcome.

    Notes
    -----
    Re-fitting weights across k folds × n_test_stores × n_outcomes runs
    fit_unit_weights (a CVXPY solve) for each combination. With 5 folds,
    19 stores, and 2 outcomes this is ~190 optimization runs — expect
    several minutes of runtime.
    """
    all_fit_outcomes = slsc._all_fit_outcomes()
    ratio_metrics = slsc.ratio_metrics or {}
    declared_outcomes = (
        [slsc.outcome_col] if isinstance(slsc.outcome_col, str)
        else list(slsc.outcome_col)
    )

    pre_mask    = ~data[slsc.post_col].astype(bool)
    treat_mask  = data[slsc.treat_col].astype(bool)
    control_mask = ~treat_mask

    pre_dates = np.sort(data.loc[pre_mask, slsc.time_col].unique())
    n_chunks  = k + 1

    if len(pre_dates) < n_chunks:
        raise ValueError(
            f"Only {len(pre_dates)} unique pre-period dates — need at least "
            f"{n_chunks} for k={k} folds."
        )

    chunks      = np.array_split(pre_dates, n_chunks)
    test_stores = data.loc[treat_mask, slsc.unit_col].unique()

    rows = []

    for fold_idx in tqdm(range(k), desc="CV folds"):
        train_dates = set(np.concatenate(chunks[: fold_idx + 1]))
        val_dates   = set(chunks[fold_idx + 1])

        # ── Restrict data to train + val pre-period rows only ─────────────
        fold_data = data[
            pre_mask & data[slsc.time_col].isin(train_dates | val_dates)
        ].copy()

        # Mark val dates as the temporary "post" so fit_unit_weights trains
        # only on training dates
        fold_data["_fold_post"] = fold_data[slsc.time_col].isin(val_dates)

        # ── Per-date collection of (actual, predicted) across all stores ──
        # {outcome: list of (actual_value, predicted_value)}
        val_records = {outcome: [] for outcome in all_fit_outcomes}

        for store in test_stores:
            store_or_ctrl = (fold_data[slsc.unit_col] == store) | ~fold_data[slsc.treat_col].astype(bool)
            store_data = fold_data[store_or_ctrl].copy()
            store_data["_treat_store"] = store_data[slsc.unit_col] == store

            # Control pivot for val dates (reused across all outcomes for this store)
            ctrl_val_rows = store_data[
                ~store_data["_treat_store"].astype(bool) & store_data["_fold_post"]
            ]

            # Actual test store val values
            test_val_rows = store_data[
                store_data["_treat_store"].astype(bool) & store_data["_fold_post"]
            ].set_index(slsc.time_col)

            for outcome in all_fit_outcomes:
                try:
                    weights, intercept = fit_unit_weights(
                        store_data,
                        outcome_col=outcome,
                        time_col=slsc.time_col,
                        unit_col=slsc.unit_col,
                        treat_col="_treat_store",
                        post_col="_fold_post",
                        regularization_multiplier=slsc.regularization_multiplier,
                        tail_periods=slsc.tail_periods,
                    )
                except Exception:
                    continue

                ctrl_pivot = ctrl_val_rows.pivot(
                    index=slsc.time_col, columns=slsc.unit_col, values=outcome
                )
                aligned_w  = weights.reindex(ctrl_pivot.columns).fillna(0)
                sc_series  = ctrl_pivot @ aligned_w + intercept  # predicted, indexed by date

                # Align actual and predicted on the same dates
                actual_series = test_val_rows[outcome].reindex(sc_series.index)
                combined      = pd.DataFrame({"actual": actual_series, "predicted": sc_series}).dropna()

                for _, r in combined.iterrows():
                    val_records[outcome].append((r["actual"], r["predicted"]))

        # ── Compute metrics per declared outcome ──────────────────────────
        def _metrics(actuals, predicted):
            a, p = np.array(actuals), np.array(predicted)
            nz   = a != 0
            mape = np.mean(np.abs((a[nz] - p[nz]) / a[nz])) if nz.any() else np.nan
            ss_res = np.sum((a - p) ** 2)
            ss_tot = np.sum((a - a.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            return mape, r2

        for outcome in declared_outcomes:
            recs = val_records[outcome]
            if not recs:
                continue
            actuals, predicted = zip(*recs)
            mape, r2 = _metrics(actuals, predicted)
            rows.append({
                "outcome":          outcome,
                "fold":             fold_idx + 1,
                "n_train_days":     len(train_dates),
                "n_val_store_days": len(recs),
                "mape":             mape,
                "r2":               r2,
                "train_end":        pd.Timestamp(sorted(train_dates)[-1]).date(),
                "val_start":        pd.Timestamp(sorted(val_dates)[0]).date(),
                "val_end":          pd.Timestamp(sorted(val_dates)[-1]).date(),
            })

        # ── Ratio metrics: derive from component records ───────────────────
        for ratio, (num_col, denom_col) in ratio_metrics.items():
            num_recs   = val_records.get(num_col, [])
            denom_recs = val_records.get(denom_col, [])
            if not num_recs or not denom_recs or len(num_recs) != len(denom_recs):
                continue
            act_num,  pred_num   = zip(*num_recs)
            act_denom, pred_denom = zip(*denom_recs)

            act_ratio  = np.array(act_num)  / np.clip(act_denom,  1e-9, None)
            pred_ratio = np.array(pred_num) / np.clip(pred_denom, 1e-9, None)
            mape, r2   = _metrics(act_ratio, pred_ratio)

            rows.append({
                "outcome":          ratio,
                "fold":             fold_idx + 1,
                "n_train_days":     len(train_dates),
                "n_val_store_days": len(num_recs),
                "mape":             mape,
                "r2":               r2,
                "train_end":        pd.Timestamp(sorted(train_dates)[-1]).date(),
                "val_start":        pd.Timestamp(sorted(val_dates)[0]).date(),
                "val_end":          pd.Timestamp(sorted(val_dates)[-1]).date(),
            })

    cv_df = pd.DataFrame(rows)

    # Append forward-chained mean per outcome
    summary = (
        cv_df.groupby("outcome")[["mape", "r2"]]
        .mean()
        .reset_index()
        .assign(
            fold="mean",
            n_train_days=None,
            n_val_store_days=cv_df.groupby("outcome")["n_val_store_days"].sum().values,
            train_end=None, val_start=None, val_end=None,
        )
    )
    cv_df = pd.concat([cv_df, summary], ignore_index=True)

    display_df = cv_df.copy()
    display_df["mape"] = display_df["mape"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "")
    display_df["r2"]   = display_df["r2"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
    return display_df
