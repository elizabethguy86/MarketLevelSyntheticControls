"""
Unit-Level Synthetic Control
==============================
Fits a separate synthetic control for **each individual test unit** and
returns the original dataframe with an ``expected`` column (or
``expected_{outcome}`` columns for multiple outcomes) representing each test
unit's counterfactual trajectory.

Unlike pooled synthetic DiD methods, ``UnitLevelSyntheticControl`` fits
independent unit weights per unit, so each unit gets its own tailored
synthetic control.  Other test units are excluded from each unit's donor
pool to prevent cross-contamination.

Credit given to Causal Inference for the Brave and True for methodology assistance and coding implementation
[https://matheusfacure.github.io/python-causality-handbook/landing-page.html]
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm


# ── Helper functions ──────────────────────────────────────────────────────────

def calculate_regularization(data, outcome_col, time_col, unit_col, treat_col, post_col):
    n_treated_post = data.query(post_col).query(treat_col).shape[0]

    first_diff_std = (
        data
        .query(f"~{post_col}")
        .query(f"~{treat_col}")
        .sort_values(time_col)
        .groupby(unit_col)
        [outcome_col]
        .diff()
        .std()
    )

    return n_treated_post ** (1 / 4) * first_diff_std


def fit_unit_weights(
    data,
    outcome_col,
    time_col,
    unit_col,
    treat_col,
    post_col,
    regularization_multiplier: float = 3.0,
    tail_periods: int = None,
    zeta_override: float = None,
):
    """
    Fit unit weights for the synthetic control.

    Returns a tuple of (unit_weights, intercept) where:
    - unit_weights : pd.Series of non-negative weights summing to 1 over control units
    - intercept    : float that shifts the weighted control mean to match the test
                     unit's level, absorbing systematic level differences between
                     treatment and control units.

    Parameters
    ----------
    regularization_multiplier : float
        Scales the auto-computed zeta upward. Values > 1 push weights
        closer to uniform, reducing overfitting. Try 2–5.
    tail_periods : int or None
        If set, the last `tail_periods` pre-period dates receive
        quadratically-increasing MSE weight so the fit is tightest
        right before the intervention. Try 30–60.
    zeta_override : float or None
        If provided, uses this value directly as zeta instead of
        computing it from the data.
    """
    if zeta_override is not None:
        zeta = zeta_override
    else:
        zeta = (
            calculate_regularization(
                data, outcome_col, time_col, unit_col, treat_col, post_col
            )
            * regularization_multiplier
        )

    pre_data = data.query(f"~{post_col}")

    y_pre_control = pre_data.query(f"~{treat_col}").pivot(
        index=time_col, columns=unit_col, values=outcome_col
    )

    y_pre_treat_mean = (
        pre_data.query(f"{treat_col}")
        .groupby(time_col)[outcome_col]
        .mean()
    )

    T_pre = y_pre_control.shape[0]
    X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1)

    obs_weights = np.ones(T_pre)
    if tail_periods is not None:
        tail_n = min(tail_periods, T_pre)
        ramp = np.linspace(1, tail_n, tail_n) ** 2
        ramp = ramp / ramp.mean()
        obs_weights[-tail_n:] = ramp

    sqrt_w = np.sqrt(obs_weights)

    w = cp.Variable(X.shape[1])
    residuals = cp.multiply(sqrt_w, X @ w - y_pre_treat_mean.values)
    objective = cp.Minimize(
        cp.sum_squares(residuals) + T_pre * zeta**2 * cp.sum_squares(w[1:])
    )
    constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)

    unit_weights = pd.Series(
        w.value[1:],
        name="unit_weights",
        index=y_pre_control.columns,
    )
    intercept = float(w.value[0])

    return unit_weights, intercept


# ── Main class ────────────────────────────────────────────────────────────────

class UnitLevelSyntheticControl(BaseEstimator):
    """
    Fits a per-unit synthetic control and returns a dataframe with expected
    (counterfactual) values for each test unit.

    For each test unit a synthetic twin is constructed from the weighted
    combination of control units (non-test units) that best tracks the test
    unit during the pre-intervention period. Other test units are excluded
    from each unit's donor pool to prevent cross-contamination.

    An intercept is estimated alongside the unit weights to absorb any
    systematic level difference between a test unit and its donor pool, so
    that observed lifts reflect only changes in *trend* rather than
    pre-existing volume differences.

    Parameters
    ----------
    outcome_col : str or list of str
        Outcome column(s) for which to build synthetic controls.
    time_col : str
        Column identifying the time period.
    unit_col : str
        Column identifying the unit.
    treat_col : str
        Boolean column flagging test unit rows (True = test unit).
    post_col : str
        Boolean column flagging post-intervention rows (True = post).
    regularization_multiplier : float, default 3.0
        Scales the auto-computed zeta for unit weight regularization.
        Higher values push weights closer to uniform (less sparse).
    tail_periods : int or None, default None
        If set, the last N pre-period dates receive quadratically increasing
        MSE weight so the fit is tightest just before the intervention.
    ratio_metrics : dict or None, default None
        Mapping of derived ratio metric names to their (numerator, denominator)
        component columns, e.g. ``{'avgTicket': ('grossSales', 'transactions')}``.
        Component columns are automatically fitted even if not listed in
        ``outcome_col``. The expected ratio is derived as
        ``expected_{ratio} = expected_{numerator} / expected_{denominator}``
        rather than fitting SC directly on the ratio, avoiding the convex-hull
        and ratio-aggregation distortions that arise when SC is fit on ratio
        metrics directly.
        In ``permutation_p_values``, the portfolio ratio residual at each date
        is ``|Σ actual_num / Σ actual_denom - Σ expected_num / Σ expected_denom|``
        across all test units, and RMSPE is computed from those residuals.

    Attributes
    ----------
    unit_weights_ : dict
        After fitting, a nested dict of
        ``{unit_id: {outcome: (pd.Series of unit_weights, float intercept)}}``.

    Output columns
    --------------
    Single outcome  → column named ``expected``
    Multiple outcomes → columns named ``expected_{outcome}`` for each outcome.
    Ratio metrics → columns named ``expected_{ratio_name}`` derived from components.
    Expected values for control unit rows are left as NaN.
    """

    def __init__(
        self,
        outcome_col,
        time_col,
        unit_col,
        treat_col,
        post_col,
        regularization_multiplier: float = 3.0,
        tail_periods: int = None,
        ratio_metrics: dict = None,
    ):
        self.outcome_col = outcome_col
        self.time_col = time_col
        self.unit_col = unit_col
        self.treat_col = treat_col
        self.post_col = post_col
        self.regularization_multiplier = regularization_multiplier
        self.tail_periods = tail_periods
        self.ratio_metrics = ratio_metrics

    def _all_fit_outcomes(self):
        """
        All component outcomes to fit SC on: declared outcome_col plus any
        numerator/denominator columns from ratio_metrics not already included.
        """
        outcomes = (
            [self.outcome_col]
            if isinstance(self.outcome_col, str)
            else list(self.outcome_col)
        )
        if self.ratio_metrics:
            for num_col, denom_col in self.ratio_metrics.values():
                if num_col not in outcomes:
                    outcomes.append(num_col)
                if denom_col not in outcomes:
                    outcomes.append(denom_col)
        return outcomes

    def _expected_col_name(self, outcome):
        """
        Return the expected column name for a given outcome.
        Uses 'expected' only for a single declared outcome_col; otherwise
        'expected_{outcome}' for all outcomes including auto-added components.
        """
        if isinstance(self.outcome_col, str) and self.outcome_col == outcome:
            return "expected"
        return f"expected_{outcome}"

    def fit(self, data: pd.DataFrame):
        """
        Fit a synthetic control for each test unit.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with columns for time, unit, treatment flag, post flag,
            and outcome(s).

        Returns
        -------
        self
        """
        outcomes = self._all_fit_outcomes()

        control_mask = ~data[self.treat_col].astype(bool)
        test_units = data.loc[~control_mask, self.unit_col].unique()

        self.unit_weights_ = {}

        for unit in test_units:
            self.unit_weights_[unit] = {}

            # Donor pool: this unit + all control units (other test units excluded)
            unit_or_control = (data[self.unit_col] == unit) | control_mask
            unit_data = data.loc[unit_or_control].copy()
            unit_data["_treat_unit"] = unit_data[self.unit_col] == unit

            for outcome in outcomes:
                weights, intercept = fit_unit_weights(
                    unit_data,
                    outcome_col=outcome,
                    time_col=self.time_col,
                    unit_col=self.unit_col,
                    treat_col="_treat_unit",
                    post_col=self.post_col,
                    regularization_multiplier=self.regularization_multiplier,
                    tail_periods=self.tail_periods,
                )
                self.unit_weights_[unit][outcome] = (weights, intercept)

        self.is_fitted_ = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of `data` with expected (counterfactual) column(s) added.

        Expected values are filled for test unit rows only. Control unit rows
        receive NaN in the expected column(s).

        The intercept estimated during fitting is applied to each unit's
        synthetic control series so that pre-period level differences between
        the test unit and its donor pool do not inflate post-period lift estimates.

        For ratio metrics declared in ``ratio_metrics``, the expected ratio is
        derived as ``expected_{numerator} / expected_{denominator}`` rather than
        fitting SC directly on ratio values.

        Parameters
        ----------
        data : pd.DataFrame
            Same panel data passed to ``fit``.

        Returns
        -------
        pd.DataFrame
        """
        check_is_fitted(self)

        outcomes = self._all_fit_outcomes()
        result = data.copy()
        control_mask = ~data[self.treat_col].astype(bool)

        # Build time × control_unit pivot once per outcome (reused across units)
        control_pivots = {
            outcome: data.loc[control_mask].pivot(
                index=self.time_col, columns=self.unit_col, values=outcome
            )
            for outcome in outcomes
        }

        for outcome in outcomes:
            col_name = self._expected_col_name(outcome)
            result[col_name] = np.nan
            pivot = control_pivots[outcome]

            for unit, weights_dict in self.unit_weights_.items():
                unit_weights, intercept = weights_dict[outcome]

                aligned_w = unit_weights.reindex(pivot.columns).fillna(0)
                sc_series = pivot @ aligned_w + intercept

                unit_mask = result[self.unit_col] == unit
                result.loc[unit_mask, col_name] = (
                    result.loc[unit_mask, self.time_col].map(sc_series)
                )

        if self.ratio_metrics:
            for ratio_name, (num_col, denom_col) in self.ratio_metrics.items():
                num_expected_col = self._expected_col_name(num_col)
                denom_expected_col = self._expected_col_name(denom_col)
                result[f"expected_{ratio_name}"] = (
                    result[num_expected_col] / result[denom_expected_col]
                )

        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def permutation_p_values(
        self,
        data: pd.DataFrame,
        n_permutations: int = 1000,
        n_placebos: int = None,
        seed: int = None,
    ) -> pd.DataFrame:
        """
        Compute portfolio-level RMSPE-ratio p-values for each outcome.

        Uses the RMSPE ratio method (Abadie, Diamond & Hainmueller 2010):
        for each unit compute the ratio of post-period RMSPE to pre-period
        RMSPE.  A large ratio for the treated portfolio means the post-period
        residuals are much larger than the pre-period misfit — evidence of a
        real treatment effect.  Units with a poor pre-period fit are
        automatically down-weighted because their large pre-period RMSPE
        deflates their ratio toward 1, unlike raw-lift permutation tests
        which give every placebo equal weight.

        Algorithm
        ---------
        1. Compute the **treated portfolio RMSPE ratio**: pool all
           ((unit × date)) squared residuals across every test unit →
           pre-period RMSPE and post-period RMSPE → ratio = post / pre.
        2. For each placebo unit, fit a leave-one-out pseudo-SC and compute
           that unit's individual RMSPE ratio from its pseudo-SC residuals.
        3. p-value = fraction of placebo RMSPE ratios ≥ treated RMSPE ratio
           (+1 to numerator and denominator for conservatism;
           Phipson & Smyth 2010):
               p = (1 + #{placebo_ratio ≥ treated_ratio}) / (1 + n_valid_placebos)
        4. For ratio metrics (e.g. avgTicket), the residual at each date is
           the difference between the portfolio ratio (Σ num / Σ denom across
           all test units) and its SC counterpart; for control unit placebos
           it is the single-unit ratio difference against the pseudo-SC.

        Parameters
        ----------
        data : pd.DataFrame
            The panel data used in ``fit_transform``.
        n_permutations : int
            Unused; retained for API compatibility.
        n_placebos : int or None
            Maximum number of control units to use as placebos.  When
            ``None`` (default) all control units are used.  Set to e.g.
            ``200`` to randomly subsample the donor pool — the null
            distribution stabilises well before exhausting all units and
            p-value resolution is ``1 / (1 + n_placebos)``, so 200 gives
            0.005 resolution.  The sample is drawn without replacement and
            is reproducible when ``seed`` is set.
        seed : int or None
            Random seed for reproducibility (applies to ``n_placebos``
            subsampling).

        Returns
        -------
        pd.DataFrame
            One row per outcome (including ratio metrics) with columns:
            outcome, observed_lift, pre_rmspe, post_rmspe, rmspe_ratio,
            p_value, n_valid_placebos
        """
        check_is_fitted(self)

        all_outcomes = self._all_fit_outcomes()
        declared_outcomes = (
            [self.outcome_col]
            if isinstance(self.outcome_col, str)
            else list(self.outcome_col)
        )
        ratio_metrics = self.ratio_metrics or {}

        post_mask  = data[self.post_col].astype(bool)
        treat_mask = data[self.treat_col].astype(bool)

        test_units    = list(self.unit_weights_.keys())
        control_units = data.loc[~treat_mask, self.unit_col].unique().tolist()
        post_times     = set(data.loc[ post_mask, self.time_col].unique())
        pre_times      = set(data.loc[~post_mask, self.time_col].unique())

        # ── Shared control pivots (all control units × all time periods) ─────
        control_data = data.loc[~treat_mask]
        control_pivots = {
            outcome: control_data.pivot(
                index=self.time_col, columns=self.unit_col, values=outcome
            )
            for outcome in all_outcomes
        }

        # ── Step 1: Treated portfolio RMSPE (pre and post) ───────────────────
        result    = self.transform(data)
        test_pre  = result[ treat_mask & ~post_mask]
        test_post = result[ treat_mask &  post_mask]

        treated_rmspe       = {}
        treated_ratio_rmspe = {}

        for outcome in declared_outcomes:
            exp_col    = self._expected_col_name(outcome)
            pre_resid  = test_pre[outcome].values  - test_pre[exp_col].values
            post_resid = test_post[outcome].values - test_post[exp_col].values
            pre_rmspe  = np.sqrt(np.nanmean(pre_resid  ** 2))
            post_rmspe = np.sqrt(np.nanmean(post_resid ** 2))
            ratio      = post_rmspe / pre_rmspe if (np.isfinite(pre_rmspe) and pre_rmspe != 0) else np.nan
            treated_rmspe[outcome] = (pre_rmspe, post_rmspe, ratio)

        for ratio_name, (num_col, denom_col) in ratio_metrics.items():
            exp_num_col   = self._expected_col_name(num_col)
            exp_denom_col = self._expected_col_name(denom_col)

            def _portfolio_ratio_sq(df):
                by_date = (
                    df.groupby(self.time_col)
                    .apply(lambda g: pd.Series({
                        "act": g[num_col].sum() / g[denom_col].sum()
                              if g[denom_col].sum() != 0 else np.nan,
                        "exp": g[exp_num_col].sum() / g[exp_denom_col].sum()
                              if g[exp_denom_col].sum() != 0 else np.nan,
                    }))
                )
                return (by_date["act"] - by_date["exp"]).dropna().values ** 2

            pre_sq     = _portfolio_ratio_sq(test_pre)
            post_sq    = _portfolio_ratio_sq(test_post)
            pre_rmspe  = np.sqrt(np.mean(pre_sq))  if len(pre_sq)  > 0 else np.nan
            post_rmspe = np.sqrt(np.mean(post_sq)) if len(post_sq) > 0 else np.nan
            ratio_val  = post_rmspe / pre_rmspe if (np.isfinite(pre_rmspe) and pre_rmspe != 0) else np.nan
            treated_ratio_rmspe[ratio_name] = (pre_rmspe, post_rmspe, ratio_val)

        # ── Step 2: Observed portfolio lift (for reporting only) ─────────────
        actual_stats = {unit: {} for unit in test_units}
        for unit in test_units:
            unit_post = data.loc[(data[self.unit_col] == unit) & post_mask]
            for outcome in all_outcomes:
                pivot                    = control_pivots[outcome]
                unit_weights, intercept  = self.unit_weights_[unit][outcome]
                aligned_w                = unit_weights.reindex(pivot.columns).fillna(0)
                sc_series                = pivot @ aligned_w + intercept
                sc_post_mean             = sc_series[sc_series.index.isin(post_times)].mean()
                actual_post_mean         = unit_post[outcome].mean()
                impact                   = actual_post_mean - sc_post_mean
                actual_stats[unit][outcome] = (impact, sc_post_mean, actual_post_mean)

        # ── Step 3: Select placebo units ────────────────────────────────────
        if seed is not None:
            np.random.seed(seed)
        if n_placebos is not None and n_placebos < len(control_units):
            placebo_units = list(
                np.random.choice(control_units, size=n_placebos, replace=False)
            )
            print(f"Subsampling {n_placebos} of {len(control_units)} control units as placebos.")
        else:
            placebo_units = control_units

        # ── Step 4: Pseudo-SC for each placebo unit (leave-one-out) ─────────
        pseudo_rmspe     = {ctrl: {} for ctrl in placebo_units}
        pseudo_sc_series = {ctrl: {} for ctrl in placebo_units}

        print(f"Fitting pseudo-SCs for {len(placebo_units)} control units...")
        for ctrl_unit in tqdm(placebo_units, desc="Pseudo-SC fits"):
            donors = [c for c in control_units if c != ctrl_unit]
            pseudo_data = data.loc[
                (data[self.unit_col] == ctrl_unit) | data[self.unit_col].isin(donors)
            ].copy()
            pseudo_data["_pseudo_treat"] = pseudo_data[self.unit_col] == ctrl_unit

            donor_pivot_full = {
                outcome: control_pivots[outcome].drop(columns=[ctrl_unit], errors="ignore")
                for outcome in all_outcomes
            }

            ctrl_df = (
                data.loc[data[self.unit_col] == ctrl_unit]
                .set_index(self.time_col)
            )

            for outcome in all_outcomes:
                try:
                    weights, intercept = fit_unit_weights(
                        pseudo_data,
                        outcome_col=outcome,
                        time_col=self.time_col,
                        unit_col=self.unit_col,
                        treat_col="_pseudo_treat",
                        post_col=self.post_col,
                        regularization_multiplier=self.regularization_multiplier,
                        tail_periods=self.tail_periods,
                    )
                    donor_pivot = donor_pivot_full[outcome]
                    aligned_w   = weights.reindex(donor_pivot.columns).fillna(0)
                    sc_series   = donor_pivot @ aligned_w + intercept

                    pseudo_sc_series[ctrl_unit][outcome] = sc_series

                    pre_resid = np.array([
                        ctrl_df.loc[d, outcome] - float(sc_series.loc[d])
                        for d in pre_times
                        if d in sc_series.index and d in ctrl_df.index
                    ])
                    post_resid = np.array([
                        ctrl_df.loc[d, outcome] - float(sc_series.loc[d])
                        for d in post_times
                        if d in sc_series.index and d in ctrl_df.index
                    ])

                    pre_rmspe  = np.sqrt(np.mean(pre_resid  ** 2)) if len(pre_resid)  > 0 else np.nan
                    post_rmspe = np.sqrt(np.mean(post_resid ** 2)) if len(post_resid) > 0 else np.nan
                    ratio      = post_rmspe / pre_rmspe if (np.isfinite(pre_rmspe) and pre_rmspe != 0) else np.nan
                    pseudo_rmspe[ctrl_unit][outcome] = (pre_rmspe, post_rmspe, ratio)
                except Exception:
                    pseudo_sc_series[ctrl_unit][outcome] = None
                    pseudo_rmspe[ctrl_unit][outcome]     = (np.nan, np.nan, np.nan)

        # ── Step 5: Placebo RMSPE ratios for ratio metrics ────────────────────
        pseudo_ratio_rmspe = {ctrl: {} for ctrl in placebo_units}
        for ratio_name, (num_col, denom_col) in ratio_metrics.items():
            for ctrl_unit in placebo_units:
                sc_num   = pseudo_sc_series[ctrl_unit].get(num_col)
                sc_denom = pseudo_sc_series[ctrl_unit].get(denom_col)
                if sc_num is None or sc_denom is None:
                    pseudo_ratio_rmspe[ctrl_unit][ratio_name] = (np.nan, np.nan, np.nan)
                    continue

                ctrl_df = (
                    data.loc[data[self.unit_col] == ctrl_unit]
                    .set_index(self.time_col)
                )

                pre_sq, post_sq = [], []
                for date, sq_list in (
                    [(d, pre_sq) for d in pre_times] + [(d, post_sq) for d in post_times]
                ):
                    if (date not in sc_num.index or date not in sc_denom.index
                            or date not in ctrl_df.index):
                        continue
                    d_denom = ctrl_df.loc[date, denom_col]
                    sc_d    = float(sc_denom.loc[date])
                    if d_denom == 0 or sc_d == 0:
                        continue
                    act_r = ctrl_df.loc[date, num_col] / d_denom
                    sc_r  = float(sc_num.loc[date]) / sc_d
                    sq_list.append((act_r - sc_r) ** 2)

                pre_rmspe  = np.sqrt(np.mean(pre_sq))  if pre_sq  else np.nan
                post_rmspe = np.sqrt(np.mean(post_sq)) if post_sq else np.nan
                ratio_val  = post_rmspe / pre_rmspe if (np.isfinite(pre_rmspe) and pre_rmspe != 0) else np.nan
                pseudo_ratio_rmspe[ctrl_unit][ratio_name] = (pre_rmspe, post_rmspe, ratio_val)

        # ── Step 6: Assemble results and compute p-values ────────────────────
        rows = []

        for outcome in declared_outcomes:
            pre_r, post_r, treated_ratio = treated_rmspe[outcome]

            placebo_ratios = [
                pseudo_rmspe[ctrl][outcome][2]
                for ctrl in placebo_units
                if np.isfinite(pseudo_rmspe[ctrl][outcome][2])
            ]

            if np.isfinite(treated_ratio) and placebo_ratios:
                p_value = (1 + sum(r >= treated_ratio for r in placebo_ratios)) / (
                    1 + len(placebo_ratios)
                )
            else:
                p_value = np.nan

            total_impact   = sum(actual_stats[s][outcome][0] for s in test_units if np.isfinite(actual_stats[s][outcome][0]))
            total_expected = sum(actual_stats[s][outcome][1] for s in test_units if np.isfinite(actual_stats[s][outcome][1]))
            obs_lift = total_impact / abs(total_expected) if (total_expected != 0 and np.isfinite(total_expected)) else np.nan

            rows.append({
                "outcome":          outcome,
                "observed_lift":    obs_lift,
                "pre_rmspe":        pre_r,
                "post_rmspe":       post_r,
                "rmspe_ratio":      treated_ratio,
                "p_value":          p_value,
                "n_valid_placebos": len(placebo_ratios),
            })

        for ratio_name, (num_col, denom_col) in ratio_metrics.items():
            pre_r, post_r, treated_ratio = treated_ratio_rmspe[ratio_name]

            placebo_ratios = [
                pseudo_ratio_rmspe[ctrl][ratio_name][2]
                for ctrl in placebo_units
                if np.isfinite(pseudo_ratio_rmspe[ctrl][ratio_name][2])
            ]

            if np.isfinite(treated_ratio) and placebo_ratios:
                p_value = (1 + sum(r >= treated_ratio for r in placebo_ratios)) / (
                    1 + len(placebo_ratios)
                )
            else:
                p_value = np.nan

            total_act_num   = sum(actual_stats[s][num_col][2]   for s in test_units if np.isfinite(actual_stats[s][num_col][2]))
            total_act_denom = sum(actual_stats[s][denom_col][2]  for s in test_units if np.isfinite(actual_stats[s][denom_col][2]))
            total_exp_num   = sum(actual_stats[s][num_col][1]    for s in test_units if np.isfinite(actual_stats[s][num_col][1]))
            total_exp_denom = sum(actual_stats[s][denom_col][1]  for s in test_units if np.isfinite(actual_stats[s][denom_col][1]))

            if total_act_denom != 0 and total_exp_denom != 0:
                act_ratio_obs = total_act_num / total_act_denom
                exp_ratio_obs = total_exp_num / total_exp_denom
                obs_lift = (act_ratio_obs - exp_ratio_obs) / abs(exp_ratio_obs) if exp_ratio_obs != 0 else np.nan
            else:
                obs_lift = np.nan

            rows.append({
                "outcome":          ratio_name,
                "observed_lift":    obs_lift,
                "pre_rmspe":        pre_r,
                "post_rmspe":       post_r,
                "rmspe_ratio":      treated_ratio,
                "p_value":          p_value,
                "n_valid_placebos": len(placebo_ratios),
            })

        return pd.DataFrame(rows)
