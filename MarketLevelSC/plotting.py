import pandas as pd
import plotly.graph_objects as go

class SyntheticControlPlotter:
    """
    Visualizes actual vs expected (synthetic control) time series
    for test stores, aggregated across the portfolio.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of StoreLevelSyntheticControl.fit_transform().
    time_col : str
        Column identifying the time period.
    unit_col : str
        Column identifying the store/unit.
    treat_col : str
        Boolean column flagging test store rows.
    post_col : str
        Boolean column flagging post-intervention rows.
    intervention_datetime : str or datetime-like
        Date of intervention — rendered as a dotted vertical line.
    ratio_metrics : dict or None
        Same mapping passed to StoreLevelSyntheticControl, e.g.
        ``{'avgTicket': ('grossSales', 'transactions')}``.
        Required to correctly volume-weight ratio metrics in aggregation.
    """

    def __init__(
        self,
        result_df,
        time_col,
        unit_col,
        treat_col,
        post_col,
        intervention_datetime,
        ratio_metrics=None,
    ):
        self.result_df = result_df
        self.time_col = time_col
        self.unit_col = unit_col
        self.treat_col = treat_col
        self.post_col = post_col
        self.intervention_datetime = pd.to_datetime(intervention_datetime)
        self.ratio_metrics = ratio_metrics or {}

    def plot(self, outcome, yaxis_label=None, title=None, width=900, height=450):
        """
        Plot actual vs expected time series for a given outcome.

        For ratio metrics declared in ``ratio_metrics``, the portfolio-level
        series is computed as ``Σ numerator / Σ denominator`` per date
        (volume-weighted), matching the permutation p-value method.

        Parameters
        ----------
        outcome : str
            Outcome to plot. Must be either a column in result_df or a key
            in ``ratio_metrics``.
        yaxis_label : str or None
            Y-axis label. Defaults to ``outcome``.
        title : str or None
            Plot title. Defaults to ``'{outcome}: Actual vs Synthetic Control'``.
        width : int
        height : int

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        test_df = self.result_df[self.result_df[self.treat_col].astype(bool)].copy()

        if outcome in self.ratio_metrics:
            num_col, denom_col = self.ratio_metrics[outcome]
            exp_num_col = f'expected_{num_col}'
            exp_denom_col = f'expected_{denom_col}'

            agg = (
                test_df.groupby(self.time_col)
                .apply(
                    lambda g: pd.Series({
                        'actual':   g[num_col].sum() / g[denom_col].sum(),
                        'expected': g[exp_num_col].sum() / g[exp_denom_col].sum(),
                    })
                )
                .reset_index()
                .sort_values(self.time_col)
            )
        else:
            exp_col = (
                f'expected_{outcome}'
                if f'expected_{outcome}' in test_df.columns
                else 'expected'
            )
            agg = (
                test_df.groupby(self.time_col)[[outcome, exp_col]]
                .sum()
                .reset_index()
                .rename(columns={outcome: 'actual', exp_col: 'expected'})
                .sort_values(self.time_col)
            )

        intervention_ms = self.intervention_datetime.timestamp() * 1000

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=agg[self.time_col],
            y=agg['actual'],
            mode='lines',
            name=f'Actual {outcome}',
            line=dict(color='mediumseagreen', width=2),
        ))

        fig.add_trace(go.Scatter(
            x=agg[self.time_col],
            y=agg['expected'],
            mode='lines',
            name=f'Expected {outcome} (SC)',
            line=dict(color='darkorange', width=2, dash='dash'),
        ))

        fig.add_vline(
            x=intervention_ms,
            line_dash='dot',
            line_color='gray',
            line_width=2,
            annotation_text='Intervention',
            annotation_position='top right',
        )

        fig.update_layout(
            title=title or f'{outcome}: Actual vs Synthetic Control',
            xaxis_title='Date',
            yaxis_title=yaxis_label or outcome,
            template='plotly_white',
            width=width,
            height=height,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )

        return fig
