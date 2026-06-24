"""
All Plotly chart builders.
Every function accepts a DataFrame (or dict for metrics) and returns a go.Figure.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


_TEMPLATE = "plotly_white"
_MARGIN   = dict(l=40, r=40, t=55, b=40)


def _base(fig, title="") -> go.Figure:
    fig.update_layout(
        title=title, template=_TEMPLATE,
        margin=_MARGIN, font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


# ── Executive Summary ─────────────────────────────────────────────────────────

def win_rate_trend(df: pd.DataFrame) -> go.Figure:
    m = (
        df.assign(MONTH=pd.to_datetime(df["QUOTE_DATE"]).dt.to_period("M").astype(str))
        .groupby("MONTH")
        .agg(WIN_RATE=("WON", "mean"), QUOTES=("QUOTE_GROUP_ID", "nunique"))
        .reset_index()
    )
    fig = px.line(m, x="MONTH", y="WIN_RATE", markers=True,
                  custom_data=["QUOTES"],
                  labels={"WIN_RATE": "Win Rate", "MONTH": "Month"})
    fig.update_traces(hovertemplate="Month: %{x}<br>Win Rate: %{y:.1%}<br>Quotes: %{customdata[0]:,}")
    fig.update_yaxes(tickformat=".0%")
    return _base(fig, "Win Rate Trend by Month")


def grate_trend(df: pd.DataFrame) -> go.Figure:
    won = df[df["WON"] == 1].copy()
    won["MONTH"] = pd.to_datetime(won["QUOTE_DATE"]).dt.to_period("M").astype(str)
    m = won.groupby("MONTH").agg(TOTAL_GRATE=("GRATE", "sum")).reset_index()
    fig = px.bar(m, x="MONTH", y="TOTAL_GRATE",
                 labels={"TOTAL_GRATE": "Total GRATE ($)", "MONTH": "Month"})
    fig.update_traces(hovertemplate="Month: %{x}<br>GRATE: $%{y:,.0f}")
    return _base(fig, "GRATE Trend by Month")


def quote_vs_shipment_volume(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["MONTH"] = pd.to_datetime(df2["QUOTE_DATE"]).dt.to_period("M").astype(str)
    m = df2.groupby("MONTH").agg(QUOTES=("QUOTE_GROUP_ID", "nunique"), SHIPS=("WON", "sum")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=m["MONTH"], y=m["QUOTES"], name="Quotes",    marker_color="#4C78A8")
    fig.add_bar(x=m["MONTH"], y=m["SHIPS"],  name="Shipments", marker_color="#72B7B2")
    fig.update_layout(barmode="group")
    return _base(fig, "Quote vs Shipment Volume by Month")


def top_opportunity_lanes(lane_df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = lane_df[lane_df["EXPECTED_GRATE_LIFT"] > 0].nlargest(n, "EXPECTED_GRATE_LIFT")
    if top.empty:
        fig = go.Figure()
        fig.add_annotation(text="No opportunity lanes found", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return _base(fig, f"Top {n} Opportunity Lanes by Expected GRATE Lift")

    fig = px.bar(
        top, x="EXPECTED_GRATE_LIFT", y="LANE", orientation="h",
        color="ACTUAL_WIN_RATE", color_continuous_scale="RdYlGn",
        hover_data=["QUOTE_COUNT", "ACTUAL_WIN_RATE", "GRATE"],
        labels={"EXPECTED_GRATE_LIFT": "Expected GRATE Lift ($)", "LANE": "Lane",
                "ACTUAL_WIN_RATE": "Win Rate"},
    )
    fig.update_coloraxes(colorbar_tickformat=".0%")
    return _base(fig, f"Top {n} Opportunity Lanes by Expected GRATE Lift")


def win_rate_by_dimension(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    grp = (
        df.groupby(col)
        .agg(WIN_RATE=("WON", "mean"), COUNT=("WON", "count"))
        .reset_index()
        .sort_values("WIN_RATE")
    )
    fig = px.bar(grp, x="WIN_RATE", y=col, orientation="h",
                 text=grp["COUNT"].apply(lambda x: f"{x:,}"),
                 color="WIN_RATE", color_continuous_scale="RdYlGn",
                 labels={"WIN_RATE": "Win Rate", col: col.replace("_", " ").title()})
    fig.update_xaxes(tickformat=".0%")
    fig.update_coloraxes(showscale=False)
    return _base(fig, title)


def scatter_win_rate_vs_grate(lane_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        lane_df.dropna(subset=["ACTUAL_WIN_RATE", "GRATE"]),
        x="ACTUAL_WIN_RATE", y="GRATE",
        size="QUOTE_COUNT", color="RECOMMENDATION",
        hover_name="LANE",
        hover_data=["PRODUCT", "RATE_TYPE", "QUOTE_COUNT"],
        labels={"ACTUAL_WIN_RATE": "Actual Win Rate", "GRATE": "Total GRATE ($)"},
    )
    fig.update_xaxes(tickformat=".0%")
    return _base(fig, "Win Rate vs GRATE by Lane")


# ── Drilldown ─────────────────────────────────────────────────────────────────

def drilldown_monthly(df: pd.DataFrame) -> tuple[go.Figure, go.Figure, go.Figure]:
    df2 = df.copy()
    df2["MONTH"] = pd.to_datetime(df2["QUOTE_DATE"]).dt.to_period("M").astype(str)
    m = (
        df2.groupby("MONTH")
        .agg(QUOTES=("QUOTE_GROUP_ID", "nunique"), SHIPS=("WON", "sum"),
             WIN_RATE=("WON", "mean"), GRATE=("GRATE", "sum"))
        .reset_index()
    )

    fig_vol = go.Figure()
    fig_vol.add_bar(x=m["MONTH"], y=m["QUOTES"], name="Quotes",    marker_color="#4C78A8")
    fig_vol.add_bar(x=m["MONTH"], y=m["SHIPS"],  name="Shipments", marker_color="#72B7B2")
    fig_vol.update_layout(barmode="group")
    _base(fig_vol, "Monthly Quote vs Shipment Volume")

    fig_wr = px.line(m, x="MONTH", y="WIN_RATE", markers=True,
                     labels={"WIN_RATE": "Win Rate", "MONTH": "Month"})
    fig_wr.update_yaxes(tickformat=".0%")
    _base(fig_wr, "Monthly Win Rate")

    fig_gr = px.bar(m, x="MONTH", y="GRATE",
                    labels={"GRATE": "Total GRATE ($)", "MONTH": "Month"})
    _base(fig_gr, "Monthly GRATE")

    return fig_vol, fig_wr, fig_gr


def distribution_chart(df: pd.DataFrame, col: str, title: str, is_pct: bool = False) -> go.Figure:
    vals = df[col].dropna()
    if vals.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {col}", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper")
        return _base(fig, title)
    fig = px.histogram(df, x=col, nbins=30,
                       color="WON", barmode="overlay",
                       color_discrete_map={0: "#EF553B", 1: "#00CC96"},
                       labels={col: col.replace("_", " ").title()})
    if is_pct:
        fig.update_xaxes(tickformat=".0%")
    return _base(fig, title)


def mix_chart(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    grp = df.groupby([col, "WON"]).size().reset_index(name="COUNT")
    fig = px.bar(grp, x=col, y="COUNT", color=grp["WON"].map({0: "Lost", 1: "Won"}),
                 barmode="stack", color_discrete_map={"Lost": "#EF553B", "Won": "#00CC96"},
                 labels={"COUNT": "Quote Count", col: col.replace("_", " ").title()})
    return _base(fig, title)


def won_vs_lost_comparison(df: pd.DataFrame) -> go.Figure:
    cols = ["PRICE_PER_KG", "DISCOUNT_PCT", "QUOTE_WGT_KG"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return go.Figure()

    records = []
    for col in available:
        for won_val, label in [(1, "Won"), (0, "Lost")]:
            v = df.loc[df["WON"] == won_val, col].median()
            records.append({"Metric": col, "Group": label, "Median": v})
    cmp = pd.DataFrame(records)
    fig = px.bar(cmp, x="Metric", y="Median", color="Group", barmode="group",
                 color_discrete_map={"Lost": "#EF553B", "Won": "#00CC96"})
    return _base(fig, "Won vs Lost: Median Key Metrics")


# ── Model performance ─────────────────────────────────────────────────────────

def roc_curve_chart(roc_data: dict, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode="lines",
                    name=f"ROC (AUC = {auc:.3f})", line=dict(color="#4C78A8", width=2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                    name="Random classifier")
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _base(fig, "ROC Curve")


def calibration_chart(cal_data: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=cal_data["mean_predicted"], y=cal_data["fraction_positives"],
                    mode="lines+markers", name="Model", line=dict(color="#4C78A8"))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="gray"), name="Perfect calibration")
    fig.update_layout(xaxis_title="Mean Predicted Probability",
                      yaxis_title="Fraction of Positives")
    return _base(fig, "Calibration Curve")


def confusion_matrix_chart(cm_list: list) -> go.Figure:
    cm     = np.array(cm_list)
    labels = ["Lost (0)", "Won (1)"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues", showscale=False,
        text=cm, texttemplate="%{text}",
    ))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return _base(fig, "Confusion Matrix")


def feature_importance_chart(fi_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top = fi_df.head(top_n)
    fig = px.bar(top, x="importance", y="feature", orientation="h",
                 labels={"importance": "Importance", "feature": "Feature"})
    fig.update_yaxes(autorange="reversed")
    return _base(fig, f"Top {top_n} Feature Importances")


def lift_chart(lift_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=lift_df["decile"].astype(str), y=lift_df["actual_win_rate"],
                name="Actual Win Rate", marker_color="#4C78A8")
    fig.add_scatter(x=lift_df["decile"].astype(str), y=lift_df["avg_prob"],
                    mode="lines+markers", name="Avg Predicted Prob",
                    line=dict(color="#F58518", width=2))
    fig.update_layout(xaxis_title="Probability Decile (0 = lowest)",
                      yaxis_title="Win Rate / Predicted Probability")
    return _base(fig, "Lift Chart by Probability Decile")
