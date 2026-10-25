"""Interactive e-commerce analytics dashboard built from EDA_task.ipynb."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="E-commerce Store Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).resolve().parent / "ecommerce_sales_data.csv"
REQUIRED_COLUMNS = {
    "Year", "Month", "Category", "Region", "Units_Sold", "Revenue", "Profit",
    "Customer_Rating", "Ad_Spend",
}
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_NUMBER = {month: number for number, month in enumerate(MONTH_ORDER, start=1)}
PALETTE = {
    "navy": "#172B4D",
    "blue": "#3B82F6",
    "teal": "#14B8A6",
    "purple": "#8B5CF6",
    "amber": "#F59E0B",
    "coral": "#F97373",
    "indigo": "#6366F1",
    "slate": "#64748B",
    "line": "#E2E8F0",
}
CATEGORY_COLORS = {
    "Beauty Products": "#14B8A6",
    "Books": "#64748B",
    "Electronics": "#3B82F6",
    "Fashion": "#8B5CF6",
    "Groceries": "#F59E0B",
    "Home Decor": "#F97373",
}
REGION_COLORS = {
    "East": "#14B8A6",
    "North": "#3B82F6",
    "South": "#F59E0B",
    "West": "#64748B",
}
YEAR_COLORS = {
    2020: "#64748B",
    2021: "#3B82F6",
    2022: "#6366F1",
    2023: "#14B8A6",
    2024: "#8B5CF6",
}
REVENUE_HEATMAP_COLORS = [
    [0.0, "#F8FAFC"],
    [0.35, "#DBEAFE"],
    [0.7, "#93C5FD"],
    [1.0, "#6366F1"],
]
CORRELATION_HEATMAP_COLORS = [
    [0.0, "#F97373"],
    [0.5, "#F8FAFC"],
    [1.0, "#3B82F6"],
]
CHART_COLOR_SEQUENCE = [
    PALETTE["blue"],
    PALETTE["purple"],
    PALETTE["amber"],
    PALETTE["coral"],
    PALETTE["indigo"],
    PALETTE["teal"],
    PALETTE["slate"],
]
PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
}

st.markdown(
    """
    <style>
        .block-container {max-width: 1500px; padding: 2rem 2.25rem 3.5rem;}
        [data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E2E8F0;}
        [data-testid="stSidebar"] > div:first-child {padding: 1.25rem 1.15rem 2rem;}
        [data-testid="stSidebar"] h2 {color: #172B4D; letter-spacing: -.02em; margin-bottom: .35rem;}
        [data-testid="stSidebar"] h3 {color: #172B4D; font-size: .92rem; margin-top: 1.25rem;}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #64748B; line-height: 1.45;}
        [data-testid="stSidebar"] [data-testid="stButton"] > button,
        [data-testid="stDownloadButton"] > button,
        [data-testid="stButton"] > button {
            background: #FFFFFF; color: #115E59; border: 1px solid #99C9C2;
            border-radius: 8px; font-weight: 650; transition: background .15s ease, border-color .15s ease, box-shadow .15s ease;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] > button:hover,
        [data-testid="stDownloadButton"] > button:hover,
        [data-testid="stButton"] > button:hover {
            background: #E6F4F1; border-color: #0F766E; color: #115E59;
            box-shadow: 0 2px 6px rgba(15,118,110,.12);
        }
        [data-testid="stSidebar"] [data-testid="stButton"] > button:focus-visible,
        [data-testid="stDownloadButton"] > button:focus-visible,
        [data-testid="stButton"] > button:focus-visible {box-shadow: 0 0 0 3px rgba(15,118,110,.18);}
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background: #E6F4F1 !important; border: 1px solid #B7DED7 !important;
            border-radius: 6px !important; margin: 2px 3px 2px 0 !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {color: #115E59 !important; fill: #115E59 !important;}
        [data-testid="stMultiSelect"] [data-baseweb="select"] {
            border-color: #CBD5E1 !important; border-radius: 8px !important; background: #FFFFFF;
            min-height: 42px; transition: border-color .15s ease, box-shadow .15s ease;
        }
        [data-testid="stMultiSelect"] [data-baseweb="select"]:hover {border-color: #94A3B8 !important;}
        [data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within {
            border-color: #0F766E !important; box-shadow: 0 0 0 1px #0F766E !important;
        }
        [data-baseweb="menu"] [role="option"][aria-selected="true"] {
            background: #E6F4F1 !important; color: #115E59 !important;
        }
        [data-testid="stSidebar"] hr {border-color: #E2E8F0; margin: 1.25rem 0;}
        [data-baseweb="tab-list"] {gap: .35rem; border-bottom: 1px solid #E2E8F0; margin-bottom: 1.25rem;}
        [data-baseweb="tab-list"] [data-baseweb="tab"] {color: #64748B; padding: .65rem .8rem; transition: color .15s ease, background .15s ease;}
        [data-baseweb="tab-list"] [data-baseweb="tab"]:hover {color: #115E59; background: #F1F5F9;}
        [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] {
            color: #115E59; border-bottom-color: #0F766E; font-weight: 700;
        }
        [data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E2E8F0;
            border-radius: 12px; padding: 1.05rem 1.15rem; box-shadow: 0 3px 12px rgba(15,23,42,.045);
            min-height: 108px; transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;}
        [data-testid="stMetric"]:hover {transform: translateY(-2px); border-color: #B7DED7; box-shadow: 0 7px 18px rgba(15,23,42,.08);}
        [data-testid="stMetricLabel"] {color: #64748B; font-size: .84rem; line-height: 1.3;}
        [data-testid="stMetricValue"] {color: #172B4D; font-size: clamp(1.25rem, 3vw, 2rem); line-height: 1.15;}
        [data-testid="stMarkdownContainer"] h1 {color: #172B4D; letter-spacing: -.035em; margin-bottom: .25rem;}
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {color: #172B4D; letter-spacing: -.02em; margin-top: 1.4rem;}
        .eyebrow {color: #0F766E; font-size: .74rem; font-weight: 750;
            letter-spacing: .11em; text-transform: uppercase; margin-bottom: .2rem;}
        .insight {background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #0F766E;
            border-radius: 0 10px 10px 0; padding: .8rem 1rem; margin: .65rem 0 1.35rem;}
        .insight p {margin: .3rem 0; color: #334155; line-height: 1.45;}
        @media (max-width: 900px) {
            .block-container {padding: 1.5rem 1.25rem 3rem;}
            [data-baseweb="tab-list"] {overflow-x: auto; scrollbar-width: thin;}
            [data-baseweb="tab-list"] [data-baseweb="tab"] {white-space: nowrap;}
        }
        @media (max-width: 640px) {
            .block-container {padding: 1rem .75rem 2rem;}
            [data-testid="stMetric"] {padding: .75rem; min-height: 94px;}
            [data-testid="stMetricValue"] {font-size: 1.2rem;}
            [data-testid="stSidebar"] > div:first-child {padding: 1rem .8rem 1.5rem;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading and validating sales data…")
def load_data(path: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the original CSV and apply the cleaning documented in the notebook."""
    try:
        raw = pd.read_csv(path)
    except FileNotFoundError:
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError, PermissionError, UnicodeDecodeError) as error:
        raise ValueError(f"The dataset could not be read safely: {error}") from error

    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError("The CSV is missing required columns: " + ", ".join(sorted(missing)))

    data = raw.copy()
    quality = {
        "source_rows": len(data),
        "duplicate_rows": int(data.duplicated().sum()),
    }
    data = data.drop_duplicates().copy()

    for column in ("Month", "Category", "Region"):
        data[column] = data[column].astype("string").str.strip()
    data["Month"] = data["Month"].str.title()

    numeric_columns = ["Year", "Units_Sold", "Revenue", "Profit", "Customer_Rating", "Ad_Spend"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["Month_Number"] = data["Month"].map(MONTH_NUMBER)
    data["Profit_Margin"] = np.where(
        data["Revenue"].notna() & data["Revenue"].ne(0),
        data["Profit"].div(data["Revenue"]).mul(100),
        np.nan,
    )
    data["Ad_Spend_Percentage"] = np.where(
        data["Revenue"].notna() & data["Revenue"].ne(0),
        data["Ad_Spend"].div(data["Revenue"]).mul(100),
        np.nan,
    )
    quality["rows_after_deduplication"] = len(data)
    quality["missing_numeric_values"] = int(data[numeric_columns].isna().sum().sum())
    quality["unrecognised_months"] = int(data["Month_Number"].isna().sum())
    return data, quality


def money(value: float) -> str:
    if pd.isna(value):
        return "—"
    value = float(value)
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(value) >= divisor:
            return f"${value / divisor:,.2f}{suffix}"
    return f"${value:,.0f}"


def chart_style(
    fig: go.Figure,
    title: str,
    height: int = 380,
    legend_orientation: str = "h",
    margins: dict[str, int] | None = None,
) -> go.Figure:
    chart_margins = margins or dict(l=18, r=18, t=92, b=54)
    if legend_orientation == "v":
        legend = dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,.85)", bordercolor=PALETTE["line"], borderwidth=1,
        )
    else:
        legend = dict(
            orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0,
            bgcolor="rgba(255,255,255,.85)",
        )
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top", "font": {"size": 17, "color": PALETTE["navy"]}},
        height=height,
        margin=chart_margins,
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=CHART_COLOR_SEQUENCE,
        font=dict(color=PALETTE["navy"], family="Arial, sans-serif"),
        legend=legend,
        hovermode="closest",
        uniformtext=dict(minsize=10, mode="hide"),
        hoverlabel=dict(bgcolor="white", font_color=PALETTE["navy"]),
    )
    fig.update_xaxes(showgrid=False, linecolor=PALETTE["line"], automargin=True)
    fig.update_yaxes(gridcolor=PALETTE["line"], zerolinecolor=PALETTE["line"], automargin=True)
    return fig


def category_bar_style(fig: go.Figure) -> go.Figure:
    """Keep categorical tick labels readable on narrow, responsive chart columns."""
    return fig.update_xaxes(tickangle=-28, automargin=True)


def filtered_rows(data: pd.DataFrame, years, months, categories, regions) -> pd.DataFrame:
    return data.loc[
        data["Year"].isin(years)
        & data["Month"].isin(months)
        & data["Category"].isin(categories)
        & data["Region"].isin(regions)
    ].copy()


def category_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("Category", observed=True)[["Revenue", "Profit", "Units_Sold"]]
        .sum(min_count=1)
        .assign(Profit_Margin=lambda x: np.where(x["Revenue"].ne(0), x["Profit"].div(x["Revenue"]).mul(100), np.nan))
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )


def region_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("Region", observed=True)[["Revenue", "Profit", "Units_Sold"]]
        .sum(min_count=1)
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )


def correlation(data: pd.DataFrame, left: str, right: str) -> float:
    valid = data[[left, right]].dropna()
    if len(valid) < 2 or valid[left].nunique() < 2 or valid[right].nunique() < 2:
        return np.nan
    return float(valid[left].corr(valid[right]))


def insight_card(observation: str, insight: str, decision: str) -> None:
    st.markdown(
        f"""<div class="insight">
        <p><strong>Observation</strong> — {observation}</p>
        <p><strong>Insight</strong> — {insight}</p>
        <p><strong>Business decision</strong> — {decision}</p>
        </div>""",
        unsafe_allow_html=True,
    )


try:
    df, quality = load_data(str(DATA_PATH))
except FileNotFoundError:
    st.error(f"Dataset not found: `{DATA_PATH.name}`. Place it beside `app.py` and reload.")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()

all_years = sorted(df["Year"].dropna().astype(int).unique().tolist())
all_months = [month for month in MONTH_ORDER if month in set(df["Month"].dropna())]
all_categories = sorted(df["Category"].dropna().unique().tolist())
all_regions = sorted(df["Region"].dropna().unique().tolist())
filter_defaults = {
    "filter_years": all_years,
    "filter_months": all_months,
    "filter_categories": all_categories,
    "filter_regions": all_regions,
}
for filter_key, defaults in filter_defaults.items():
    if filter_key not in st.session_state:
        st.session_state[filter_key] = defaults.copy()

with st.sidebar:
    st.markdown("## Store insights")
    st.caption("Use any combination of filters. KPIs, charts, tables, and recommendations update together.")
    if st.button("Reset all filters", width="stretch"):
        for filter_key, defaults in filter_defaults.items():
            st.session_state[filter_key] = defaults.copy()

    st.markdown("### Filter the analysis")
    years = st.multiselect("Year", all_years, key="filter_years")
    months = st.multiselect("Month", all_months, key="filter_months")
    categories = st.multiselect("Product category", all_categories, key="filter_categories")
    regions = st.multiselect("Region", all_regions, key="filter_regions")
    st.divider()
    st.caption(f"Source: `{DATA_PATH.name}` · {quality['rows_after_deduplication']:,} validated records")

selected = filtered_rows(df, years, months, categories, regions)

st.markdown('<div class="eyebrow">E-commerce analytics portfolio</div>', unsafe_allow_html=True)
st.title("E-commerce store insights")
st.caption("A filterable view of revenue, profitability, sales volume, advertising relationships, and regional opportunities.")

if quality["duplicate_rows"]:
    st.info(f"{quality['duplicate_rows']:,} exact duplicate source rows were removed, matching the notebook cleaning step.")
if quality["missing_numeric_values"] or quality["unrecognised_months"]:
    st.warning(
        f"{quality['missing_numeric_values']} invalid/missing numeric values and "
        f"{quality['unrecognised_months']} unrecognised month values were retained but excluded where calculations require them."
    )

total_revenue = selected["Revenue"].sum(min_count=1)
total_profit = selected["Profit"].sum(min_count=1)
total_units = selected["Units_Sold"].sum(min_count=1)
overall_margin = total_profit / total_revenue * 100 if pd.notna(total_revenue) and total_revenue != 0 else np.nan

kpi = st.columns(4)
kpi[0].metric("Total revenue", money(total_revenue if pd.notna(total_revenue) else 0))
kpi[1].metric("Total profit", money(total_profit if pd.notna(total_profit) else 0))
kpi[2].metric("Total units sold", f"{int(total_units):,}" if pd.notna(total_units) else "0")
kpi[3].metric("Overall profit margin", f"{overall_margin:.2f}%" if pd.notna(overall_margin) else "—")

if selected.empty:
    st.warning("No records match the current filters. Adjust a filter or reset the analysis.")
    st.stop()

overview, performance, trends, insights, explorer = st.tabs(
    ["Overview", "Performance", "Trends & relationships", "Business insights", "Data explorer"]
)

with overview:
    st.subheader("Revenue and profit distribution")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(selected, x="Revenue", nbins=30, color_discrete_sequence=[PALETTE["blue"]])
        fig.update_traces(hovertemplate="Revenue: $%{x:,.0f}<br>Records: %{y}<extra></extra>")
        chart_style(fig, "Revenue distribution").update_xaxes(tickprefix="$", tickformat="~s")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    with right:
        fig = px.histogram(selected, x="Profit", nbins=30, color_discrete_sequence=[PALETTE["purple"]])
        fig.update_traces(hovertemplate="Profit: $%{x:,.0f}<br>Records: %{y}<extra></extra>")
        chart_style(fig, "Profit distribution").update_xaxes(tickprefix="$", tickformat="~s")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    st.subheader("Category and regional snapshot")
    category_data = category_summary(selected)
    region_data = region_summary(selected)
    left, right = st.columns(2)
    with left:
        fig = px.bar(category_data, x="Revenue", y="Category", orientation="h", text_auto=".2s", color="Category", color_discrete_map=CATEGORY_COLORS)
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>Revenue: $%{x:,.2f}<extra></extra>")
        chart_style(fig, "Revenue by product category", 460, legend_orientation="v", margins=dict(l=142, r=132, t=112, b=34)).update_xaxes(tickprefix="$", tickformat="~s", title=None)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    with right:
        fig = px.bar(region_data, x="Region", y="Revenue", color="Region", text_auto=".2s", color_discrete_map=REGION_COLORS)
        fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>Revenue: $%{y:,.2f}<extra></extra>")
        chart_style(fig, "Revenue by region", 460, legend_orientation="v", margins=dict(l=18, r=132, t=112, b=82)).update_xaxes(tickangle=-20).update_yaxes(tickprefix="$", tickformat="~s", title=None)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    avg_rating = selected["Customer_Rating"].mean()
    avg_record_revenue = selected["Revenue"].mean()
    ad_ratio = selected["Ad_Spend_Percentage"].mean()
    st.subheader("Additional signals")
    signal = st.columns(3)
    signal[0].metric("Average customer rating", f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "—")
    signal[1].metric("Average revenue per record", money(avg_record_revenue))
    signal[2].metric("Average ad spend (% of revenue)", f"{ad_ratio:.2f}%" if pd.notna(ad_ratio) else "—")

with performance:
    st.subheader("Revenue and profit by region")
    region_long = region_data.melt(id_vars="Region", value_vars=["Revenue", "Profit"], var_name="Metric", value_name="Amount")
    fig = px.bar(region_long, x="Region", y="Amount", color="Metric", barmode="group", color_discrete_map={"Revenue": PALETTE["blue"], "Profit": PALETTE["purple"]})
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: $%{y:,.2f}<extra></extra>")
    chart_style(fig, "Regional revenue and profit").update_yaxes(tickprefix="$", tickformat="~s", title=None)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    st.subheader("Units sold and profit margin by category")
    left, right = st.columns(2)
    with left:
        fig = px.bar(category_data.sort_values("Units_Sold", ascending=False), x="Category", y="Units_Sold", color_discrete_sequence=[PALETTE["amber"]])
        fig.update_traces(hovertemplate="%{x}<br>Units sold: %{y:,.0f}<extra></extra>")
        chart_style(fig, "Units sold by category", 410, margins=dict(l=18, r=18, t=92, b=94))
        category_bar_style(fig).update_yaxes(title=None)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    with right:
        fig = px.bar(category_data.sort_values("Profit_Margin", ascending=False), x="Category", y="Profit_Margin", color_discrete_sequence=[PALETTE["teal"]])
        fig.update_traces(hovertemplate="%{x}<br>Profit margin: %{y:.2f}%<extra></extra>")
        chart_style(fig, "Profit margin by category", 410, margins=dict(l=18, r=18, t=92, b=94))
        category_bar_style(fig).update_yaxes(ticksuffix="%", title=None)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    st.subheader("Category × region revenue")
    matrix = selected.pivot_table(index="Category", columns="Region", values="Revenue", aggfunc="sum", observed=True)
    matrix = matrix.reindex(
        index=[x for x in all_categories if x in set(selected["Category"])],
        columns=[x for x in all_regions if x in set(selected["Region"])],
    )
    fig = go.Figure(go.Heatmap(
        z=matrix.fillna(0).values,
        x=matrix.columns,
        y=matrix.index,
        colorscale=REVENUE_HEATMAP_COLORS,
        text=matrix.fillna(0).values,
        texttemplate="$%{text:,.0f}",
        hovertemplate="Category: %{y}<br>Region: %{x}<br>Revenue: $%{z:,.2f}<extra></extra>",
        colorbar=dict(tickprefix="$", tickformat="~s"),
    ))
    chart_style(fig, "Revenue by category and region", 430, margins=dict(l=142, r=72, t=96, b=72))
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

with trends:
    st.subheader("Revenue trend over time")
    monthly = (
        selected.groupby(["Year", "Month", "Month_Number"], observed=True)["Revenue"]
        .sum(min_count=1)
        .reset_index()
        .sort_values(["Year", "Month_Number"])
    )
    fig = px.line(monthly, x="Month", y="Revenue", color="Year", markers=True, category_orders={"Month": MONTH_ORDER}, color_discrete_map=YEAR_COLORS)
    fig.update_traces(hovertemplate="Year %{fullData.name}<br>%{x}<br>Revenue: $%{y:,.2f}<extra></extra>")
    chart_style(fig, "Monthly revenue by year", 410).update_yaxes(tickprefix="$", tickformat="~s", title=None)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    yearly = selected.groupby("Year", as_index=False, observed=True)["Revenue"].sum(min_count=1)
    fig = px.bar(yearly, x="Year", y="Revenue", text_auto=".2s", color_discrete_sequence=[PALETTE["indigo"]])
    fig.update_traces(hovertemplate="%{x}<br>Revenue: $%{y:,.2f}<extra></extra>")
    chart_style(fig, "Revenue by year").update_yaxes(tickprefix="$", tickformat="~s", title=None)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    left, right = st.columns((1.15, 0.85))
    with left:
        st.subheader("Advertising spend vs revenue")
        valid = selected.dropna(subset=["Ad_Spend", "Revenue"])
        ad_corr = correlation(selected, "Ad_Spend", "Revenue")
        fig = px.scatter(valid, x="Ad_Spend", y="Revenue", color="Category", custom_data=["Region", "Units_Sold", "Profit"], color_discrete_map=CATEGORY_COLORS, opacity=.75)
        fig.update_traces(marker=dict(size=9), hovertemplate="Category: %{fullData.name}<br>Ad spend: $%{x:,.2f}<br>Revenue: $%{y:,.2f}<br>Region: %{customdata[0]}<br>Units: %{customdata[1]:,.0f}<br>Profit: $%{customdata[2]:,.2f}<extra></extra>")
        chart_style(fig, "Advertising spend vs revenue", 410, legend_orientation="v", margins=dict(l=18, r=148, t=112, b=68)).update_xaxes(tickprefix="$", tickformat="~s", title="Advertising spend").update_yaxes(tickprefix="$", tickformat="~s", title="Revenue")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        st.caption(f"Pearson correlation: **{ad_corr:.4f}**" if pd.notna(ad_corr) else "Pearson correlation: insufficient valid records")
    with right:
        st.subheader("Numerical correlation heatmap")
        numerical = ["Year", "Units_Sold", "Revenue", "Profit", "Customer_Rating", "Ad_Spend", "Profit_Margin", "Ad_Spend_Percentage"]
        corr = selected[numerical].corr()
        labels = [name.replace("_", " ") for name in corr.columns]
        fig = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels, zmin=-1, zmax=1, zmid=0, colorscale=CORRELATION_HEATMAP_COLORS, text=corr.values, texttemplate="%{text:.2f}", hovertemplate="%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>", colorbar=dict(title="r")))
        chart_style(fig, "Correlation heatmap", 410, margins=dict(l=126, r=72, t=104, b=112))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

with insights:
    st.subheader("Business insights")
    st.caption("These observations recalculate with the active filters. Correlation describes association and does not prove causation.")

    st.markdown("### Case 1 · Advertising spend → revenue")
    valid = selected.dropna(subset=["Ad_Spend", "Revenue"])
    ad_corr = correlation(selected, "Ad_Spend", "Revenue")
    fig = px.scatter(valid, x="Ad_Spend", y="Revenue", color="Category", color_discrete_map=CATEGORY_COLORS, opacity=.75)
    fig.update_traces(marker=dict(size=9), hovertemplate="Category: %{fullData.name}<br>Ad spend: $%{x:,.2f}<br>Revenue: $%{y:,.2f}<extra></extra>")
    chart_style(fig, "Graph · advertising spend vs revenue", 410, legend_orientation="v", margins=dict(l=18, r=148, t=112, b=68)).update_xaxes(tickprefix="$", tickformat="~s").update_yaxes(tickprefix="$", tickformat="~s")
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    if pd.notna(ad_corr):
        strength = "negligible" if abs(ad_corr) < .10 else "weak" if abs(ad_corr) < .30 else "moderate" if abs(ad_corr) < .50 else "strong"
        direction = "positive" if ad_corr > 0 else "negative" if ad_corr < 0 else "no"
        insight_card(
            f"The selected records show a {strength} {direction} linear relationship (r = {ad_corr:.4f}) between ad spend and revenue.",
            "Advertising expenditure alone does not explain revenue performance, and this relationship does not establish causation.",
            "Evaluate campaign conversion and incremental sales by category and region before increasing or reallocating ad budgets.",
        )
    else:
        st.info("At least two valid advertising and revenue records are needed for this analysis.")

    st.markdown("### Case 2 · Product category → revenue")
    category_data = category_summary(selected)
    fig = px.bar(category_data, x="Category", y="Revenue", color="Category", text_auto=".2s", color_discrete_map=CATEGORY_COLORS)
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>Revenue: $%{y:,.2f}<extra></extra>")
    chart_style(fig, "Graph · revenue by category", 430, legend_orientation="v", margins=dict(l=18, r=132, t=112, b=96)).update_yaxes(tickprefix="$", tickformat="~s", title=None)
    category_bar_style(fig)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    highest = category_data.iloc[0]
    lowest = category_data.iloc[-1]
    insight_card(
        f"{highest['Category']} leads revenue at {money(highest['Revenue'])}; {lowest['Category']} is lowest at {money(lowest['Revenue'])}.",
        "Revenue contribution differs materially by category, but revenue alone does not determine profitability or explain the underlying cause.",
        f"Investigate the assortment and demand drivers behind {highest['Category']}; review pricing, costs, margin, and marketing opportunities for {lowest['Category']}.",
    )

    st.markdown("### Case 3 · Product category × region → revenue")
    combinations = selected.groupby(["Category", "Region"], observed=True)["Revenue"].sum(min_count=1)
    matrix = combinations.unstack("Region").reindex(
        index=[x for x in all_categories if x in set(selected["Category"])],
        columns=[x for x in all_regions if x in set(selected["Region"])],
    )
    fig = go.Figure(go.Heatmap(z=matrix.fillna(0).values, x=matrix.columns, y=matrix.index, colorscale=REVENUE_HEATMAP_COLORS, text=matrix.fillna(0).values, texttemplate="$%{text:,.0f}", hovertemplate="Category: %{y}<br>Region: %{x}<br>Revenue: $%{z:,.2f}<extra></extra>", colorbar=dict(tickprefix="$", tickformat="~s")))
    chart_style(fig, "Graph · category-region revenue heatmap", 430, margins=dict(l=142, r=72, t=96, b=72))
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    strongest = combinations.idxmax()
    weakest = combinations.idxmin()
    insight_card(
        f"{strongest[0]} in {strongest[1]} is strongest at {money(combinations.max())}; {weakest[0]} in {weakest[1]} is weakest at {money(combinations.min())}.",
        "Category performance varies by region, so one broad strategy may miss local strengths and weaknesses.",
        f"Investigate demand, availability, price, and campaign context for {strongest[0]} / {strongest[1]}, and validate the drivers of {weakest[0]} / {weakest[1]} before changing inventory or promotions.",
    )

with explorer:
    st.subheader("Filtered records")
    st.caption(f"{len(selected):,} records selected from {quality['rows_after_deduplication']:,} validated records.")
    display_columns = ["Year", "Month", "Category", "Region", "Units_Sold", "Revenue", "Profit", "Profit_Margin", "Customer_Rating", "Ad_Spend", "Ad_Spend_Percentage"]
    display_data = selected[display_columns + ["Month_Number"]].sort_values(["Year", "Month_Number", "Category"]).drop(columns="Month_Number")
    st.dataframe(
        display_data,
        width="stretch",
        hide_index=True,
        height=440,
        column_config={
            "Revenue": st.column_config.NumberColumn(format="$%.2f"),
            "Profit": st.column_config.NumberColumn(format="$%.2f"),
            "Profit_Margin": st.column_config.NumberColumn("Profit margin (%)", format="%.2f%%"),
            "Customer_Rating": st.column_config.NumberColumn(format="%.1f"),
            "Ad_Spend": st.column_config.NumberColumn("Ad spend", format="$%.2f"),
            "Ad_Spend_Percentage": st.column_config.NumberColumn("Ad spend (% of revenue)", format="%.2f%%"),
        },
    )
    st.download_button("Download filtered data (CSV)", data=display_data.to_csv(index=False).encode("utf-8"), file_name="filtered_ecommerce_sales.csv", mime="text/csv", width="content")
