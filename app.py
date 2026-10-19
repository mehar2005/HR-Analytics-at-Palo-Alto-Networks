
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Employee Engagement & Burnout Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Employee Engagement, Satisfaction & Burnout Diagnostics")
st.caption(
    "An interactive HR analytics platform for exploring employee engagement, "
    "work-life balance, workload indicators and attrition."
)

RATING_COLUMNS = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction"
]

REQUIRED_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction",
    "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole",
    "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
]


def prepare_data(data):
    data = data.copy()

    if "Attrition" in data.columns:
        if data["Attrition"].dtype == "object":
            data["Attrition"] = (
                data["Attrition"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "yes": 1, "no": 0,
                    "true": 1, "false": 0,
                    "1": 1, "0": 0
                })
            )
        else:
            data["Attrition"] = pd.to_numeric(
                data["Attrition"], errors="coerce"
            )

    rating_columns = [
        col for col in RATING_COLUMNS if col in data.columns
    ]

    if rating_columns:
        scaled_ratings = pd.DataFrame(index=data.index)

        for col in rating_columns:
            values = pd.to_numeric(data[col], errors="coerce")
            scaled_ratings[col] = ((values - 1) / 3 * 100).clip(0, 100)

        data["EngagementIndex"] = scaled_ratings.mean(axis=1)
    else:
        data["EngagementIndex"] = np.nan

    if "WorkLifeBalance" in data.columns:
        values = pd.to_numeric(
            data["WorkLifeBalance"], errors="coerce"
        )
        data["WorkLifeBalanceIndex"] = (
            (values - 1) / 3 * 100
        ).clip(0, 100)
        low_wlb = values.le(2).fillna(False)
    else:
        data["WorkLifeBalanceIndex"] = np.nan
        low_wlb = pd.Series(False, index=data.index)

    if "OverTime" in data.columns:
        overtime = (
            data["OverTime"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "true", "1"])
        )
    else:
        overtime = pd.Series(False, index=data.index)

    if "BusinessTravel" in data.columns:
        frequent_travel = (
            data["BusinessTravel"]
            .astype(str)
            .str.lower()
            .str.contains("frequent", na=False)
        )
    else:
        frequent_travel = pd.Series(False, index=data.index)

    data["BurnoutRiskScore"] = (
        overtime.astype(int) + low_wlb.astype(int)
    )

    data["BurnoutRiskCategory"] = pd.cut(
        data["BurnoutRiskScore"],
        bins=[-1, 0, 1, 2],
        labels=["Low", "Medium", "High"]
    )

    data["WorkloadStressIndicator"] = (
        overtime.astype(int) + frequent_travel.astype(int)
    )

    if rating_columns:
        ratings = data[rating_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        rating_range = ratings.max(axis=1) - ratings.min(axis=1)

        data["SatisfactionStabilityScore"] = (
            100 - rating_range / 3 * 100
        ).clip(0, 100)
    else:
        data["SatisfactionStabilityScore"] = np.nan

    if "Attrition" in data.columns:
        data["AttritionLabel"] = data["Attrition"].map({
            0: "No",
            1: "Yes"
        }).fillna("Unknown")

    if "YearsAtCompany" in data.columns:
        years = pd.to_numeric(
            data["YearsAtCompany"], errors="coerce"
        )

        data["CareerStage"] = pd.cut(
            years,
            bins=[-np.inf, 2, 5, 10, np.inf],
            labels=[
                "0–2 years",
                "3–5 years",
                "6–10 years",
                "10+ years"
            ],
            right=True
        )

    return data


def data_quality_report(data):
    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    report = pd.DataFrame({
        "Check": [
            "Number of rows",
            "Number of columns",
            "Missing cells",
            "Duplicate rows"
        ],
        "Result": [
            len(data),
            len(data.columns),
            int(data.isna().sum().sum()),
            int(data.duplicated().sum())
        ]
    })

    return report, missing


def group_attrition(data, column):
    result = (
        data.groupby(column, dropna=False)
        .agg(
            Employees=("Attrition", "size"),
            AttritionRate=("Attrition", "mean")
        )
        .reset_index()
    )

    result["AttritionRate"] *= 100

    return result


def safe_mean(data, column):
    if column not in data.columns:
        return None

    values = pd.to_numeric(data[column], errors="coerce")

    if values.notna().sum() == 0:
        return None

    return values.mean()


with st.sidebar:
    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload employee HR dataset",
        type=["csv"]
    )

if uploaded_file is None:
    st.info("Upload your employee dataset to start the analysis.")

    st.markdown("### Platform features")

    st.write(
        "- Data quality assessment\n"
        "- Univariate, bivariate and multivariate EDA\n"
        "- Employee engagement and satisfaction indicators\n"
        "- Attrition analysis\n"
        "- Workload and burnout-risk indicators\n"
        "- Career-stage analysis\n"
        "- Manager action recommendations\n"
        "- Filtered dataset export"
    )

    st.stop()

try:
    original_df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the uploaded file: {error}")
    st.stop()

if original_df.empty:
    st.error("The uploaded dataset contains no rows.")
    st.stop()

df = prepare_data(original_df)

missing_required = [
    col for col in REQUIRED_COLUMNS
    if col not in original_df.columns
]

if missing_required:
    st.warning(
        f"{len(missing_required)} expected columns are missing. "
        "Some analyses may be unavailable."
    )

with st.sidebar:
    st.header("Filters")

    filtered = df.copy()

    filter_columns = [
        "Department",
        "JobRole",
        "Gender",
        "BusinessTravel",
        "OverTime"
    ]

    for col in filter_columns:
        if col in filtered.columns:
            options = sorted(
                filtered[col].dropna().astype(str).unique().tolist()
            )

            selected = st.multiselect(
                col,
                options,
                default=options,
                key=f"filter_{col}"
            )

            filtered = filtered[
                filtered[col].astype(str).isin(selected)
            ]

    if "YearsAtCompany" in filtered.columns:
        years = pd.to_numeric(
            filtered["YearsAtCompany"], errors="coerce"
        ).dropna()

        if not years.empty:
            minimum = int(years.min())
            maximum = int(years.max())

            if minimum < maximum:
                year_range = st.slider(
                    "Years at company",
                    minimum,
                    maximum,
                    (minimum, maximum)
                )

                numeric_years = pd.to_numeric(
                    filtered["YearsAtCompany"], errors="coerce"
                )

                filtered = filtered[
                    numeric_years.between(
                        year_range[0], year_range[1]
                    )
                ]

    if filtered["EngagementIndex"].notna().any():
        minimum_engagement = st.slider(
            "Minimum engagement index",
            0,
            100,
            0
        )

        filtered = filtered[
            filtered["EngagementIndex"].isna()
            | (filtered["EngagementIndex"] >= minimum_engagement)
        ]

if filtered.empty:
    st.warning("No employees match the selected filters.")
    st.stop()


overview_tab, quality_tab, eda_tab, attrition_tab, risk_tab, career_tab, actions_tab = st.tabs([
    "Overview",
    "Data Quality",
    "Exploratory Analysis",
    "Attrition",
    "Burnout Risk",
    "Career Stage",
    "Manager Actions"
])


with overview_tab:
    st.subheader("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Employees in view",
        f"{len(filtered):,}"
    )

    if "Attrition" in filtered.columns:
        attrition_rate = safe_mean(filtered, "Attrition")

        col2.metric(
            "Observed attrition",
            f"{attrition_rate * 100:.2f}%"
            if attrition_rate is not None else "N/A"
        )
    else:
        col2.metric("Observed attrition", "N/A")

    engagement = safe_mean(filtered, "EngagementIndex")

    col3.metric(
        "Mean engagement index",
        f"{engagement:.2f}/100"
        if engagement is not None else "N/A"
    )

    high_risk_share = (
        filtered["BurnoutRiskCategory"].eq("High").mean() * 100
    )

    col4.metric(
        "High-risk indicator share",
        f"{high_risk_share:.2f}%"
    )

    left, right = st.columns(2)

    with left:
        if (
            "Department" in filtered.columns
            and filtered["EngagementIndex"].notna().any()
        ):
            department_engagement = (
                filtered.groupby("Department", dropna=False)
                ["EngagementIndex"].mean()
                .reset_index()
            )

            fig = px.bar(
                department_engagement,
                x="Department",
                y="EngagementIndex",
                title="Mean engagement index by department",
                text_auto=".1f"
            )

            st.plotly_chart(fig, use_container_width=True)

    with right:
        risk_counts = (
            filtered["BurnoutRiskCategory"]
            .value_counts()
            .rename_axis("Risk category")
            .reset_index(name="Employees")
        )

        fig = px.bar(
            risk_counts,
            x="Risk category",
            y="Employees",
            title="Burnout-risk indicator distribution",
            text="Employees"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These indicators are exploratory proxies, not validated clinical "
        "burnout measurements or evidence of causation."
    )


with quality_tab:
    st.subheader("Data Quality Assessment")

    report, missing_values = data_quality_report(original_df)

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    if missing_values.empty:
        st.success("No missing values were detected.")
    else:
        st.write("Missing values by column")

        st.dataframe(
            missing_values.rename("Missing values")
            .rename_axis("Column")
            .reset_index(),
            use_container_width=True,
            hide_index=True
        )

    if missing_required:
        st.write("Expected columns not found")

        st.dataframe(
            pd.DataFrame({"Missing expected columns": missing_required}),
            use_container_width=True,
            hide_index=True
        )

    st.write("Column data types")

    dtypes = pd.DataFrame({
        "Column": original_df.columns,
        "Data type": original_df.dtypes.astype(str).values
    })

    st.dataframe(dtypes, use_container_width=True, hide_index=True)

    st.write("Dataset preview")

    st.dataframe(
        original_df.head(30),
        use_container_width=True,
        hide_index=True
    )


with eda_tab:
    st.subheader("Exploratory Data Analysis")

    analysis_type = st.radio(
        "Choose analysis",
        ["Univariate", "Bivariate", "Correlation"],
        horizontal=True
    )

    numeric_columns = filtered.select_dtypes(
        include=np.number
    ).columns.tolist()

    if analysis_type == "Univariate":
        selected_column = st.selectbox(
            "Select a column",
            filtered.columns.tolist()
        )

        if selected_column in numeric_columns:
            fig = px.histogram(
                filtered,
                x=selected_column,
                marginal="box",
                title=f"Distribution of {selected_column}"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                filtered[selected_column].describe().to_frame("Statistics"),
                use_container_width=True
            )
        else:
            counts = (
                filtered[selected_column]
                .astype(str)
                .value_counts()
                .head(30)
                .rename_axis(selected_column)
                .reset_index(name="Count")
            )

            fig = px.bar(
                counts,
                x=selected_column,
                y="Count",
                title=f"Distribution of {selected_column}",
                text="Count"
            )

            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Bivariate":
        x_column = st.selectbox(
            "Select X-axis",
            filtered.columns.tolist(),
            key="bivariate_x"
        )

        possible_y = [
            col for col in filtered.columns
            if col != x_column
        ]

        y_column = st.selectbox(
            "Select Y-axis",
            possible_y,
            key="bivariate_y"
        )

        x_numeric = x_column in numeric_columns
        y_numeric = y_column in numeric_columns

        if x_numeric and y_numeric:
            fig = px.scatter(
                filtered,
                x=x_column,
                y=y_column,
                title=f"{y_column} versus {x_column}"
            )
        elif x_numeric:
            fig = px.box(
                filtered,
                x=y_column,
                y=x_column,
                title=f"{x_column} by {y_column}"
            )
        elif y_numeric:
            fig = px.box(
                filtered,
                x=x_column,
                y=y_column,
                title=f"{y_column} by {x_column}"
            )
        else:
            contingency = pd.crosstab(
                filtered[x_column].astype(str),
                filtered[y_column].astype(str)
            )

            fig = px.imshow(
                contingency,
                text_auto=True,
                aspect="auto",
                title=f"{x_column} versus {y_column}"
            )

        st.plotly_chart(fig, use_container_width=True)

    else:
        if len(numeric_columns) >= 2:
            correlation = filtered[numeric_columns].corr()

            fig = px.imshow(
                correlation,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                title="Correlation matrix"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two numeric columns are required.")


with attrition_tab:
    st.subheader("Attrition Analysis")

    if "Attrition" not in filtered.columns:
        st.warning("The Attrition column is unavailable.")
    elif filtered["Attrition"].notna().sum() == 0:
        st.warning("The Attrition column could not be interpreted.")
    else:
        attrition_distribution = (
            filtered["AttritionLabel"]
            .value_counts()
            .rename_axis("Attrition")
            .reset_index(name="Employees")
        )

        left, right = st.columns(2)

        with left:
            fig = px.pie(
                attrition_distribution,
                names="Attrition",
                values="Employees",
                hole=0.35,
                title="Attrition distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        with right:
            if "Department" in filtered.columns:
                department_attrition = group_attrition(
                    filtered, "Department"
                )

                fig = px.bar(
                    department_attrition,
                    x="Department",
                    y="AttritionRate",
                    text_auto=".1f",
                    title="Attrition rate by department",
                    labels={"AttritionRate": "Attrition rate (%)"}
                )

                st.plotly_chart(fig, use_container_width=True)

        if "OverTime" in filtered.columns:
            overtime_attrition = group_attrition(
                filtered, "OverTime"
            )

            fig = px.bar(
                overtime_attrition,
                x="OverTime",
                y="AttritionRate",
                text_auto=".1f",
                title="Attrition rate by overtime status",
                labels={"AttritionRate": "Attrition rate (%)"}
            )

            st.plotly_chart(fig, use_container_width=True)

        group_options = [
            col for col in [
                "JobRole",
                "BusinessTravel",
                "MaritalStatus",
                "Gender"
            ]
            if col in filtered.columns
        ]

        if group_options:
            group_column = st.selectbox(
                "Explore attrition by",
                group_options
            )

            group_table = group_attrition(
                filtered, group_column
            )

            group_table["AttritionRate"] = (
                group_table["AttritionRate"].round(2)
            )

            st.dataframe(
                group_table.rename(
                    columns={"AttritionRate": "Attrition rate (%)"}
                ),
                use_container_width=True,
                hide_index=True
            )


with risk_tab:
    st.subheader("Burnout and Workload Indicators")

    c1, c2, c3 = st.columns(3)

    wlb = safe_mean(filtered, "WorkLifeBalanceIndex")
    workload = safe_mean(filtered, "WorkloadStressIndicator")

    c1.metric(
        "Mean work-life balance index",
        f"{wlb:.2f}/100" if wlb is not None else "N/A"
    )

    overtime_share = None

    if "OverTime" in filtered.columns:
        overtime_share = (
            filtered["OverTime"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "true", "1"])
            .mean() * 100
        )

    c2.metric(
        "Overtime prevalence",
        f"{overtime_share:.2f}%"
        if overtime_share is not None else "N/A"
    )

    c3.metric(
        "Mean workload indicator",
        f"{workload:.2f}/2" if workload is not None else "N/A"
    )

    risk_counts = (
        filtered["BurnoutRiskCategory"]
        .value_counts()
        .rename_axis("Risk category")
        .reset_index(name="Employees")
    )

    fig = px.bar(
        risk_counts,
        x="Risk category",
        y="Employees",
        title="Distribution of project-defined risk categories",
        text="Employees"
    )

    st.plotly_chart(fig, use_container_width=True)

    if "Department" in filtered.columns:
        department_risk = (
            filtered.groupby("Department", dropna=False)
            .agg(
                Employees=("Department", "size"),
                MeanEngagement=("EngagementIndex", "mean"),
                MeanWorkLifeBalance=("WorkLifeBalanceIndex", "mean"),
                HighRiskShare=(
                    "BurnoutRiskCategory",
                    lambda values: values.eq("High").mean() * 100
                )
            )
            .reset_index()
            .round(2)
        )

        st.write("Risk indicators by department")

        st.dataframe(
            department_risk,
            use_container_width=True,
            hide_index=True
        )

    st.warning(
        "Risk score definition: +1 for overtime and +1 for work-life "
        "balance rating of 2 or below. This is a project-defined screening "
        "proxy, not a validated clinical burnout scale."
    )


with career_tab:
    st.subheader("Career-Stage Analysis")

    if "CareerStage" not in filtered.columns:
        st.info("YearsAtCompany is required for career-stage analysis.")
    else:
        aggregation = {
            "Employees": ("CareerStage", "size"),
            "EngagementIndex": ("EngagementIndex", "mean"),
            "WorkLifeBalanceIndex": ("WorkLifeBalanceIndex", "mean")
        }

        if "Attrition" in filtered.columns:
            aggregation["AttritionRate"] = ("Attrition", "mean")

        career_summary = (
            filtered.groupby("CareerStage", observed=False)
            .agg(**aggregation)
            .reset_index()
        )

        if "AttritionRate" in career_summary.columns:
            career_summary["AttritionRate"] *= 100

        st.dataframe(
            career_summary.round(2),
            use_container_width=True,
            hide_index=True
        )

        if career_summary["EngagementIndex"].notna().any():
            fig = px.bar(
                career_summary,
                x="CareerStage",
                y="EngagementIndex",
                title="Engagement by career stage",
                text_auto=".1f"
            )

            st.plotly_chart(fig, use_container_width=True)

        if "AttritionRate" in career_summary.columns:
            fig = px.line(
                career_summary,
                x="CareerStage",
                y="AttritionRate",
                markers=True,
                title="Attrition rate by career stage",
                labels={"AttritionRate": "Attrition rate (%)"}
            )

            st.plotly_chart(fig, use_container_width=True)


with actions_tab:
    st.subheader("Manager Action Panel")

    st.write(
        "These recommendations are prompts for aggregate organizational "
        "review. They should not be used to label, rank, or penalize individual employees."
    )

    recommendations = []

    overtime_share = None

    if "OverTime" in filtered.columns:
        overtime_share = (
            filtered["OverTime"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "true", "1"])
            .mean()
        )

    if overtime_share is not None and overtime_share >= 0.25:
        recommendations.append((
            "Review overtime and workload",
            "Review capacity, staffing, recurring after-hours work, scheduling, "
            "and workload distribution."
        ))

    if wlb is not None and wlb < 60:
        recommendations.append((
            "Explore work-life balance",
            "Review flexibility, workload predictability, recovery time, "
            "and confidential employee feedback."
        ))

    engagement = safe_mean(filtered, "EngagementIndex")

    if engagement is not None and engagement < 60:
        recommendations.append((
            "Investigate engagement drivers",
            "Explore role-level differences, recognition, job involvement, "
            "manager support, and development opportunities."
        ))

    attrition_rate = safe_mean(filtered, "Attrition")

    if attrition_rate is not None and attrition_rate > 0.15:
        recommendations.append((
            "Review retention signals",
            "Explore exit feedback, career growth, compensation fairness, "
            "and employee support."
        ))

    if not recommendations:
        recommendations.append((
            "Continue monitoring",
            "No configured thresholds were crossed. Continue periodic "
            "aggregate reviews and validate results with qualitative feedback."
        ))

    for title, description in recommendations:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    if "Department" in filtered.columns:
        st.markdown("#### Department-level summary")

        summary = (
            filtered.groupby("Department", dropna=False)
            .agg(
                Employees=("Department", "size"),
                EngagementIndex=("EngagementIndex", "mean"),
                WorkLifeBalanceIndex=("WorkLifeBalanceIndex", "mean"),
                HighRiskShare=(
                    "BurnoutRiskCategory",
                    lambda values: values.eq("High").mean() * 100
                )
            )
            .reset_index()
            .round(2)
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


st.markdown("---")

st.subheader("Export results")

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered dataset",
    data=csv_data,
    file_name="filtered_employee_diagnostics.csv",
    mime="text/csv"
)

st.caption(
    "Dataset provenance limitation: a filename does not establish that records "
    "represent employees of a named company. Verify the original source before "
    "making company-specific claims. All reported associations are descriptive "
    "and do not establish causation."
)
