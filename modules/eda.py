
"""tab1: Numerical Summary Table + Tab 2: Categorical Summary + Tab 3: Visualization Analyze"""

import io
import re
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from shiny import reactive, render, ui


sns.set_theme(style="whitegrid")

# Default order of statistics shown in the summary table
SUMMARY_STATS_ORDER = ["count", "min", "Q1", "median", "mean", "Q3", "max", "sd"]

# Labels for the statistics checkbox group
SUMMARY_STAT_CHOICES = {
    "all": "All",
    "count": "Count",
    "min": "Min",
    "Q1": "Q1",
    "median": "Median",
    "mean": "Mean",
    "Q3": "Q3",
    "max": "Max",
    "sd": "SD",
}

CATEGORICAL_STATS_ORDER = ["count", "percentage"]

PLOT_TYPE_DEFAULT_TITLES = {
    "histogram": "Histogram Plot",
    "density": "Density Plot",
    "boxplot": "Boxplot",
    "bar": "Bar Plot",
    "line": "Line Plot",
}


def eda_ui():
    """
    UI for the EDA tab.
    """
    return ui.div(
        {"class": "main-container"},
        ui.h2("Exploratory Data Analysis", {"class": "section-title"}),
        ui.navset_card_tab(
            ui.nav_panel(
                "Numerical Summary Table",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.div(
                            {"class": "eda-controls"},
                            # 1) Dataset selector
                            ui.input_select(
                                "summary_dataset",
                                "Dataset",
                                {
                                    "raw": "Raw Data",
                                    "cleaned": "Cleaned Data",
                                    "feature_engineered": "Feature Engineered Data",
                                },
                                selected="raw",
                            ),
                            # 2) Numeric variables to summarize
                            ui.div(
                                {"class": "eda-var-group"},
                                ui.tags.label("Numeric Variable", {"class": "form-label"}),
                                ui.output_ui("summary_numeric_vars_ui"),
                            ),
                            # 3) Optional group-by using categorical variables only
                            ui.input_selectize(
                                "summary_group_by",
                                "Group by",
                                choices=[],
                                selected=[],
                                multiple=True,
                                options={"placeholder": "Optional"},
                            ),
                            # 4) Filter variable
                            ui.input_select(
                                "summary_filter_var",
                                "Filter variable",
                                {"": "None"},
                                selected="",
                            ),
                            # Dynamic filter control appears here
                            ui.output_ui("summary_filter_value_ui"),
                            # 5) Statistics to show
                            ui.h5("Statistics to show"),
                            ui.input_checkbox("summary_stats_all", "All", value=True),
                            ui.input_checkbox("summary_stats_count", "Count", value=True),
                            ui.input_checkbox("summary_stats_min", "Min", value=True),
                            ui.input_checkbox("summary_stats_q1", "Q1", value=True),
                            ui.input_checkbox("summary_stats_median", "Median", value=True),
                            ui.input_checkbox("summary_stats_mean", "Mean", value=True),
                            ui.input_checkbox("summary_stats_q3", "Q3", value=True),
                            ui.input_checkbox("summary_stats_max", "Max", value=True),
                            ui.input_checkbox("summary_stats_sd", "SD", value=True),
                            ui.br(),
                            ui.input_action_button(
                                "create_summary_table",
                                "Create Table",
                                class_="btn btn-primary",
                            ),
                            ui.hr(),
                            ui.div(
                                ui.output_text("summary_status"),
                                style="color:#4b5563; font-size:0.95rem;",
                            ),
                        ),
                        title="Controls",
                        width="320px",
                        open="open",
                    ),
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.h5("Numerical Summary Table", class_="mb-0"),
                                ui.download_button(
                                    "download_numeric_summary",
                                    "Export CSV",
                                    class_="btn-primary",
                                ),
                                style=(
                                    "display:flex; justify-content:space-between; "
                                    "align-items:center; gap:12px;"
                                ),
                            )
                        ),
                        ui.output_data_frame("numerical_summary_table"),
                        full_screen=True,
                    ),
                ),
            ),
            ui.nav_panel(
                "Categorical Summary",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.div(
                            {"class": "eda-controls"},
                            ui.input_select(
                                "categorical_summary_dataset",
                                "Dataset",
                                {
                                    "raw": "Raw Data",
                                    "cleaned": "Cleaned Data",
                                    "feature_engineered": "Feature Engineered Data",
                                },
                                selected="raw",
                            ),
                            ui.div(
                                {"class": "eda-var-group"},
                                ui.tags.label("Categorical Variable", {"class": "form-label"}),
                                ui.output_ui("categorical_summary_vars_ui"),
                            ),
                            ui.output_ui("categorical_top_n_ui"),
                            ui.h5("Statistic to display"),
                            ui.input_checkbox("categorical_stats_all", "All", value=True),
                            ui.input_checkbox("categorical_stats_count", "Count", value=True),
                            ui.input_checkbox(
                                "categorical_stats_percentage",
                                "Percentage",
                                value=True,
                            ),
                            ui.br(),
                            ui.input_action_button(
                                "create_categorical_summary",
                                "Create Table",
                                class_="btn btn-primary",
                            ),
                            ui.hr(),
                            ui.div(
                                ui.output_text("categorical_summary_status"),
                                style="color:#4b5563; font-size:0.95rem;",
                            ),
                        ),
                        title="Controls",
                        width="320px",
                        open="open",
                    ),
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.h5("Categorical Summary", class_="mb-0"),
                                ui.download_button(
                                    "download_categorical_summary",
                                    "Export CSV",
                                    class_="btn-primary",
                                ),
                                style=(
                                    "display:flex; justify-content:space-between; "
                                    "align-items:center; gap:12px;"
                                ),
                            )
                        ),
                        ui.output_data_frame("categorical_summary_table"),
                        full_screen=True,
                    ),
                ),
            ),
            ui.nav_panel(
                "Visualization Analyze",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.div(
                            {"class": "eda-controls"},
                            ui.input_select(
                                "visualization_dataset",
                                "Dataset",
                                {
                                    "raw": "Raw Data",
                                    "cleaned": "Cleaned Data",
                                    "feature_engineered": "Feature Engineered Data",
                                },
                                selected="raw",
                            ),
                            ui.input_select(
                                "visualization_plot_type",
                                "Plot Type",
                                {
                                    "": "Please select a plot type",
                                    "histogram": "Histogram",
                                    "density": "Density",
                                    "boxplot": "Boxplot",
                                    "bar": "Bar",
                                    "line": "Line",
                                },
                                selected="",
                            ),
                            ui.output_ui("visualization_variable_controls_ui"),
                            ui.input_select(
                                "visualization_filter_var",
                                "Filter Variable",
                                {"": "None"},
                                selected="",
                            ),
                            ui.output_ui("visualization_filter_value_ui"),
                            ui.input_text(
                                "visualization_plot_title",
                                "Title",
                                value="",
                            ),
                            ui.hr(),
                            ui.div(
                                ui.output_text("visualization_status"),
                                style="color:#4b5563; font-size:0.95rem;",
                            ),
                        ),
                        title="Controls",
                        width="320px",
                        open="open",
                    ),
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.h5("Visualization", class_="mb-0"),
                                ui.download_button(
                                    "download_visualization_plot",
                                    "Export PNG",
                                    class_="btn-primary",
                                ),
                                style=(
                                    "display:flex; justify-content:space-between; "
                                    "align-items:center; gap:12px;"
                                ),
                            )
                        ),
                        ui.output_plot("visualization_plot", height="650px"),
                        full_screen=True,
                    ),
                ),
            ),
        ),
    )


def pick_eda_dataset(choice, raw_df, cleaned_df_value, fe_df):
    """
    Return the dataset selected in the EDA tab.
    """
    if choice == "cleaned":
        return cleaned_df_value
    if choice == "feature_engineered":
        return fe_df
    return raw_df


def is_numeric_series(series):
    """
    Return True if a Series is numeric-like and not boolean.
    """
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def get_numeric_columns(df):
    """
    Return numeric columns from a DataFrame.
    """
    if df is None:
        return []

    return [col for col in df.columns if is_numeric_series(df[col])]


def get_categorical_columns(df):
    """
    Return categorical-like columns from a DataFrame.
    """
    if df is None:
        return []

    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def is_datetime_like_series(series, threshold=0.8, sample_size=200):
    """
    Return True if the series is datetime dtype or looks like datetime strings.
    """
    if series is None:
        return False

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False

    non_missing = series.dropna()
    if non_missing.empty:
        return False

    sample = non_missing.astype(str).head(sample_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")

    return bool(parsed.notna().mean() >= threshold)


def get_datetime_columns(df):
    """
    Return columns that are datetime dtype or strongly datetime-like.
    """
    if df is None:
        return []

    return [col for col in df.columns if is_datetime_like_series(df[col])]


def coerce_datetime_series_if_needed(series):
    """
    Convert a series to datetime if it is already or appears datetime-like.
    """
    if series is None:
        return series

    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    if is_datetime_like_series(series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(series, errors="coerce")

    return series


def get_line_x_choices(df):
    """
    Return line-plot X choices, preferring datetime-like columns and then numeric columns.
    """
    if df is None:
        return {}

    datetime_cols = get_datetime_columns(df)
    numeric_cols = get_numeric_columns(df)

    choices = {}
    for col in datetime_cols:
        choices[col] = f"{col} (date/time)"
    for col in numeric_cols:
        if col not in choices:
            choices[col] = col

    return choices


def default_plot_title(plot_type):
    """
    Default title for each supported plot type.
    """
    return PLOT_TYPE_DEFAULT_TITLES.get(plot_type or "", "")


def safe_input_value(input, input_id, default=None):
    """
    Safely read a possibly dynamic input value.
    """
    try:
        value = input[input_id]()
    except Exception:
        return default

    if value is None:
        return default
    return value


def format_plot_categorical(series):
    """
    Convert a categorical-like series to display-friendly labels, including missing values.
    """
    return series.astype("object").where(~series.isna(), "(Missing)")


def make_safe_filename(text, fallback="visualization_plot"):
    """
    Convert a title into a safe filename stem.
    """
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip()).strip("_")
    return safe or fallback


def make_placeholder_figure(message, title="Visualization"):
    """
    Create a simple placeholder figure with centered text.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def normalize_selected_stats(selected_stats):
    """
    Normalize the selected statistics.
    If 'all' is selected, return every statistic in a fixed order.
    """
    if not selected_stats or "all" in selected_stats:
        return SUMMARY_STATS_ORDER.copy()

    selected_set = set(selected_stats)
    return [stat for stat in SUMMARY_STATS_ORDER if stat in selected_set]


def empty_summary_frame(group_by_cols=None, stats_to_show=None):
    """
    Return an empty summary DataFrame with the correct column names.
    """
    group_by_cols = group_by_cols or []
    stats_to_show = stats_to_show or SUMMARY_STATS_ORDER
    return pd.DataFrame(columns=group_by_cols + ["variable"] + stats_to_show)


def build_numerical_summary(df, numeric_cols, group_by_cols=None, stats_to_show=None):
    """
    Build the numerical summary table.

    Output columns:
    [optional group-by columns] + variable + selected statistics
    """
    visible_stats = normalize_selected_stats(stats_to_show)

    if df is None:
        return empty_summary_frame(group_by_cols, visible_stats)

    group_by_cols = [col for col in (group_by_cols or []) if col in df.columns]

    numeric_cols = [
        col for col in (numeric_cols or [])
        if col in df.columns and is_numeric_series(df[col])
    ]

    if not numeric_cols:
        return empty_summary_frame(group_by_cols, visible_stats)

    # Convert selected numeric columns to long format
    long_df = df[group_by_cols + numeric_cols].melt(
        id_vars=group_by_cols,
        value_vars=numeric_cols,
        var_name="variable",
        value_name="value",
    )

    group_keys = group_by_cols + ["variable"]

    summary = (
        long_df.groupby(group_keys, dropna=False)["value"]
        .agg(
            count="count",
            min="min",
            Q1=lambda s: s.quantile(0.25),
            median="median",
            mean="mean",
            Q3=lambda s: s.quantile(0.75),
            max="max",
            sd="std",
        )
        .reset_index()
    )

    # Round displayed numeric statistics for readability
    numeric_stat_cols = [
        col for col in SUMMARY_STATS_ORDER if col in summary.columns and col != "count"
    ]
    if numeric_stat_cols:
        summary[numeric_stat_cols] = summary[numeric_stat_cols].round(3)

    sort_cols = group_by_cols + ["variable"]
    if sort_cols and not summary.empty:
        summary = summary.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    ordered_cols = group_by_cols + ["variable"] + visible_stats
    return summary.loc[:, ordered_cols]


def normalize_selected_categorical_stats(selected_stats):
    """
    Normalize selected categorical statistics.
    """
    if not selected_stats or "all" in selected_stats:
        return CATEGORICAL_STATS_ORDER.copy()

    selected_set = set(selected_stats)
    return [stat for stat in CATEGORICAL_STATS_ORDER if stat in selected_set]


def empty_categorical_summary_frame(stats_to_show=None):
    """
    Return an empty categorical summary DataFrame with the correct columns.
    """
    stats_to_show = stats_to_show or CATEGORICAL_STATS_ORDER
    return pd.DataFrame(columns=["variable", "category"] + stats_to_show)


def build_categorical_summary(df, categorical_col, top_n=None, stats_to_show=None):
    """
    Build the categorical summary table.

    Output columns:
    variable + category + selected statistics
    """
    visible_stats = normalize_selected_categorical_stats(stats_to_show)

    if df is None or categorical_col is None or categorical_col not in df.columns:
        return empty_categorical_summary_frame(visible_stats)

    series = df[categorical_col]

    if not (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    ):
        return empty_categorical_summary_frame(visible_stats)

    display_series = format_plot_categorical(series)

    summary = (
        display_series.value_counts(dropna=False)
        .rename_axis("category")
        .reset_index(name="count")
    )

    total_rows = len(df)
    if total_rows == 0:
        summary["percentage"] = 0.0
    else:
        summary["percentage"] = (summary["count"] / total_rows * 100).round(2)

    summary["variable"] = categorical_col
    summary["_category_sort"] = summary["category"].astype(str)
    summary = summary.sort_values(
        ["count", "_category_sort"],
        ascending=[False, True],
        kind="stable",
    ).drop(columns="_category_sort")

    if top_n is not None and pd.notna(top_n):
        try:
            top_n = int(top_n)
            if top_n > 0:
                summary = summary.head(min(top_n, len(summary)))
        except (TypeError, ValueError):
            pass

    ordered_cols = ["variable", "category"] + visible_stats
    return summary.loc[:, ordered_cols].reset_index(drop=True)


def build_visualization_figure(
    df,
    plot_type,
    title="",
    x_var=None,
    y_var=None,
    group_by=None,
    bins=30,
    bw_adjust=1.0,
    show_points=False,
):
    """
    Build a matplotlib/seaborn figure for the selected plot type.
    """
    resolved_title = title or default_plot_title(plot_type) or "Visualization"

    if df is None:
        return make_placeholder_figure(
            "Selected dataset is not available yet.",
            resolved_title,
        )

    if not plot_type:
        return make_placeholder_figure("Please select a plot type.", resolved_title)

    try:
        if plot_type == "histogram":
            if not x_var or x_var not in df.columns or not is_numeric_series(df[x_var]):
                return make_placeholder_figure(
                    "Select one numeric X variable for the histogram.",
                    resolved_title,
                )

            cols = [x_var] + ([group_by] if group_by and group_by in df.columns else [])
            plot_df = df.loc[:, cols].copy()
            plot_df = plot_df.dropna(subset=[x_var])

            if group_by and group_by in plot_df.columns:
                plot_df[group_by] = format_plot_categorical(plot_df[group_by])

            if plot_df.empty:
                return make_placeholder_figure(
                    "No rows are available to plot after removing missing values.",
                    resolved_title,
                )

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(
                data=plot_df,
                x=x_var,
                hue=group_by if group_by and group_by in plot_df.columns else None,
                bins=max(int(bins or 30), 1),
                multiple="layer",
                ax=ax,
            )
            ax.set_xlabel(x_var)
            ax.set_ylabel("Count")

        elif plot_type == "density":
            if not x_var or x_var not in df.columns or not is_numeric_series(df[x_var]):
                return make_placeholder_figure(
                    "Select one numeric X variable for the density plot.",
                    resolved_title,
                )

            cols = [x_var] + ([group_by] if group_by and group_by in df.columns else [])
            plot_df = df.loc[:, cols].copy()
            plot_df = plot_df.dropna(subset=[x_var])

            if group_by and group_by in plot_df.columns:
                plot_df[group_by] = format_plot_categorical(plot_df[group_by])

            if plot_df.empty:
                return make_placeholder_figure(
                    "No rows are available to plot after removing missing values.",
                    resolved_title,
                )

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(
                data=plot_df,
                x=x_var,
                hue=group_by if group_by and group_by in plot_df.columns else None,
                bw_adjust=max(float(bw_adjust or 1.0), 0.1),
                common_norm=False if group_by else True,
                fill=False,
                ax=ax,
            )
            ax.set_xlabel(x_var)
            ax.set_ylabel("Density")

        elif plot_type == "boxplot":
            if not y_var or y_var not in df.columns or not is_numeric_series(df[y_var]):
                return make_placeholder_figure(
                    "Select one numeric Y variable for the boxplot.",
                    resolved_title,
                )

            cols = [y_var] + ([x_var] if x_var and x_var in df.columns else [])
            plot_df = df.loc[:, cols].copy()

            if x_var and x_var in plot_df.columns:
                plot_df[x_var] = format_plot_categorical(plot_df[x_var])

            plot_df = plot_df.dropna(subset=[y_var])

            if plot_df.empty:
                return make_placeholder_figure(
                    "No rows are available to plot after removing missing values.",
                    resolved_title,
                )

            if x_var and x_var in plot_df.columns:
                n_levels = int(plot_df[x_var].nunique(dropna=False))
                fig_width = min(max(8, 1.1 * n_levels), 18)
                fig, ax = plt.subplots(figsize=(fig_width, 6))

                sns.boxplot(data=plot_df, x=x_var, y=y_var, ax=ax)

                if show_points:
                    sns.stripplot(
                        data=plot_df,
                        x=x_var,
                        y=y_var,
                        ax=ax,
                        color="black",
                        alpha=0.45,
                        size=3,
                    )

                if n_levels > 5:
                    ax.tick_params(axis="x", rotation=45)
                ax.set_xlabel(x_var)
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(data=plot_df, y=y_var, ax=ax)

                if show_points:
                    sns.stripplot(
                        data=plot_df,
                        y=y_var,
                        ax=ax,
                        color="black",
                        alpha=0.45,
                        size=3,
                    )
                ax.set_xlabel("")

            ax.set_ylabel(y_var)

        elif plot_type == "bar":
            if not x_var or x_var not in df.columns or x_var not in get_categorical_columns(df):
                return make_placeholder_figure(
                    "Select one categorical X variable for the bar plot.",
                    resolved_title,
                )

            plot_df = df.loc[:, [x_var]].copy()
            plot_df[x_var] = format_plot_categorical(plot_df[x_var])

            if plot_df.empty:
                return make_placeholder_figure(
                    "No rows are available to plot.",
                    resolved_title,
                )

            order = plot_df[x_var].value_counts(dropna=False).index.tolist()
            fig_width = min(max(8, 0.75 * len(order) + 4), 18)
            fig, ax = plt.subplots(figsize=(fig_width, 6))

            sns.countplot(
                data=plot_df,
                x=x_var,
                order=order,
                ax=ax,
            )

            if len(order) > 5:
                ax.tick_params(axis="x", rotation=45)

            ax.set_xlabel(x_var)
            ax.set_ylabel("Count")

        elif plot_type == "line":
            if not x_var or x_var not in df.columns:
                return make_placeholder_figure(
                    "Select one X variable for the line plot.",
                    resolved_title,
                )

            if not y_var or y_var not in df.columns or not is_numeric_series(df[y_var]):
                return make_placeholder_figure(
                    "Select one numeric Y variable for the line plot.",
                    resolved_title,
                )

            line_x_allowed = set(get_line_x_choices(df).keys())
            if x_var not in line_x_allowed:
                return make_placeholder_figure(
                    "For line plots, choose a date/time or numeric X variable.",
                    resolved_title,
                )

            cols = [x_var, y_var] + ([group_by] if group_by and group_by in df.columns else [])
            plot_df = df.loc[:, cols].copy()
            plot_df[x_var] = coerce_datetime_series_if_needed(plot_df[x_var])

            if group_by and group_by in plot_df.columns:
                plot_df[group_by] = format_plot_categorical(plot_df[group_by])

            plot_df = plot_df.dropna(subset=[x_var, y_var])

            if plot_df.empty:
                return make_placeholder_figure(
                    "No rows are available to plot after removing missing values.",
                    resolved_title,
                )

            sort_cols = [group_by, x_var] if group_by and group_by in plot_df.columns else [x_var]
            plot_df = plot_df.sort_values(sort_cols, kind="stable")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(
                data=plot_df,
                x=x_var,
                y=y_var,
                hue=group_by if group_by and group_by in plot_df.columns else None,
                errorbar=None,
                ax=ax,
            )

            if pd.api.types.is_datetime64_any_dtype(plot_df[x_var]):
                fig.autofmt_xdate()

            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)

        else:
            return make_placeholder_figure("Unsupported plot type.", resolved_title)

        ax.set_title(resolved_title)
        fig.tight_layout()
        return fig

    except Exception as exc:
        return make_placeholder_figure(
            f"Unable to render plot: {exc}",
            resolved_title,
        )


def eda_server(input, output, session, raw_data, cleaned_data, fe_data):
    """
    Server logic for the EDA tab.

    Parameters
    ----------
    raw_data : reactive calc
        Returns raw dataset
    cleaned_data : reactive calc
        Returns cleaned dataset
    fe_data : reactive calc
        Returns feature-engineered dataset
    """
    selected_categorical_token = reactive.Value(None)

    @reactive.calc
    def selected_eda_df():
        """
        Return the dataset currently selected by the user.
        """
        return pick_eda_dataset(
            input.summary_dataset(),
            raw_data(),
            cleaned_data(),
            fe_data(),
        )

    @reactive.calc
    def selected_categorical_eda_df():
        """
        Return the dataset currently selected in the categorical summary tab.
        """
        return pick_eda_dataset(
            input.categorical_summary_dataset(),
            raw_data(),
            cleaned_data(),
            fe_data(),
        )

    @reactive.calc
    def selected_visualization_df():
        """
        Return the dataset currently selected in the visualization tab.
        """
        return pick_eda_dataset(
            input.visualization_dataset(),
            raw_data(),
            cleaned_data(),
            fe_data(),
        )

    @output
    @render.ui
    def summary_numeric_vars_ui():
        df = selected_eda_df()
        numeric_cols = get_numeric_columns(df)

        if not numeric_cols:
            return ui.p("No numeric variables available.", class_="text-muted")

        return ui.div(
            {"class": "eda-var-checkboxes"},
            *[
                ui.div(
                    {"class": "eda-var-option"},
                    ui.input_checkbox(f"summary_numvar_{col}", col, value=True),
                )
                for col in numeric_cols
            ],
        )

    @reactive.effect
    def _update_numerical_summary_inputs():
        """
        Update sidebar controls whenever the selected dataset changes.
        """
        df = selected_eda_df()
        categorical_cols = get_categorical_columns(df)
        all_cols = df.columns.tolist() if df is not None else []

        ui.update_selectize(
            "summary_group_by",
            choices={col: col for col in categorical_cols},
            selected=[],
            server=True,
        )

        filter_choices = {"": "None"}
        filter_choices.update({col: col for col in all_cols})
        ui.update_select(
            "summary_filter_var",
            choices=filter_choices,
            selected="",
        )

    stats_ids = [
        "summary_stats_count",
        "summary_stats_min",
        "summary_stats_q1",
        "summary_stats_median",
        "summary_stats_mean",
        "summary_stats_q3",
        "summary_stats_max",
        "summary_stats_sd",
    ]

    @reactive.effect
    @reactive.event(input.summary_stats_all)
    def _handle_all_checkbox():
        if input.summary_stats_all():
            for stat_id in stats_ids:
                ui.update_checkbox(stat_id, value=True, session=session)

    @reactive.effect
    def _sync_all_checkbox():
        values = [
            input.summary_stats_count(),
            input.summary_stats_min(),
            input.summary_stats_q1(),
            input.summary_stats_median(),
            input.summary_stats_mean(),
            input.summary_stats_q3(),
            input.summary_stats_max(),
            input.summary_stats_sd(),
        ]
        should_all_be_checked = all(values)
        if input.summary_stats_all() != should_all_be_checked:
            ui.update_checkbox(
                "summary_stats_all",
                value=should_all_be_checked,
                session=session,
            )

    @reactive.calc
    def categorical_filter_choice_map():
        """
        Build a token -> original value map for numerical-summary filtering.
        """
        df = selected_eda_df()
        filter_col = input.summary_filter_var()

        if df is None or not filter_col or filter_col not in df.columns:
            return {}

        series = df[filter_col]

        if is_numeric_series(series):
            return {}

        unique_values = pd.Series(pd.unique(series.dropna())).tolist()
        unique_values = sorted(unique_values, key=lambda x: str(x))
        return {f"choice_{i}": value for i, value in enumerate(unique_values)}

    @output
    @render.ui
    def summary_filter_value_ui():
        """
        Show the correct filter widget depending on the selected filter variable.
        """
        df = selected_eda_df()
        filter_col = input.summary_filter_var()

        if df is None or not filter_col or filter_col not in df.columns:
            return ui.div()

        series = df[filter_col]

        if is_numeric_series(series):
            non_missing = series.dropna()

            if non_missing.empty:
                return ui.p(
                    f'"{filter_col}" only contains missing values.',
                    class_="text-muted",
                )

            min_val = non_missing.min()
            max_val = non_missing.max()

            if pd.isna(min_val) or pd.isna(max_val):
                return ui.div()

            if min_val == max_val:
                return ui.p(
                    f'All non-missing values of "{filter_col}" are {min_val}.',
                    class_="text-muted",
                )

            is_integer = pd.api.types.is_integer_dtype(non_missing)

            if is_integer:
                min_val = int(min_val)
                max_val = int(max_val)
                slider_step = 1
            else:
                min_val = float(min_val)
                max_val = float(max_val)
                slider_step = None

            return ui.input_slider(
                "summary_numeric_filter",
                f"Filter range: {filter_col}",
                min=min_val,
                max=max_val,
                value=(min_val, max_val),
                step=slider_step,
            )

        choice_map = categorical_filter_choice_map()
        choices = {token: str(value) for token, value in choice_map.items()}

        if series.isna().any():
            choices["__missing__"] = "(Missing)"

        return ui.input_selectize(
            "summary_categorical_filter",
            f"Values: {filter_col}",
            choices=choices,
            selected=list(choices.keys()),
            multiple=True,
            options={"placeholder": "Select one or more values"},
        )

    @reactive.calc
    def selected_summary_numeric_vars():
        df = selected_eda_df()
        numeric_cols = get_numeric_columns(df)

        selected = []
        for col in numeric_cols:
            try:
                if input[f"summary_numvar_{col}"]():
                    selected.append(col)
            except Exception:
                pass
        return selected

    @reactive.calc
    def filtered_eda_df():
        """
        Apply the optional filter chosen in the numerical summary sidebar.
        """
        df = selected_eda_df()
        if df is None:
            return None

        filter_col = input.summary_filter_var()
        if not filter_col or filter_col not in df.columns:
            return df

        series = df[filter_col]

        if is_numeric_series(series):
            filter_range = input.summary_numeric_filter()
            if filter_range is None:
                return df

            low, high = filter_range
            return df[series.between(low, high, inclusive="both")]

        choice_map = categorical_filter_choice_map()
        selected_tokens = input.summary_categorical_filter()

        if selected_tokens is None:
            return df

        selected_tokens = list(selected_tokens)

        if len(selected_tokens) == 0:
            return df.iloc[0:0].copy()

        selected_values = [
            choice_map[token]
            for token in selected_tokens
            if token in choice_map
        ]

        mask = series.isin(selected_values)

        if "__missing__" in selected_tokens:
            mask = mask | series.isna()

        return df[mask]

    @reactive.calc
    def selected_summary_stats():
        """
        Collect selected statistics from the individual checkboxes.
        """
        stats = []

        if input.summary_stats_count():
            stats.append("count")
        if input.summary_stats_min():
            stats.append("min")
        if input.summary_stats_q1():
            stats.append("Q1")
        if input.summary_stats_median():
            stats.append("median")
        if input.summary_stats_mean():
            stats.append("mean")
        if input.summary_stats_q3():
            stats.append("Q3")
        if input.summary_stats_max():
            stats.append("max")
        if input.summary_stats_sd():
            stats.append("sd")

        return stats

    @reactive.calc
    @reactive.event(input.create_summary_table)
    def numerical_summary_df():
        """
        Build the final summary table displayed in the main panel.
        Only updates after the user clicks 'Create Table'.
        """
        df = filtered_eda_df()
        if df is None:
            return empty_summary_frame()

        numeric_cols = selected_summary_numeric_vars()
        group_by_cols = list(input.summary_group_by() or [])
        stats_to_show = selected_summary_stats()

        return build_numerical_summary(
            df=df,
            numeric_cols=numeric_cols,
            group_by_cols=group_by_cols,
            stats_to_show=stats_to_show,
        )

    @output
    @render.text
    def summary_status():
        """
        Small status text shown at the bottom of the numerical summary sidebar.
        """
        selected_name = {
            "raw": "Raw Data",
            "cleaned": "Cleaned Data",
            "feature_engineered": "Feature Engineered Data",
        }.get(input.summary_dataset() or "raw", "Raw Data")

        df = selected_eda_df()
        filtered_df = filtered_eda_df()

        if df is None:
            return f"{selected_name} is not available yet."

        return (
            f"Dataset: {selected_name} | "
            f"Rows before filter: {df.shape[0]:,} | "
            f"Rows after filter: {filtered_df.shape[0]:,}"
        )

    @output
    @render.data_frame
    def numerical_summary_table():
        """
        Render the interactive summary table.
        """
        return render.DataGrid(
            numerical_summary_df(),
            width="100%",
            height="600px",
            summary=True,
            filters=False,
        )

    @render.download(
        filename="numerical_summary.csv",
        media_type="text/csv",
    )
    def download_numeric_summary():
        """
        Export the current summary table as CSV.
        """
        current_view = numerical_summary_table.data_view()
        if current_view is None:
            current_view = numerical_summary_df()

        yield current_view.to_csv(index=False)

    # -----------------------------
    # Tab 2: Categorical Summary
    # -----------------------------
    @reactive.calc
    def categorical_summary_var_choice_map():
        """
        Build a token -> categorical column name map for safe checkbox IDs.
        """
        df = selected_categorical_eda_df()
        categorical_cols = get_categorical_columns(df)
        return {f"catvar_{i}": col for i, col in enumerate(categorical_cols)}

    @output
    @render.ui
    def categorical_summary_vars_ui():
        choice_map = categorical_summary_var_choice_map()

        if not choice_map:
            return ui.p("No categorical variables available.", class_="text-muted")

        selected_token = selected_categorical_token.get()
        if selected_token not in choice_map:
            selected_token = next(iter(choice_map))

        return ui.div(
            {"class": "eda-var-checkboxes"},
            *[
                ui.div(
                    {"class": "eda-var-option"},
                    ui.input_checkbox(
                        f"categorical_summary_var_{token}",
                        label,
                        value=(token == selected_token),
                    ),
                )
                for token, label in choice_map.items()
            ],
        )

    @reactive.effect
    def _enforce_single_categorical_summary_var():
        """
        Allow only one categorical variable checkbox to remain selected.
        """
        choice_map = categorical_summary_var_choice_map()
        tokens = list(choice_map.keys())

        if not tokens:
            selected_categorical_token.set(None)
            return

        current_values = {}
        for token in tokens:
            input_id = f"categorical_summary_var_{token}"
            try:
                current_values[token] = bool(input[input_id]())
            except Exception:
                current_values[token] = False

        previous_token = selected_categorical_token.get()
        selected_tokens = [token for token, is_checked in current_values.items() if is_checked]

        if not selected_tokens:
            keep_token = previous_token if previous_token in tokens else tokens[0]
            ui.update_checkbox(
                f"categorical_summary_var_{keep_token}",
                value=True,
                session=session,
            )
            selected_categorical_token.set(keep_token)
            return

        if len(selected_tokens) == 1:
            selected_categorical_token.set(selected_tokens[0])
            return

        keep_token = next(
            (token for token in selected_tokens if token != previous_token),
            selected_tokens[0],
        )

        for token in tokens:
            ui.update_checkbox(
                f"categorical_summary_var_{token}",
                value=(token == keep_token),
                session=session,
            )

        selected_categorical_token.set(keep_token)

    @reactive.calc
    def selected_categorical_summary_var():
        """
        Return the selected categorical column name.
        """
        choice_map = categorical_summary_var_choice_map()
        selected_token = selected_categorical_token.get()
        return choice_map.get(selected_token)

    @reactive.calc
    def categorical_unique_count():
        """
        Number of unique categories in the selected categorical variable.
        Missing values count as one category if present.
        """
        df = selected_categorical_eda_df()
        selected_col = selected_categorical_summary_var()

        if df is None or selected_col is None or selected_col not in df.columns:
            return 0

        return int(df[selected_col].nunique(dropna=False))

    @output
    @render.ui
    def categorical_top_n_ui():
        """
        Numeric input for top N categories. Defaults to all categories.
        """
        default_n = categorical_unique_count()
        if default_n <= 0:
            default_n = 1

        return ui.input_numeric(
            "categorical_top_n",
            "Show top N",
            value=default_n,
            min=1,
            step=1,
        )

    categorical_stats_ids = [
        "categorical_stats_count",
        "categorical_stats_percentage",
    ]

    @reactive.effect
    @reactive.event(input.categorical_stats_all)
    def _handle_categorical_stats_all():
        if input.categorical_stats_all():
            for stat_id in categorical_stats_ids:
                ui.update_checkbox(stat_id, value=True, session=session)

    @reactive.effect
    def _sync_categorical_stats_all():
        values = [
            input.categorical_stats_count(),
            input.categorical_stats_percentage(),
        ]
        should_all_be_checked = all(values)

        if input.categorical_stats_all() != should_all_be_checked:
            ui.update_checkbox(
                "categorical_stats_all",
                value=should_all_be_checked,
                session=session,
            )

    @reactive.calc
    def selected_categorical_stats():
        """
        Collect selected categorical statistics from the individual checkboxes.
        """
        stats = []

        if input.categorical_stats_count():
            stats.append("count")
        if input.categorical_stats_percentage():
            stats.append("percentage")

        return stats

    @reactive.calc
    @reactive.event(input.create_categorical_summary)
    def categorical_summary_df():
        """
        Build the categorical summary table.
        Only updates after the user clicks 'Create Table'.
        """
        df = selected_categorical_eda_df()
        selected_col = selected_categorical_summary_var()
        top_n = input.categorical_top_n()
        stats_to_show = selected_categorical_stats()

        return build_categorical_summary(
            df=df,
            categorical_col=selected_col,
            top_n=top_n,
            stats_to_show=stats_to_show,
        )

    @output
    @render.text
    def categorical_summary_status():
        """
        Status text for the categorical summary sidebar.
        """
        selected_name = {
            "raw": "Raw Data",
            "cleaned": "Cleaned Data",
            "feature_engineered": "Feature Engineered Data",
        }.get(input.categorical_summary_dataset() or "raw", "Raw Data")

        df = selected_categorical_eda_df()
        selected_col = selected_categorical_summary_var()
        n_categories = categorical_unique_count()

        if df is None:
            return f"{selected_name} is not available yet."

        if selected_col is None:
            return f"Dataset: {selected_name} | No categorical variable selected."

        return (
            f"Dataset: {selected_name} | "
            f"Variable: {selected_col} | "
            f"Categories: {n_categories:,} | "
            f"Rows: {df.shape[0]:,}"
        )

    @output
    @render.data_frame
    def categorical_summary_table():
        """
        Render the categorical summary table.
        """
        return render.DataGrid(
            categorical_summary_df(),
            width="100%",
            height="600px",
            summary=True,
            filters=False,
        )

    @render.download(
        filename="categorical_summary.csv",
        media_type="text/csv",
    )
    def download_categorical_summary():
        """
        Export the current categorical summary table as CSV.
        """
        current_view = categorical_summary_table.data_view()
        if current_view is None:
            current_view = categorical_summary_df()

        yield current_view.to_csv(index=False)

    # -----------------------------
    # Tab 3: Visualization Analyze
    # -----------------------------
    @reactive.effect
    def _update_visualization_filter_var_choices():
        """
        Update visualization filter choices whenever the selected dataset changes.
        """
        df = selected_visualization_df()
        all_cols = df.columns.tolist() if df is not None else []

        filter_choices = {"": "None"}
        filter_choices.update({col: col for col in all_cols})

        ui.update_select(
            "visualization_filter_var",
            choices=filter_choices,
            selected="",
        )

    @reactive.effect
    @reactive.event(input.visualization_plot_type)
    def _update_visualization_default_title():
        """
        Set the plot title to the default for the selected plot type.
        """
        ui.update_text(
            "visualization_plot_title",
            value=default_plot_title(input.visualization_plot_type()),
            session=session,
        )

    @output
    @render.ui
    def visualization_variable_controls_ui():
        """
        Show variable controls that depend on the selected plot type.
        """
        df = selected_visualization_df()
        plot_type = input.visualization_plot_type()

        if df is None:
            return ui.p("Selected dataset is not available yet.", class_="text-muted")

        if not plot_type:
            return ui.p("Select a plot type to choose variables.", class_="text-muted")

        numeric_cols = get_numeric_columns(df)
        categorical_cols = get_categorical_columns(df)
        line_x_choices = get_line_x_choices(df)

        if plot_type == "histogram":
            if not numeric_cols:
                return ui.p("No numeric variables available for histogram.", class_="text-muted")

            return ui.TagList(
                ui.input_select(
                    "visualization_hist_x",
                    "X Variable",
                    {col: col for col in numeric_cols},
                    selected=numeric_cols[0],
                ),
                ui.input_select(
                    "visualization_hist_group_by",
                    "Group by",
                    {"": "None", **{col: col for col in categorical_cols}},
                    selected="",
                ),
                ui.input_numeric(
                    "visualization_hist_bins",
                    "Number of bins in histogram",
                    value=30,
                    min=1,
                    step=1,
                ),
            )

        if plot_type == "density":
            if not numeric_cols:
                return ui.p("No numeric variables available for density plot.", class_="text-muted")

            return ui.TagList(
                ui.input_select(
                    "visualization_density_x",
                    "X Variable",
                    {col: col for col in numeric_cols},
                    selected=numeric_cols[0],
                ),
                ui.input_select(
                    "visualization_density_group_by",
                    "Group by",
                    {"": "None", **{col: col for col in categorical_cols}},
                    selected="",
                ),
                ui.input_slider(
                    "visualization_density_bw_adjust",
                    "Bandwidth adjustment",
                    min=0.1,
                    max=3.0,
                    value=1.0,
                    step=0.1,
                ),
            )

        if plot_type == "boxplot":
            if not numeric_cols:
                return ui.p("No numeric variables available for boxplot.", class_="text-muted")

            return ui.TagList(
                ui.input_select(
                    "visualization_box_y",
                    "Y Variable",
                    {col: col for col in numeric_cols},
                    selected=numeric_cols[0],
                ),
                ui.input_select(
                    "visualization_box_x",
                    "X Variable",
                    {"": "None", **{col: col for col in categorical_cols}},
                    selected="",
                ),
                ui.input_checkbox(
                    "visualization_box_show_points",
                    "Show points",
                    value=False,
                ),
            )

        if plot_type == "bar":
            if not categorical_cols:
                return ui.p("No categorical variables available for bar plot.", class_="text-muted")

            return ui.TagList(
                ui.input_select(
                    "visualization_bar_x",
                    "X Variable",
                    {col: col for col in categorical_cols},
                    selected=categorical_cols[0],
                ),
            )

        if plot_type == "line":
            if not line_x_choices:
                return ui.p(
                    "No date/time or numeric variables available for the line-plot X axis.",
                    class_="text-muted",
                )

            if not numeric_cols:
                return ui.p("No numeric variables available for the line-plot Y axis.", class_="text-muted")

            return ui.TagList(
                ui.input_select(
                    "visualization_line_x",
                    "X Variable",
                    line_x_choices,
                    selected=next(iter(line_x_choices)),
                ),
                ui.input_select(
                    "visualization_line_y",
                    "Y Variable",
                    {col: col for col in numeric_cols},
                    selected=numeric_cols[0],
                ),
                ui.input_select(
                    "visualization_line_group_by",
                    "Group by",
                    {"": "None", **{col: col for col in categorical_cols}},
                    selected="",
                ),
            )

        return ui.div()

    @reactive.calc
    def visualization_filter_choice_map():
        """
        Build a token -> original value map for visualization filtering.
        """
        df = selected_visualization_df()
        filter_col = input.visualization_filter_var()

        if df is None or not filter_col or filter_col not in df.columns:
            return {}

        series = df[filter_col]

        if is_numeric_series(series):
            return {}

        unique_values = pd.Series(pd.unique(series.dropna())).tolist()
        unique_values = sorted(unique_values, key=lambda x: str(x))
        return {f"choice_{i}": value for i, value in enumerate(unique_values)}

    @output
    @render.ui
    def visualization_filter_value_ui():
        """
        Show the correct visualization filter widget based on the selected filter variable.
        """
        df = selected_visualization_df()
        filter_col = input.visualization_filter_var()

        if df is None or not filter_col or filter_col not in df.columns:
            return ui.div()

        series = df[filter_col]

        if is_numeric_series(series):
            non_missing = series.dropna()

            if non_missing.empty:
                return ui.p(
                    f'"{filter_col}" only contains missing values.',
                    class_="text-muted",
                )

            min_val = non_missing.min()
            max_val = non_missing.max()

            if pd.isna(min_val) or pd.isna(max_val):
                return ui.div()

            if min_val == max_val:
                return ui.p(
                    f'All non-missing values of "{filter_col}" are {min_val}.',
                    class_="text-muted",
                )

            is_integer = pd.api.types.is_integer_dtype(non_missing)

            if is_integer:
                min_val = int(min_val)
                max_val = int(max_val)
                slider_step = 1
            else:
                min_val = float(min_val)
                max_val = float(max_val)
                slider_step = None

            return ui.input_slider(
                "visualization_numeric_filter",
                f"Filter range: {filter_col}",
                min=min_val,
                max=max_val,
                value=(min_val, max_val),
                step=slider_step,
            )

        choice_map = visualization_filter_choice_map()
        choices = {token: str(value) for token, value in choice_map.items()}

        if series.isna().any():
            choices["__missing__"] = "(Missing)"

        return ui.input_selectize(
            "visualization_categorical_filter",
            f"Values: {filter_col}",
            choices=choices,
            selected=list(choices.keys()),
            multiple=True,
            options={"placeholder": "Select one or more values"},
        )

    @reactive.calc
    def filtered_visualization_df():
        """
        Apply the optional visualization filter before drawing the plot.
        """
        df = selected_visualization_df()
        if df is None:
            return None

        filter_col = input.visualization_filter_var()
        if not filter_col or filter_col not in df.columns:
            return df

        series = df[filter_col]

        if is_numeric_series(series):
            filter_range = safe_input_value(input, "visualization_numeric_filter")
            if filter_range is None:
                return df

            low, high = filter_range
            return df[series.between(low, high, inclusive="both")]

        choice_map = visualization_filter_choice_map()
        selected_tokens = safe_input_value(input, "visualization_categorical_filter")

        if selected_tokens is None:
            return df

        selected_tokens = list(selected_tokens)

        if len(selected_tokens) == 0:
            return df.iloc[0:0].copy()

        selected_values = [
            choice_map[token]
            for token in selected_tokens
            if token in choice_map
        ]

        mask = series.isin(selected_values)

        if "__missing__" in selected_tokens:
            mask = mask | series.isna()

        return df[mask]

    @reactive.calc
    def current_visualization_title():
        """
        Title to use for the current plot.
        """
        title = (input.visualization_plot_title() or "").strip()
        if title:
            return title
        return default_plot_title(input.visualization_plot_type())

    @reactive.calc
    def visualization_plot_spec():
        """
        Collect the current plot configuration.
        """
        plot_type = input.visualization_plot_type()
        df = filtered_visualization_df()

        spec = {
            "df": df,
            "plot_type": plot_type,
            "title": current_visualization_title(),
            "x_var": None,
            "y_var": None,
            "group_by": None,
            "bins": 30,
            "bw_adjust": 1.0,
            "show_points": False,
        }

        if plot_type == "histogram":
            spec["x_var"] = safe_input_value(input, "visualization_hist_x")
            spec["group_by"] = safe_input_value(input, "visualization_hist_group_by", "")
            spec["bins"] = safe_input_value(input, "visualization_hist_bins", 30)

        elif plot_type == "density":
            spec["x_var"] = safe_input_value(input, "visualization_density_x")
            spec["group_by"] = safe_input_value(input, "visualization_density_group_by", "")
            spec["bw_adjust"] = safe_input_value(input, "visualization_density_bw_adjust", 1.0)

        elif plot_type == "boxplot":
            spec["x_var"] = safe_input_value(input, "visualization_box_x", "")
            spec["y_var"] = safe_input_value(input, "visualization_box_y")
            spec["show_points"] = bool(
                safe_input_value(input, "visualization_box_show_points", False)
            )

        elif plot_type == "bar":
            spec["x_var"] = safe_input_value(input, "visualization_bar_x")

        elif plot_type == "line":
            spec["x_var"] = safe_input_value(input, "visualization_line_x")
            spec["y_var"] = safe_input_value(input, "visualization_line_y")
            spec["group_by"] = safe_input_value(input, "visualization_line_group_by", "")

        return spec

    @output
    @render.text
    def visualization_status():
        """
        Status text shown at the bottom of the visualization sidebar.
        """
        selected_name = {
            "raw": "Raw Data",
            "cleaned": "Cleaned Data",
            "feature_engineered": "Feature Engineered Data",
        }.get(input.visualization_dataset() or "raw", "Raw Data")

        df = selected_visualization_df()
        filtered_df = filtered_visualization_df()
        plot_type = input.visualization_plot_type()

        if df is None:
            return f"{selected_name} is not available yet."

        plot_label = {
            "": "None",
            "histogram": "Histogram",
            "density": "Density",
            "boxplot": "Boxplot",
            "bar": "Bar",
            "line": "Line",
        }.get(plot_type or "", "None")

        return (
            f"Dataset: {selected_name} | "
            f"Plot type: {plot_label} | "
            f"Rows before filter: {df.shape[0]:,} | "
            f"Rows after filter: {filtered_df.shape[0]:,}"
        )

    @output
    @render.plot
    def visualization_plot():
        """
        Render the current visualization.
        """
        spec = visualization_plot_spec()
        return build_visualization_figure(**spec)

    @render.download(
        filename=lambda: f"{make_safe_filename(current_visualization_title())}.png",
        media_type="image/png",
    )
    def download_visualization_plot():
        """
        Export the current visualization as a PNG image.
        """
        fig = build_visualization_figure(**visualization_plot_spec())
        try:
            with io.BytesIO() as buf:
                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                yield buf.getvalue()
        finally:
            plt.close(fig)
