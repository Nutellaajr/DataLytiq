from __future__ import annotations

from shiny import ui
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def _make_unique(names: list[str]) -> list[str]:
    """
    Make column names unique after cleaning.

    Example:
    ['age', 'age', 'age'] -> ['age', 'age_2', 'age_3']
    """
    seen = {}
    unique_names = []

    for name in names:
        if name not in seen:
            seen[name] = 1
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")

    return unique_names


def _standardize_column_names(columns) -> list[str]:
    """
    Standardize column names so they are easier to work with later.
    """
    cleaned = (
        pd.Index(columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
        .tolist()
    )
    return _make_unique(cleaned)


def _trim_string_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading and trailing spaces from text values
    without converting missing values into strings.
    """
    out = df.copy()
    text_cols = out.select_dtypes(include=["object", "string"]).columns

    for col in text_cols:
        out[col] = out[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    return out


def _missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact column-level missing value summary.
    """
    total_missing = df.isna().sum()
    pct_missing = (total_missing / len(df) * 100).round(2) if len(df) > 0 else 0

    summary = pd.DataFrame(
        {
            "column": total_missing.index,
            "missing_count": total_missing.values,
            "missing_pct": pct_missing.values if hasattr(pct_missing, "values") else pct_missing,
            "dtype": df.dtypes.astype(str).values,
        }
    )

    return summary.sort_values(
        by=["missing_count", "column"],
        ascending=[False, True]
    ).reset_index(drop=True)


def cleaning_ui():
    """
    Build the UI for the data cleaning and preprocessing section.

    The controls here let users apply common preprocessing steps
    interactively and inspect the updated dataset right away.
    """
    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Cleaning Controls"),
                ui.p("Choose the preprocessing steps you want to apply."),

                ui.accordion(
                    ui.accordion_panel(
                        "1) Standardization",
                        ui.input_checkbox(
                            "standardize_names",
                            "Standardize column names",
                            value=True,
                        ),
                        ui.input_checkbox(
                            "trim_strings",
                            "Trim whitespace in text columns",
                            value=True,
                        ),
                    ),
                    ui.accordion_panel(
                        "2) Duplicates",
                        ui.input_checkbox(
                            "remove_duplicates",
                            "Remove duplicate rows",
                            value=False,
                        ),
                    ),
                    ui.accordion_panel(
                        "3) Missing Values",
                        ui.input_select(
                            "missing_strategy",
                            "Missing value strategy",
                            {
                                "none": "Do nothing",
                                "drop_rows": "Drop rows with missing values",
                                "drop_cols": "Drop columns with missing values",
                                "mean_mode": "Impute numeric with mean, categorical with mode",
                                "median_mode": "Impute numeric with median, categorical with mode",
                            },
                            selected="none",
                        ),
                    ),
                    ui.accordion_panel(
                        "4) Scaling",
                        ui.input_select(
                            "scaling_method",
                            "Scaling method",
                            {
                                "none": "None",
                                "standard": "Standard scaling (z-score)",
                                "minmax": "Min-max scaling",
                            },
                            selected="none",
                        ),
                        ui.input_selectize(
                            "scale_cols",
                            "Numeric columns to scale",
                            choices=[],
                            multiple=True,
                        ),
                    ),
                    ui.accordion_panel(
                        "5) Encoding",
                        ui.input_checkbox(
                            "encode_categorical",
                            "One-hot encode categorical columns",
                            value=False,
                        ),
                    ),
                    ui.accordion_panel(
                        "6) Outliers",
                        ui.input_checkbox(
                            "handle_outliers",
                            "Handle outliers with IQR",
                            value=False,
                        ),
                        ui.input_select(
                            "outlier_action",
                            "Outlier strategy",
                            {
                                "remove": "Remove rows with outliers",
                                "cap": "Cap outliers",
                            },
                            selected="cap",
                        ),
                        ui.input_numeric(
                            "iqr_multiplier",
                            "IQR multiplier",
                            value=1.5,
                            min=0.5,
                            max=5.0,
                            step=0.1,
                        ),
                    ),
                ),

                ui.hr(),
                ui.download_button("download_cleaned", "Download Cleaned Data"),
                full_screen=False,
            ),

            ui.card(
                ui.card_header("Before vs After Overview"),
                ui.layout_columns(
                    ui.value_box("Original rows", ui.output_text("raw_rows")),
                    ui.value_box("Cleaned rows", ui.output_text("clean_rows")),
                    ui.value_box("Original missing", ui.output_text("raw_missing")),
                    ui.value_box("Cleaned missing", ui.output_text("clean_missing")),
                    ui.value_box("Original duplicates", ui.output_text("raw_dupes")),
                    ui.value_box("Cleaned duplicates", ui.output_text("clean_dupes")),
                    col_widths=[4, 4, 4, 4, 4, 4],
                ),
                full_screen=False,
            ),
            col_widths=[4, 8],
        ),

        ui.layout_columns(
            ui.card(
                ui.card_header("Applied Operations"),
                ui.output_text_verbatim("cleaning_summary"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Missing Value Summary (Original Data)"),
                ui.output_data_frame("missing_table"),
                full_screen=True,
            ),
            col_widths=[5, 7],
        ),

        ui.layout_columns(
            ui.card(
                ui.card_header("Cleaned Data Preview"),
                ui.output_data_frame("cleaned_preview"),
                full_screen=True,
            ),
            col_widths=[12],
        ),
    )


def apply_cleaning(df: pd.DataFrame, input) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Apply the selected cleaning and preprocessing steps.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset.

    input : Shiny input object
        User-selected options from the UI.

    Returns
    -------
    cleaned_df : pandas.DataFrame | None
        Dataset after cleaning and preprocessing.

    log : list[str]
        A readable summary of what was applied.
    """
    if df is None:
        return None, ["No dataset loaded."]

    cleaned = df.copy()
    log = []

    # Standardize column names
    if input.standardize_names():
        old_names = list(cleaned.columns)
        new_names = _standardize_column_names(old_names)
        cleaned.columns = new_names

        if old_names != new_names:
            log.append("Standardized column names.")
        else:
            log.append("Column names were already clean.")

    # Trim whitespace in text fields
    if input.trim_strings():
        cleaned = _trim_string_values(cleaned)
        log.append("Trimmed whitespace in text columns.")

    # Remove duplicates
    if input.remove_duplicates():
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        removed = before - len(cleaned)
        log.append(f"Removed {removed} duplicate row(s).")

    # Handle missing values
    strategy = input.missing_strategy()

    if strategy == "drop_rows":
        before = len(cleaned)
        cleaned = cleaned.dropna()
        removed = before - len(cleaned)
        log.append(f"Dropped {removed} row(s) with missing values.")

    elif strategy == "drop_cols":
        before = cleaned.shape[1]
        cleaned = cleaned.dropna(axis=1)
        removed = before - cleaned.shape[1]
        log.append(f"Dropped {removed} column(s) with missing values.")

    elif strategy in {"mean_mode", "median_mode"}:
        numeric_cols = cleaned.select_dtypes(include=np.number).columns
        categorical_cols = cleaned.select_dtypes(exclude=np.number).columns

        for col in numeric_cols:
            if cleaned[col].isna().any():
                fill_value = (
                    cleaned[col].mean()
                    if strategy == "mean_mode"
                    else cleaned[col].median()
                )
                cleaned[col] = cleaned[col].fillna(fill_value)

        for col in categorical_cols:
            if cleaned[col].isna().any():
                mode_series = cleaned[col].mode(dropna=True)
                if not mode_series.empty:
                    cleaned[col] = cleaned[col].fillna(mode_series.iloc[0])

        if strategy == "mean_mode":
            log.append(
                "Imputed missing values using mean for numeric columns and mode for categorical columns."
            )
        else:
            log.append(
                "Imputed missing values using median for numeric columns and mode for categorical columns."
            )

    else:
        log.append("Missing values left unchanged.")

    # Scale numeric columns
    scaling_method = input.scaling_method()
    selected_scale_cols = input.scale_cols()

    if scaling_method != "none" and selected_scale_cols:
        valid_cols = [
            col for col in selected_scale_cols
            if col in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[col])
        ]

        if valid_cols:
            scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
            cleaned[valid_cols] = scaler.fit_transform(cleaned[valid_cols])
            log.append(f"Applied {scaling_method} scaling to: {', '.join(valid_cols)}.")
        else:
            log.append("No valid numeric columns were selected for scaling.")

    # One-hot encode categorical features
    if input.encode_categorical():
        cat_cols = cleaned.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns.tolist()

        if cat_cols:
            cleaned = pd.get_dummies(cleaned, columns=cat_cols, drop_first=False)
            log.append(
                f"Applied one-hot encoding to {len(cat_cols)} categorical column(s)."
            )
        else:
            log.append("No categorical columns were available for encoding.")

    # Handle outliers using IQR
    if input.handle_outliers():
        numeric_cols = cleaned.select_dtypes(include=np.number).columns.tolist()
        multiplier = float(input.iqr_multiplier())

        if numeric_cols:
            if input.outlier_action() == "remove":
                before = len(cleaned)
                mask = pd.Series(True, index=cleaned.index)

                for col in numeric_cols:
                    q1 = cleaned[col].quantile(0.25)
                    q3 = cleaned[col].quantile(0.75)
                    iqr = q3 - q1

                    if pd.isna(iqr) or iqr == 0:
                        continue

                    lower = q1 - multiplier * iqr
                    upper = q3 + multiplier * iqr
                    mask &= cleaned[col].between(lower, upper, inclusive="both")

                cleaned = cleaned.loc[mask].copy()
                removed = before - len(cleaned)
                log.append(f"Removed {removed} row(s) containing IQR-based outliers.")

            else:
                capped_cols = 0

                for col in numeric_cols:
                    q1 = cleaned[col].quantile(0.25)
                    q3 = cleaned[col].quantile(0.75)
                    iqr = q3 - q1

                    if pd.isna(iqr) or iqr == 0:
                        continue

                    lower = q1 - multiplier * iqr
                    upper = q3 + multiplier * iqr
                    cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)
                    capped_cols += 1

                log.append(
                    f"Capped outliers using the IQR rule on {capped_cols} numeric column(s)."
                )
        else:
            log.append("Outlier handling skipped because no numeric columns were available.")

    log.append(
        f"Final dataset shape: {cleaned.shape[0]} row(s) × {cleaned.shape[1]} column(s)."
    )

    return cleaned, log


def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table summarizing missing values by column.
    """
    if df is None:
        return pd.DataFrame(
            columns=["column", "missing_count", "missing_pct", "dtype"]
        )
    return _missing_summary(df)
