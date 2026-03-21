from shiny import App, reactive, render, ui
from modules.data_loader import load_data, upload_ui
from modules.cleaning import apply_cleaning, cleaning_ui

custom_css = """
body {
    background-color: #f6f8fb;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.navbar {
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    background-color: white !important;
}

.app-title-box {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 28px 32px;
    border-radius: 18px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.18);
}

.app-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.app-subtitle {
    font-size: 1.05rem;
    opacity: 0.95;
    margin-bottom: 0;
}

.section-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #111827;
    margin-top: 12px;
    margin-bottom: 14px;
}

.section-text {
    color: #4b5563;
    font-size: 1rem;
    margin-bottom: 20px;
}

.card {
    border-radius: 16px !important;
    border: none !important;
    box-shadow: 0 6px 22px rgba(15, 23, 42, 0.08) !important;
}

.form-control, .btn, .form-select {
    border-radius: 12px !important;
}

.btn-primary {
    background-color: #4f46e5 !important;
    border-color: #4f46e5 !important;
}

.table {
    background-color: white;
    border-radius: 12px;
    overflow: hidden;
}

.placeholder-box {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 6px 22px rgba(15, 23, 42, 0.08);
    color: #374151;
}

.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding-top: 10px;
    padding-bottom: 30px;
}
"""

navbar_content = ui.page_navbar(
    ui.nav_panel(
        "Upload",
        ui.div(
            {"class": "main-container"},
            ui.div(
                {"class": "app-title-box"},
                ui.div({"class": "app-title"}, "Project 2 Data App"),
                ui.p(
                    {"class": "app-subtitle"},
                    "An interactive data application for uploading, exploring, and transforming datasets."
                ),
            ),
            upload_ui(),
            ui.h2("Dataset Preview", {"class": "section-title"}),
            ui.p(
                "Upload a dataset to inspect its size and preview the first few rows.",
                {"class": "section-text"}
            ),
            ui.output_text_verbatim("upload_status"),
            ui.output_table("data_preview"),
        )
    ),
    ui.nav_panel(
        "Cleaning",
        cleaning_ui()
    ),
    ui.nav_panel(
        "Feature Engineering",
        ui.div(
            {"class": "main-container"},
            ui.h2("Feature Engineering", {"class": "section-title"}),
            ui.div(
                {"class": "placeholder-box"},
                ui.p("This section will be implemented by Person C."),
                ui.p("Suggested features: create new variables, transformations, binning, and feature previews.")
            )
        )
    ),
    ui.nav_panel(
        "EDA",
        ui.div(
            {"class": "main-container"},
            ui.h2("Exploratory Data Analysis", {"class": "section-title"}),
            ui.div(
                {"class": "placeholder-box"},
                ui.p("This section will be implemented by Person D."),
                ui.p("Suggested features: summary statistics, histograms, scatter plots, boxplots, and correlation analysis.")
            )
        )
    ),
    title="Project 2 Data App",
)

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(custom_css)
    ),
    navbar_content
)


def server(input, output, session):
    @reactive.calc
    def dataset():
        file_info = input.file_upload()
        return load_data(file_info)

    @reactive.calc
    def cleaned_result():
        df = dataset()
        return apply_cleaning(df, input)

    @output
    @render.text
    def upload_status():
        df = dataset()
        if df is None:
            return "No dataset uploaded yet."
        return f"Upload successful. Rows: {df.shape[0]}, Columns: {df.shape[1]}"

    @output
    @render.table
    def data_preview():
        df = dataset()
        if df is None:
            return None
        return df.head()


    
    @output
    @render.text
    def cleaning_summary():
        cleaned_df, log = cleaned_result()
        if log is None:
            return "No cleaning steps applied."
        return "\n".join(log)


    @output
    @render.data_frame
    def missing_table_original():
        df = dataset()
        if df is None:
            return None

        missing = df.isna().sum().to_frame(name="missing_count")
        missing["missing_pct"] = missing["missing_count"] / len(df)
        return missing


    @output
    @render.data_frame
    def missing_table_cleaned():
        cleaned_df, log = cleaned_result()
        if cleaned_df is None:
            return None

        missing = cleaned_df.isna().sum().to_frame(name="missing_count")
        missing["missing_pct"] = missing["missing_count"] / len(cleaned_df)
        return missing


    @output
    @render.data_frame
    def cleaned_preview():
        cleaned_df, log = cleaned_result()
        if cleaned_df is None:
            return None
        return cleaned_df.head()
    
    @output
    @render.text
    def raw_rows():
        df = dataset()
        if df is None:
            return "-"
        return str(df.shape[0])

    @output
    @render.text
    def clean_rows():
        cleaned_df, log = cleaned_result()
        if cleaned_df is None:
            return "-"
        return str(cleaned_df.shape[0])

    @output
    @render.text
    def rows_removed():
        df = dataset()
        cleaned_df, log = cleaned_result()
        if df is None or cleaned_df is None:
            return "-"
        return str(df.shape[0] - cleaned_df.shape[0])

    @output
    @render.text
    def raw_cols():
        df = dataset()
        if df is None:
            return "-"
        return str(df.shape[1])

    @output
    @render.text
    def clean_cols():
        cleaned_df, log = cleaned_result()
        if cleaned_df is None:
            return "-"
        return str(cleaned_df.shape[1])

    @output
    @render.text
    def raw_missing():
        df = dataset()
        if df is None:
            return "-"
        return str(int(df.isna().sum().sum()))

    @output
    @render.text
    def clean_missing():
        cleaned_df, log = cleaned_result()
        if cleaned_df is None:
            return "-"
        return str(int(cleaned_df.isna().sum().sum()))

    @output
    @render.text
    def raw_dupes():
        df = dataset()
        if df is None:
            return "-"
        return str(int(df.duplicated().sum()))

    @output
    @render.text
    def clean_dupes():
        cleaned_df, log = cleaned_result()
        if cleaned_df is None:
            return "-"
        return str(int(cleaned_df.duplicated().sum()))
    


app = App(app_ui, server)