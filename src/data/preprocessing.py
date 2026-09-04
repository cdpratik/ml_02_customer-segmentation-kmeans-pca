import pandas as pd
from typing import Optional


class DataCleaner:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.cleaning_report = {
            "initial_shape": None,
            "final_shape": None,
            "steps": []
        }

    def _record_step(self, name, before, after):
        self.cleaning_report["steps"].append({
            "step": name,
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after
        })

        if self.verbose:
            print(f"{name}: {before:,} -> {after:,}")

    def clean_data(
        self,
        df: pd.DataFrame,
        remove_outliers=False,
        outlier_columns: Optional[list] = None
    ):

        df = df.copy()

        self.cleaning_report["initial_shape"] = df.shape

        # Remove missing customer IDs
        before = len(df)
        df = df.dropna(subset=["Customer ID"])
        self._record_step(
            "Remove missing Customer ID",
            before,
            len(df)
        )

        # Remove cancelled invoices
        before = len(df)
        df = df[
            ~df["Invoice"].astype(str).str.startswith("C")
        ]
        self._record_step(
            "Remove cancelled transactions",
            before,
            len(df)
        )

        # Remove invalid quantities
        before = len(df)
        df = df[df["Quantity"] > 0]
        self._record_step(
            "Remove invalid quantities",
            before,
            len(df)
        )

        # Remove invalid prices
        before = len(df)
        df = df[df["Price"] > 0]
        self._record_step(
            "Remove invalid prices",
            before,
            len(df)
        )

        # Remove duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        self._record_step(
            "Remove duplicates",
            before,
            len(df)
        )

        # Create revenue
        df["Revenue"] = df["Quantity"] * df["Price"]

        # Optional outlier removal
        if remove_outliers and outlier_columns:

            for column in outlier_columns:

                if column not in df.columns:
                    continue

                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                before = len(df)

                df = df[
                    (df[column] >= lower) &
                    (df[column] <= upper)
                ]

                self._record_step(
                    f"Remove outliers from {column}",
                    before,
                    len(df)
                )

        self.cleaning_report["final_shape"] = df.shape

        return df

    def get_cleaning_report(self):
        return self.cleaning_report

    def save_cleaning_report(self, file_path):

        import json

        report = self.cleaning_report.copy()

        initial_rows = report["initial_shape"][0]
        final_rows = report["final_shape"][0]

        report["summary"] = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "rows_removed": initial_rows - final_rows,
            "retention_percentage": round(
                final_rows / initial_rows * 100,
                2
            )
        }

        with open(file_path, "w") as file:
            json.dump(
                report,
                file,
                indent=4,
                default=str
            )


def validate_cleaned_data(df):

    validation = {
        "has_customer_id": df["Customer ID"].notna().all(),
        "positive_quantities": (df["Quantity"] > 0).all(),
        "positive_prices": (df["Price"] > 0).all(),
        "no_cancelled_invoices": (
            ~df["Invoice"]
            .astype(str)
            .str.startswith("C")
            .any()
        ),
        "has_revenue": "Revenue" in df.columns,
        "no_duplicates": not df.duplicated().any(),
        "no_missing_values": not df.isna().any().any()
    }

    validation["all_checks_passed"] = all(
        validation.values()
    )

    return validation


def get_cleaning_summary(df_before, df_after):

    summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Unique Customers",
            "Unique Products",
            "Unique Invoices",
            "Missing Values",
            "Duplicates"
        ],
        "Before": [
            len(df_before),
            len(df_before.columns),
            df_before["Customer ID"].nunique(),
            df_before["StockCode"].nunique(),
            df_before["Invoice"].nunique(),
            df_before.isna().sum().sum(),
            df_before.duplicated().sum()
        ],
        "After": [
            len(df_after),
            len(df_after.columns),
            df_after["Customer ID"].nunique(),
            df_after["StockCode"].nunique(),
            df_after["Invoice"].nunique(),
            df_after.isna().sum().sum(),
            df_after.duplicated().sum()
        ]
    })

    return summary
