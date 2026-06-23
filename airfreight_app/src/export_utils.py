"""Export utilities: CSV bytes and multi-sheet Excel bytes."""

import io
import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """
    sheets: {"Sheet Name": dataframe, ...}
    Sheet names are truncated to 31 chars (Excel limit).
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()
