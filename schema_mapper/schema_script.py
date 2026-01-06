import os
import pandas as pd
import geopandas as gpd

UPLOAD_SAMPLE_LIMIT = 5  # number of rows to preview


def read_table(file_path, sample_limit=UPLOAD_SAMPLE_LIMIT):
    """
    Reads supported file formats into a DataFrame.
    Supports CSV, Excel, JSON, GeoJSON, Parquet.
    Automatically detects delimiters and skips bad lines.
    Returns: DataFrame, list of fields (dicts with name/type), preview rows (list of lists)
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.csv']:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            # Auto-detect delimiter
            df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif ext in ['.json']:
        df = pd.read_json(file_path)
    elif ext in ['.geojson', '.shp']:
        gdf = gpd.read_file(file_path)
        df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
    elif ext in ['.parquet']:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Preview rows
    rows = df.head(sample_limit).values.tolist()
    fields = [{"name": c, "type": str(df[c].dtype)} for c in df.columns]

    return df, fields, rows


def detect_header_row(file_path, max_rows=10):
    """
    Attempts to detect which row likely contains column headers.
    Returns 1-based row index.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()[:max_rows]]

        scores = []
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if not parts:
                continue
            numeric_count = sum(p.replace('.', '', 1).isdigit() for p in parts)
            unique_count = len(set(parts))
            score = (unique_count * 2) - numeric_count
            scores.append((i + 1, score))
        if scores:
            best_row = max(scores, key=lambda x: x[1])[0]
            return best_row
        else:
            return 1
    except Exception as e:
        print(f"[detect_header_row] Failed to detect header in {file_path}: {e}")
        return 1


def get_fc_preview(file_path, header_row=1):
    """
    Reads a file safely and returns:
    - schema (list of dicts with 'name' and 'dtype')
    - fields (list of column names)
    - preview rows (list of lists)
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.csv']:
        df = pd.read_csv(file_path, header=header_row - 1, sep=None, engine='python', on_bad_lines='skip')
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, header=header_row - 1)
    elif ext in ['.json']:
        df = pd.read_json(file_path)
    elif ext in ['.geojson', '.shp']:
        gdf = gpd.read_file(file_path)
        df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
    elif ext in ['.parquet']:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    df = df.dropna(axis=1, how='all')
    df_preview = df.dropna(how='all').head(UPLOAD_SAMPLE_LIMIT)

    fields = list(df.columns)
    rows = df_preview.values.tolist()
    schema = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]

    return schema, fields, rows


def create_mapping_html(master_schema, secondary_schema, master_preview, secondary_preview, output_path):
    """
    Generates a simple HTML preview of master vs secondary table schema
    """
    def build_table(schema, preview):
        header = "".join([f"<th>{col['name']}</th>" for col in schema])
        rows_html = ""
        for row in preview:
            row_cells = "".join([f"<td>{v}</td>" for v in row])
            rows_html += f"<tr>{row_cells}</tr>"
        return f"<table border='1'><thead><tr>{header}</tr></thead><tbody>{rows_html}</tbody></table>"

    html_content = f"""
    <html>
    <head><title>Schema Mapping</title></head>
    <body>
        <h2>Master Table Schema</h2>
        {build_table(master_schema, master_preview)}

        <h2>Secondary Table Schema</h2>
        {build_table(secondary_schema, secondary_preview)}
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
