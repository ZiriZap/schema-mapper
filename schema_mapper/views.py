

import os
import json
import pandas as pd
import zipfile
import openai
from time import time
import shutil
from pathlib import Path
from uuid import uuid4
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt


UPLOAD_LIMIT = 20          # per window
UPLOAD_WINDOW_SEC = 3600   # 1 hour

AI_LIMIT = 5               # per window
AI_WINDOW_SEC = 600        # 10 minutes

MAX_FILE_SIZE_MB = 25
MAX_ZIP_UNCOMPRESSED_MB = 100
MAX_ZIP_FILES = 20

ALLOWED_EXTENSIONS = (".csv", ".zip")
ALL_DTYPES = ["object", "int64", "float64", "bool", "datetime64[ns]"]

# ---------------- Helper functions ----------------


def get_session_upload_dir(request):
    """Creates / returns a session-scoped temp directory"""
    if not request.session.session_key:
        request.session.create()
    path = Path(settings.BASE_DIR) / "tmp_uploads" / request.session.session_key
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_uploaded_file(uploaded_file, target_dir):
    """Safely saves an uploaded file with a unique name"""
    safe_name = f"{uuid4()}_{uploaded_file.name}"
    path = target_dir / safe_name
    with open(path, "wb+") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return path

def validate_uploaded_file(uploaded_file):
    """Validate file size and extension"""
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"{uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit")
    if not uploaded_file.name.lower().endswith(ALLOWED_EXTENSIONS):
        raise ValueError(f"{uploaded_file.name} is not a supported file type")

def cleanup_session_uploads(request):
    """Deletes all temp files for the current session"""
    if request.session.session_key:
        path = Path(settings.BASE_DIR) / "tmp_uploads" / request.session.session_key
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

def check_rate_limit(request, key, limit, window_seconds):
    """
    Session-based rate limiter.
    key: unique action key (e.g. 'upload', 'ai')
    """
    now = int(time())
    rate_data = request.session.get(key, {
        "count": 0,
        "window_start": now
    })

    # Reset window if expired
    if now - rate_data["window_start"] > window_seconds:
        rate_data = {
            "count": 0,
            "window_start": now
        }

    # Check limit
    if rate_data["count"] >= limit:
        remaining = window_seconds - (now - rate_data["window_start"])
        return False, remaining

    # Increment and save
    rate_data["count"] += 1
    request.session[key] = rate_data
    request.session.modified = True

    return True, None

def read_csv_file(file_path, header_row=0):
    """Reads CSV into a DataFrame"""
    return pd.read_csv(file_path, header=header_row)

def build_schema(df, secondary_df=None):
    """Build schema for display (master-secondary)"""
    schema = []
    secondary_schema = []
    if secondary_df is not None:
        secondary_schema = [{"name": col, "dtype": str(secondary_df[col].dtype)} for col in secondary_df.columns]
        secondary_types = sorted(list({s['dtype'] for s in secondary_schema}))
    else:
        secondary_types = []

    for col in df.columns:
        match_col = None
        if secondary_df is not None:
            match_col = next((s for s in secondary_schema if s['name'] == col), None)
        schema.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "secondary_col": match_col['name'] if match_col else "",
            "secondary_dtype": match_col['dtype'] if match_col else "",
        })
    return schema, secondary_schema, secondary_types

def merge_dataframes(master_df, mapping_rows, secondary_df=None, secondary_dfs=None):
    """Merge DataFrames according to mapping_rows (supports multi-merge)"""
    merged_df = pd.DataFrame()
    is_multi_merge = secondary_dfs is not None

    for row in mapping_rows:
        final_name = row['Final Name']
        sec_col = row['Secondary Column']
        if row['Keep']:
            if is_multi_merge:
                value_assigned = False
                for df in secondary_dfs:
                    if sec_col in df.columns:
                        merged_df[final_name] = df[sec_col]
                        value_assigned = True
                        break
                if not value_assigned:
                    merged_df[final_name] = None
            else:
                if secondary_df is not None and sec_col in secondary_df.columns:
                    merged_df[final_name] = secondary_df[sec_col]
                else:
                    merged_df[final_name] = None

    renamed_cols = [r['Final Name'] for r in mapping_rows]
    for col in master_df.columns:
        if col not in merged_df.columns and col not in renamed_cols:
            merged_df[col] = master_df[col]

    return merged_df


def check_rate_limit(request, key, limit, window_seconds):
    """
    Session-based rate limiter.
    key: unique action key (e.g. 'upload', 'ai')
    """
    now = int(time())
    rate_data = request.session.get(key, {
        "count": 0,
        "window_start": now
    })

    # Reset window if expired
    if now - rate_data["window_start"] > window_seconds:
        rate_data = {
            "count": 0,
            "window_start": now
        }

    # Check limit
    if rate_data["count"] >= limit:
        remaining = window_seconds - (now - rate_data["window_start"])
        return False, remaining

    # Increment and save
    rate_data["count"] += 1
    request.session[key] = rate_data
    request.session.modified = True

    return True, None

def require_beta_access(request):
    if not getattr(settings, "BETA_ACCESS_ENABLED", False):
        return None

    # Always allow beta access page itself
    if request.path.startswith("/beta-access"):
        return None

    if request.session.get("beta_access_granted"):
        return None

    return render(
        request,
        "schema_mapper/beta_access.html",
        status=403
    )


# ---------------- Views ----------------

@csrf_exempt
def index(request):
    if getattr(settings, "BETA_ACCESS_ENABLED", False) and not request.session.get("beta_access_granted"):
        return redirect("submit_beta_code")
    return render(request, "schema_mapper/index.html")

# def index(request):
#     gate = require_beta_access(request)
#     if gate:
#         return gate
#
#     return render(request, "schema_mapper/index.html")


@csrf_exempt
def map_display(request):
    gate = require_beta_access(request)
    if gate:
        return gate

    context = {}
    if request.method != 'POST':
        return render(request, "schema_mapper/map_display.html", context)

    allowed, retry_after = check_rate_limit(
        request,
        key="rate_upload",
        limit=UPLOAD_LIMIT,
        window_seconds=UPLOAD_WINDOW_SEC
    )

    if not allowed:
        context["error"] = (
            f"Upload rate limit exceeded. Try again in {retry_after} seconds."
        )
        return render(request, "schema_mapper/map_display.html", context)

    MAX_TOTAL_UPLOAD_MB = 50
    total_upload_size = sum(f.size for f in request.FILES.values())

    if total_upload_size > MAX_TOTAL_UPLOAD_MB * 1024 * 1024:
        context['error'] = (
            f"Total upload size exceeds {MAX_TOTAL_UPLOAD_MB}MB limit"
        )
        return render(request, "schema_mapper/map_display.html", context)

    cleanup_session_uploads(request)
    temp_dir = get_session_upload_dir(request)
    mode = request.POST.get('mode')

    if mode == "master_secondary":
        master_file = request.FILES.get('master')
        secondary_file = request.FILES.get('secondary')
        if not master_file or not secondary_file:
            context['error'] = "Missing master or secondary file."
            return render(request, "schema_mapper/map_display.html", context)

        try:
            validate_uploaded_file(master_file)
            validate_uploaded_file(secondary_file)
        except ValueError as e:
            context['error'] = str(e)
            return render(request, "schema_mapper/map_display.html", context)

        master_path = save_uploaded_file(master_file, temp_dir)
        secondary_path = save_uploaded_file(secondary_file, temp_dir)

        request.session["master_file"] = str(master_path)
        request.session["secondary_file"] = str(secondary_path)

        master_header = int(request.POST.get('master_header_row', 1)) - 1
        secondary_header = int(request.POST.get('secondary_header_row', 1)) - 1

        try:
            master_df = read_csv_file(master_path, master_header)
            secondary_df = read_csv_file(secondary_path, secondary_header)
        except pd.errors.ParserError as e:
            context['error'] = f"Error reading CSV: {e}"
            return render(request, "schema_mapper/map_display.html", context)

        master_schema, secondary_schema, secondary_types = build_schema(master_df, secondary_df)

        context.update({
            'master_schema': master_schema,
            'secondary_schema': secondary_schema,
            'secondary_types': secondary_types,
            'master_header_row': master_header + 1,
            'secondary_header_row': secondary_header + 1,
            'file_names': ["Master File", "Secondary File"],
            'table_previews': {
                "Master File": master_df.head(3).to_html(classes="table table-bordered table-striped table-sm", index=False),
                "Secondary File": secondary_df.head(3).to_html(classes="table table-bordered table-striped table-sm", index=False)
            }
        })

    elif mode == "multi_merge":
        uploaded_files = request.FILES.getlist('files')
        header_rows = request.POST.getlist('header_rows')

        if len(uploaded_files) < 2:
            context['error'] = "Please upload at least 2 files for multi-merge."
            return render(request, "schema_mapper/map_display.html", context)

        dataframes = {}
        all_columns = set()

        for i, f in enumerate(uploaded_files):
            try:
                validate_uploaded_file(f)
            except ValueError as e:
                context['error'] = str(e)
                cleanup_session_uploads(request)
                return render(request, "schema_mapper/map_display.html", context)

            file_path = save_uploaded_file(f, temp_dir)
            header_row = int(header_rows[i]) - 1 if i < len(header_rows) else 0
            try:
                df = read_csv_file(file_path, header_row)
            except pd.errors.ParserError as e:
                context['error'] = f"Error reading {f.name}: {e}"
                return render(request, "schema_mapper/map_display.html", context)

            dataframes[f.name] = df
            all_columns.update(df.columns)

        file_names = list(dataframes.keys())
        master_fname = file_names[0]
        master_df = dataframes[master_fname]

        combined_schema = []
        for col in sorted(all_columns):
            type_file1 = str(master_df[col].dtype) if col in master_df.columns else ""
            secondary_columns = []
            secondary_schema_list = []

            for fname in file_names[1:]:
                df = dataframes[fname]
                if col in df.columns:
                    dtype = str(df[col].dtype)
                    secondary_columns.append({"name": col})
                    secondary_schema_list.append({"name": col, "dtype": dtype})

            secondary_col = secondary_columns[0]['name'] if secondary_columns else ""
            secondary_dtype = secondary_schema_list[0]['dtype'] if secondary_schema_list else ""

            combined_schema.append({
                "name": col,
                "type_file1": type_file1,
                "secondary_columns": secondary_columns,
                "secondary_schema": secondary_schema_list,
                "secondary_col": secondary_col,
                "secondary_dtype": secondary_dtype,
            })

        secondary_types_set = set()
        for entry in combined_schema:
            for s in entry['secondary_schema']:
                secondary_types_set.add(s['dtype'])
        secondary_types = list(secondary_types_set) + [dt for dt in ALL_DTYPES if dt not in secondary_types_set]

        context.update({
            'combined_schema': combined_schema,
            'file_names': file_names,
            'secondary_types': secondary_types,
            'table_previews': {
                fname: df.head(3).to_html(classes="table table-bordered table-striped table-sm", index=False)
                for fname, df in dataframes.items()
            }
        })

    return render(request, "schema_mapper/map_display.html", context)


@csrf_exempt
def save_mapping(request):
    gate = require_beta_access(request)
    if gate:
        return gate

    if request.method != 'POST':
        return HttpResponse("Invalid request", status=400)

    allowed, retry_after = check_rate_limit(
        request,
        key="rate_merge",
        limit=UPLOAD_LIMIT,
        window_seconds=UPLOAD_WINDOW_SEC
    )

    if not allowed:
        return HttpResponse(
            f"Merge rate limit exceeded. Try again in {retry_after} seconds.",
            status=429
        )

    master_path = request.session.get('master_file')
    secondary_path = request.session.get('secondary_file')

    if not master_path:
        return HttpResponse("Session expired or files missing.", status=400)

    try:
        master_df = read_csv_file(master_path)
    except Exception as e:
        return HttpResponse(f"Error reading master file: {e}", status=400)

    # Check if this is master-secondary or multi-merge
    multi_merge_files = request.FILES.getlist('files')
    is_multi_merge = len(multi_merge_files) > 0

    secondary_df = None
    secondary_dfs = []

    if not is_multi_merge:
        # Master-secondary scenario
        if not secondary_path:
            return HttpResponse("Secondary file missing in session.", status=400)
        try:
            secondary_df = read_csv_file(secondary_path)
        except Exception as e:
            return HttpResponse(f"Error reading secondary file: {e}", status=400)
    else:
        # Multi-merge scenario
        temp_dir = get_session_upload_dir(request)
        for f in multi_merge_files:
            try:
                validate_uploaded_file(f)
            except ValueError as e:
                cleanup_session_uploads(request)
                return HttpResponse(str(e), status=400)
            file_path = save_uploaded_file(f, temp_dir)
            try:
                df = read_csv_file(file_path)
            except Exception as e:
                cleanup_session_uploads(request)
                return HttpResponse(f"Error reading {f.name}: {e}", status=400)
            secondary_dfs.append(df)

    # Build mapping_rows from POST form
    mapping_rows = []
    master_columns = list(master_df.columns)
    for idx, col_name in enumerate(master_columns):
        secondary_col = request.POST.get(f'secondary_col_{col_name}', '')
        secondary_type = request.POST.get(f'secondary_type_{col_name}', 'object')
        keep = request.POST.get(f'keep_{col_name}') == 'on'
        match = request.POST.get(f'match_{col_name}') == 'on'
        master_type = request.POST.get(f'master_type_{col_name}', 'object')
        final_name = request.POST.get(f'final_name_{idx}', col_name)
        mapping_rows.append({
            'Master Column': col_name,
            'Final Name': final_name,
            'Master Type': master_type,
            'Secondary Column': secondary_col,
            'Secondary Type': secondary_type,
            'Keep': keep,
            'Match': match
        })

    # --- Save CSV mapping ---
    if 'save_csv' in request.POST:
        mapping_df = pd.DataFrame(mapping_rows)
        mapping_df['Rename Column'] = mapping_df['Final Name']
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="schema_mapping.csv"'
        mapping_df.to_csv(path_or_buf=response, index=False)
        cleanup_session_uploads(request)
        return response

    # --- Merge tables / preview ---
    elif 'merge_tables' in request.POST or 'preview_merge' in request.POST:
        merged_df = merge_dataframes(
            master_df,
            mapping_rows,
            secondary_df=secondary_df,
            secondary_dfs=secondary_dfs if is_multi_merge else None
        )

        if 'preview_merge' in request.POST:
            return JsonResponse(merged_df.fillna("").to_dict(orient='records'), safe=False)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="merged_table.csv"'
        merged_df.to_csv(path_or_buf=response, index=False)
        cleanup_session_uploads(request)
        return response

    return HttpResponse("Invalid request", status=400)

@csrf_exempt
def schema_ai_assist(request):

    if request.method != "POST":
        return JsonResponse({"error": "POST only."}, status=400)

    allowed, retry_after = check_rate_limit(
        request,
        key="rate_merge",
        limit=UPLOAD_LIMIT,
        window_seconds=UPLOAD_WINDOW_SEC
    )

    if not allowed:
        return HttpResponse(
            f"Merge rate limit exceeded. Try again in {retry_after} seconds.",
            status=429
        )

    if settings.BETA_ACCESS_ENABLED and settings.BETA_AI_ONLY:
        if not request.session.get("beta_access_granted"):
            return JsonResponse(
                {"error": "Beta access required"},
                status=403
            )

    try:
        body = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)

    schema = body.get("schema")

    if not schema or not isinstance(schema, list):
        return JsonResponse({"error": "Schema must be a non-empty list"}, status=400)

    prompt = (
        "You are an expert data engineer.\n\n"
        "Analyze the provided schema and return:\n"
        "- Column equivalences\n"
        "- Suggested best matches\n"
        "- Possible primary or merge keys\n"
        "- Type conflicts or warnings\n"
        "- Merge readiness score (0–100)\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise, concise data engineering assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=700
        )

        output = response.choices[0].message.content

    except Exception as e:
        return JsonResponse(
            {"error": "AI service unavailable", "details": str(e)},
            status=500
        )

    return JsonResponse(
        {
            "status": "success",
            "analysis": output
        }
    )

def extract_and_validate_zip(zip_path, extract_to):

    total_uncompressed = 0
    extracted_files = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if len(zip_ref.infolist()) > MAX_ZIP_FILES:
            raise ValueError("ZIP contains too many files")

        for info in zip_ref.infolist():
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_ZIP_UNCOMPRESSED_MB * 1024 * 1024:
                raise ValueError("ZIP expands beyond allowed size")

        zip_ref.extractall(extract_to)
        extracted_files = [
            extract_to / f.filename
            for f in zip_ref.infolist()
            if f.filename.lower().endswith(".csv")
        ]

    if not extracted_files:
        raise ValueError("ZIP does not contain any CSV files")

    return extracted_files

@csrf_exempt
def submit_beta_code(request):
    if request.method != "POST":
        return render(request, "schema_mapper/beta_access.html")

    code = request.POST.get("beta_code", "").strip()
    valid_codes = getattr(settings, "BETA_ACCESS_CODES", set())

    if code in valid_codes:
        # Grant access
        request.session["beta_access_granted"] = True
        request.session.modified = True

        # IMPORTANT: redirect using URL path, not name
        return redirect("/")

    return render(
        request,
        "schema_mapper/beta_access.html",
        {"error": "Invalid beta access code"},
        status=403
    )

# import os
# import json
# import pandas as pd
# import zipfile
# import openai
# from time import time
# import shutil
# from pathlib import Path
# from uuid import uuid4
# from django.conf import settings
# from django.http import HttpResponse, JsonResponse
# from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt
#
#
# UPLOAD_LIMIT = 20          # per window
# UPLOAD_WINDOW_SEC = 3600   # 1 hour
#
# AI_LIMIT = 5               # per window
# AI_WINDOW_SEC = 600        # 10 minutes
#
# MAX_FILE_SIZE_MB = 25
# MAX_ZIP_UNCOMPRESSED_MB = 100
# MAX_ZIP_FILES = 20
#
# ALLOWED_EXTENSIONS = (".csv", ".zip")
# ALL_DTYPES = ["object", "int64", "float64", "bool", "datetime64[ns]"]
#
# # ---------------- Helper functions ----------------
#
#
# def get_session_upload_dir(request):
#     """Creates / returns a session-scoped temp directory"""
#     if not request.session.session_key:
#         request.session.create()
#     path = Path(settings.BASE_DIR) / "tmp_uploads" / request.session.session_key
#     path.mkdir(parents=True, exist_ok=True)
#     return path
#
# def save_uploaded_file(uploaded_file, target_dir):
#     """Safely saves an uploaded file with a unique name"""
#     safe_name = f"{uuid4()}_{uploaded_file.name}"
#     path = target_dir / safe_name
#     with open(path, "wb+") as f:
#         for chunk in uploaded_file.chunks():
#             f.write(chunk)
#     return path
#
# def validate_uploaded_file(uploaded_file):
#     """Validate file size and extension"""
#     if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
#         raise ValueError(f"{uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit")
#     if not uploaded_file.name.lower().endswith(ALLOWED_EXTENSIONS):
#         raise ValueError(f"{uploaded_file.name} is not a supported file type")
#
# def cleanup_session_uploads(request):
#     """Deletes all temp files for the current session"""
#     if request.session.session_key:
#         path = Path(settings.BASE_DIR) / "tmp_uploads" / request.session.session_key
#         if path.exists():
#             shutil.rmtree(path, ignore_errors=True)
#
# def check_rate_limit(request, key, limit, window_seconds):
#     """
#     Session-based rate limiter.
#     key: unique action key (e.g. 'upload', 'ai')
#     """
#     now = int(time())
#     rate_data = request.session.get(key, {
#         "count": 0,
#         "window_start": now
#     })
#
#     # Reset window if expired
#     if now - rate_data["window_start"] > window_seconds:
#         rate_data = {
#             "count": 0,
#             "window_start": now
#         }
#
#     # Check limit
#     if rate_data["count"] >= limit:
#         remaining = window_seconds - (now - rate_data["window_start"])
#         return False, remaining
#
#     # Increment and save
#     rate_data["count"] += 1
#     request.session[key] = rate_data
#     request.session.modified = True
#
#     return True, None
#
# def read_csv_file(file_path, header_row=0):
#     """Reads CSV into a DataFrame"""
#     return pd.read_csv(file_path, header=header_row)
#
# def build_schema(df, secondary_df=None):
#     """Build schema for display (master-secondary)"""
#     schema = []
#     secondary_schema = []
#     if secondary_df is not None:
#         secondary_schema = [{"name": col, "dtype": str(secondary_df[col].dtype)} for col in secondary_df.columns]
#         secondary_types = sorted(list({s['dtype'] for s in secondary_schema}))
#     else:
#         secondary_types = []
#
#     for col in df.columns:
#         match_col = None
#         if secondary_df is not None:
#             match_col = next((s for s in secondary_schema if s['name'] == col), None)
#         schema.append({
#             "name": col,
#             "dtype": str(df[col].dtype),
#             "secondary_col": match_col['name'] if match_col else "",
#             "secondary_dtype": match_col['dtype'] if match_col else "",
#         })
#     return schema, secondary_schema, secondary_types
#
# def merge_dataframes(master_df, mapping_rows, secondary_df=None, secondary_dfs=None):
#     """Merge DataFrames according to mapping_rows (supports multi-merge)"""
#     merged_df = pd.DataFrame()
#     is_multi_merge = secondary_dfs is not None
#
#     for row in mapping_rows:
#         final_name = row['Final Name']
#         sec_col = row['Secondary Column']
#         if row['Keep']:
#             if is_multi_merge:
#                 value_assigned = False
#                 for df in secondary_dfs:
#                     if sec_col in df.columns:
#                         merged_df[final_name] = df[sec_col]
#                         value_assigned = True
#                         break
#                 if not value_assigned:
#                     merged_df[final_name] = None
#             else:
#                 if secondary_df is not None and sec_col in secondary_df.columns:
#                     merged_df[final_name] = secondary_df[sec_col]
#                 else:
#                     merged_df[final_name] = None
#
#     renamed_cols = [r['Final Name'] for r in mapping_rows]
#     for col in master_df.columns:
#         if col not in merged_df.columns and col not in renamed_cols:
#             merged_df[col] = master_df[col]
#
#     return merged_df
#
# def check_rate_limit(request, key, limit, window_seconds):
#     """
#     Session-based rate limiter.
#     key: unique action key (e.g. 'upload', 'ai')
#     """
#     now = int(time())
#     rate_data = request.session.get(key, {
#         "count": 0,
#         "window_start": now
#     })
#
#     # Reset window if expired
#     if now - rate_data["window_start"] > window_seconds:
#         rate_data = {
#             "count": 0,
#             "window_start": now
#         }
#
#     # Check limit
#     if rate_data["count"] >= limit:
#         remaining = window_seconds - (now - rate_data["window_start"])
#         return False, remaining
#
#     # Increment and save
#     rate_data["count"] += 1
#     request.session[key] = rate_data
#     request.session.modified = True
#
#     return True, None
#
#
# # ---------------- Views ----------------
#
# @csrf_exempt
# def index(request):
#     return render(request, "schema_mapper/index.html")
#
#
# @csrf_exempt
# def map_display(request):
#     context = {}
#     if request.method != 'POST':
#         return render(request, "schema_mapper/map_display.html", context)
#
#     allowed, retry_after = check_rate_limit(
#         request,
#         key="rate_upload",
#         limit=UPLOAD_LIMIT,
#         window_seconds=UPLOAD_WINDOW_SEC
#     )
#
#     if not allowed:
#         context["error"] = (
#             f"Upload rate limit exceeded. Try again in {retry_after} seconds."
#         )
#         return render(request, "schema_mapper/map_display.html", context)
#
#     MAX_TOTAL_UPLOAD_MB = 50
#     total_upload_size = sum(f.size for f in request.FILES.values())
#
#     if total_upload_size > MAX_TOTAL_UPLOAD_MB * 1024 * 1024:
#         context['error'] = (
#             f"Total upload size exceeds {MAX_TOTAL_UPLOAD_MB}MB limit"
#         )
#         return render(request, "schema_mapper/map_display.html", context)
#
#     cleanup_session_uploads(request)
#     temp_dir = get_session_upload_dir(request)
#     mode = request.POST.get('mode')
#
#     if mode == "master_secondary":
#         master_file = request.FILES.get('master')
#         secondary_file = request.FILES.get('secondary')
#         if not master_file or not secondary_file:
#             context['error'] = "Missing master or secondary file."
#             return render(request, "schema_mapper/map_display.html", context)
#
#         try:
#             validate_uploaded_file(master_file)
#             validate_uploaded_file(secondary_file)
#         except ValueError as e:
#             context['error'] = str(e)
#             return render(request, "schema_mapper/map_display.html", context)
#
#         master_path = save_uploaded_file(master_file, temp_dir)
#         secondary_path = save_uploaded_file(secondary_file, temp_dir)
#
#         request.session["master_file"] = str(master_path)
#         request.session["secondary_file"] = str(secondary_path)
#
#         master_header = int(request.POST.get('master_header_row', 1)) - 1
#         secondary_header = int(request.POST.get('secondary_header_row', 1)) - 1
#
#         try:
#             master_df = read_csv_file(master_path, master_header)
#             secondary_df = read_csv_file(secondary_path, secondary_header)
#         except pd.errors.ParserError as e:
#             context['error'] = f"Error reading CSV: {e}"
#             return render(request, "schema_mapper/map_display.html", context)
#
#         master_schema, secondary_schema, secondary_types = build_schema(master_df, secondary_df)
#
#         context.update({
#             'master_schema': master_schema,
#             'secondary_schema': secondary_schema,
#             'secondary_types': secondary_types,
#             'master_header_row': master_header + 1,
#             'secondary_header_row': secondary_header + 1,
#             'file_names': ["Master File", "Secondary File"],
#             'table_previews': {
#                 "Master File": master_df.head(3).to_html(classes="table table-bordered table-striped table-sm", index=False),
#                 "Secondary File": secondary_df.head(3).to_html(classes="table table-bordered table-striped table-sm", index=False)
#             }
#         })
#
#     elif mode == "multi_merge":
#         uploaded_files = request.FILES.getlist('files')
#         header_rows = request.POST.getlist('header_rows')
#
#         if len(uploaded_files) < 2:
#             context['error'] = "Please upload at least 2 files for multi-merge."
#             return render(request, "schema_mapper/map_display.html", context)
#
#         dataframes = {}
#         all_columns = set()
#
#         for i, f in enumerate(uploaded_files):
#             try:
#                 validate_uploaded_file(f)
#             except ValueError as e:
#                 context['error'] = str(e)
#                 cleanup_session_uploads(request)
#                 return render(request, "schema_mapper/map_display.html", context)
#
#             file_path = save_uploaded_file(f, temp_dir)
#             header_row = int(header_rows[i]) - 1 if i < len(header_rows) else 0
#             try:
#                 df = read_csv_file(file_path, header_row)
#             except pd.errors.ParserError as e:
#                 context['error'] = f"Error reading {f.name}: {e}"
#                 return render(request, "schema_mapper/map_display.html", context)
#
#             dataframes[f.name] = df
#             all_columns.update(df.columns)
#
#         file_names = list(dataframes.keys())
#         master_fname = file_names[0]
#         master_df = dataframes[master_fname]
#
#         combined_schema = []
#         for col in sorted(all_columns):
#             type_file1 = str(master_df[col].dtype) if col in master_df.columns else ""
#             secondary_columns = []
#             secondary_schema_list = []
#
#             for fname in file_names[1:]:
#                 df = dataframes[fname]
#                 if col in df.columns:
#                     dtype = str(df[col].dtype)
#                     secondary_columns.append({"name": col})
#                     secondary_schema_list.append({"name": col, "dtype": dtype})
#
#             secondary_col = secondary_columns[0]['name'] if secondary_columns else ""
#             secondary_dtype = secondary_schema_list[0]['dtype'] if secondary_schema_list else ""
#
#             combined_schema.append({
#                 "name": col,
#                 "type_file1": type_file1,
#                 "secondary_columns": secondary_columns,
#                 "secondary_schema": secondary_schema_list,
#                 "secondary_col": secondary_col,
#                 "secondary_dtype": secondary_dtype,
#             })
#
#         secondary_types_set = set()
#         for entry in combined_schema:
#             for s in entry['secondary_schema']:
#                 secondary_types_set.add(s['dtype'])
#         secondary_types = list(secondary_types_set) + [dt for dt in ALL_DTYPES if dt not in secondary_types_set]
#
#         context.update({
#             'combined_schema': combined_schema,
#             'file_names': file_names,
#             'secondary_types': secondary_types,
#             'table_previews': {
#                 fname: df.head(3).to_html(classes="table table-bordered table-striped table-sm", index=False)
#                 for fname, df in dataframes.items()
#             }
#         })
#
#     return render(request, "schema_mapper/map_display.html", context)
#
#
# @csrf_exempt
# def save_mapping(request):
#     if request.method != 'POST':
#         return HttpResponse("Invalid request", status=400)
#
#     allowed, retry_after = check_rate_limit(
#         request,
#         key="rate_merge",
#         limit=UPLOAD_LIMIT,
#         window_seconds=UPLOAD_WINDOW_SEC
#     )
#
#     if not allowed:
#         return HttpResponse(
#             f"Merge rate limit exceeded. Try again in {retry_after} seconds.",
#             status=429
#         )
#
#     master_path = request.session.get('master_file')
#     secondary_path = request.session.get('secondary_file')
#
#     if not master_path:
#         return HttpResponse("Session expired or files missing.", status=400)
#
#     try:
#         master_df = read_csv_file(master_path)
#     except Exception as e:
#         return HttpResponse(f"Error reading master file: {e}", status=400)
#
#     # Check if this is master-secondary or multi-merge
#     multi_merge_files = request.FILES.getlist('files')
#     is_multi_merge = len(multi_merge_files) > 0
#
#     secondary_df = None
#     secondary_dfs = []
#
#     if not is_multi_merge:
#         # Master-secondary scenario
#         if not secondary_path:
#             return HttpResponse("Secondary file missing in session.", status=400)
#         try:
#             secondary_df = read_csv_file(secondary_path)
#         except Exception as e:
#             return HttpResponse(f"Error reading secondary file: {e}", status=400)
#     else:
#         # Multi-merge scenario
#         temp_dir = get_session_upload_dir(request)
#         for f in multi_merge_files:
#             try:
#                 validate_uploaded_file(f)
#             except ValueError as e:
#                 cleanup_session_uploads(request)
#                 return HttpResponse(str(e), status=400)
#             file_path = save_uploaded_file(f, temp_dir)
#             try:
#                 df = read_csv_file(file_path)
#             except Exception as e:
#                 cleanup_session_uploads(request)
#                 return HttpResponse(f"Error reading {f.name}: {e}", status=400)
#             secondary_dfs.append(df)
#
#     # Build mapping_rows from POST form
#     mapping_rows = []
#     master_columns = list(master_df.columns)
#     for idx, col_name in enumerate(master_columns):
#         secondary_col = request.POST.get(f'secondary_col_{col_name}', '')
#         secondary_type = request.POST.get(f'secondary_type_{col_name}', 'object')
#         keep = request.POST.get(f'keep_{col_name}') == 'on'
#         match = request.POST.get(f'match_{col_name}') == 'on'
#         master_type = request.POST.get(f'master_type_{col_name}', 'object')
#         final_name = request.POST.get(f'final_name_{idx}', col_name)
#         mapping_rows.append({
#             'Master Column': col_name,
#             'Final Name': final_name,
#             'Master Type': master_type,
#             'Secondary Column': secondary_col,
#             'Secondary Type': secondary_type,
#             'Keep': keep,
#             'Match': match
#         })
#
#     # --- Save CSV mapping ---
#     if 'save_csv' in request.POST:
#         mapping_df = pd.DataFrame(mapping_rows)
#         mapping_df['Rename Column'] = mapping_df['Final Name']
#         response = HttpResponse(content_type='text/csv')
#         response['Content-Disposition'] = 'attachment; filename="schema_mapping.csv"'
#         mapping_df.to_csv(path_or_buf=response, index=False)
#         cleanup_session_uploads(request)
#         return response
#
#     # --- Merge tables / preview ---
#     elif 'merge_tables' in request.POST or 'preview_merge' in request.POST:
#         merged_df = merge_dataframes(
#             master_df,
#             mapping_rows,
#             secondary_df=secondary_df,
#             secondary_dfs=secondary_dfs if is_multi_merge else None
#         )
#
#         if 'preview_merge' in request.POST:
#             return JsonResponse(merged_df.fillna("").to_dict(orient='records'), safe=False)
#
#         response = HttpResponse(content_type='text/csv')
#         response['Content-Disposition'] = 'attachment; filename="merged_table.csv"'
#         merged_df.to_csv(path_or_buf=response, index=False)
#         cleanup_session_uploads(request)
#         return response
#
#     return HttpResponse("Invalid request", status=400)
#
# @csrf_exempt
# def schema_ai_assist(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "POST only."}, status=400)
#
#     allowed, retry_after = check_rate_limit(
#         request,
#         key="rate_merge",
#         limit=UPLOAD_LIMIT,
#         window_seconds=UPLOAD_WINDOW_SEC
#     )
#
#     if not allowed:
#         return HttpResponse(
#             f"Merge rate limit exceeded. Try again in {retry_after} seconds.",
#             status=429
#         )
#
#     try:
#         body = json.loads(request.body.decode("utf-8"))
#     except json.JSONDecodeError:
#         return JsonResponse({"error": "Invalid JSON payload"}, status=400)
#
#     schema = body.get("schema")
#
#     if not schema or not isinstance(schema, list):
#         return JsonResponse({"error": "Schema must be a non-empty list"}, status=400)
#
#     prompt = (
#         "You are an expert data engineer.\n\n"
#         "Analyze the provided schema and return:\n"
#         "- Column equivalences\n"
#         "- Suggested best matches\n"
#         "- Possible primary or merge keys\n"
#         "- Type conflicts or warnings\n"
#         "- Merge readiness score (0–100)\n\n"
#         f"Schema:\n{json.dumps(schema, indent=2)}"
#     )
#
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a precise, concise data engineering assistant."
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0.2,
#             max_tokens=700
#         )
#
#         output = response.choices[0].message.content
#
#     except Exception as e:
#         return JsonResponse(
#             {"error": "AI service unavailable", "details": str(e)},
#             status=500
#         )
#
#     return JsonResponse(
#         {
#             "status": "success",
#             "analysis": output
#         }
#     )
#
# def extract_and_validate_zip(zip_path, extract_to):
#     total_uncompressed = 0
#     extracted_files = []
#
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         if len(zip_ref.infolist()) > MAX_ZIP_FILES:
#             raise ValueError("ZIP contains too many files")
#
#         for info in zip_ref.infolist():
#             total_uncompressed += info.file_size
#             if total_uncompressed > MAX_ZIP_UNCOMPRESSED_MB * 1024 * 1024:
#                 raise ValueError("ZIP expands beyond allowed size")
#
#         zip_ref.extractall(extract_to)
#         extracted_files = [
#             extract_to / f.filename
#             for f in zip_ref.infolist()
#             if f.filename.lower().endswith(".csv")
#         ]
#
#     if not extracted_files:
#         raise ValueError("ZIP does not contain any CSV files")
#
#     return extracted_files
