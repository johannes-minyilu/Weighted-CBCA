# ---------- STEP 1: Mount Google Drive (Colab only) ----------
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False

import pandas as pd
import numpy as np
import glob
import os
import re

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font

# ---------- STEP 2: CONFIG ----------
#SOURCE_FOLDER = "/content/drive/MyDrive/Analysis Results"
SOURCE_FOLDER = r"D:\our\path\to a root\directory\Results & Analysis\Results & Analysis\Analysis-by-Model"
      
#OUTPUT_DIR = "/content/drive/MyDrive/Stat Analyses"
OUTPUT_DIR = r"D:\Final\outpu\path\for Stat Analyses"

OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, "Last Compillation")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Files whose name contains any of these keywords are IGNORED
EXCLUDE_KEYWORDS = [
    "merged", "table1", "table2", "table_s", "tables1",
    "final", "summary_output", "outputs", "all_tables"
]

# Logical row order for the tables
ROW_ORDER = [
    ("accuracy",    "overall"),
    ("macro_recall","macro"),
    ("macro_f1",    "macro"),
    ("recall",      "podo"),
    ("recall",      "scabies"),
    ("recall",      "tungiasis"),
    ("precision",   "podo"),
    ("precision",   "scabies"),
    ("precision",   "tungiasis"),
    ("specificity", "podo"),
    ("specificity", "scabies"),
    ("specificity", "tungiasis"),
    ("f1",          "podo"),
    ("f1",          "scabies"),
    ("f1",          "tungiasis"),
]

# ---------- STEP 3: FIND SOURCE FILES ----------
all_files = (
    glob.glob(os.path.join(SOURCE_FOLDER, "*.csv")) +
    glob.glob(os.path.join(SOURCE_FOLDER, "*.xlsx")) +
    glob.glob(os.path.join(SOURCE_FOLDER, "*.xls"))
)
all_files = [f for f in all_files if not os.path.basename(f).startswith("~$")]

def is_output_file(path):
    name = os.path.basename(path).lower()
    return any(k in name for k in EXCLUDE_KEYWORDS)

source_files = sorted([f for f in all_files if not is_output_file(f)])

print("Source files found:")
for f in source_files:
    print("  -", os.path.basename(f))

if not source_files:
    raise FileNotFoundError(
        "No source files found. Check SOURCE_FOLDER and EXCLUDE_KEYWORDS."
    )

# ---------- STEP 4: LOAD AND TAG ----------
dfs = []
for f in source_files:
    df = pd.read_csv(f) if f.lower().endswith(".csv") else pd.read_excel(f)

    label = os.path.splitext(os.path.basename(f))[0]
    label = re.sub(r"[-_]?statistical[-_]?summary", "", label, flags=re.I)
    label = label.strip("-_ ")

    df["model"] = label
    df["source_file"] = os.path.basename(f)
    dfs.append(df)

long = pd.concat(dfs, ignore_index=True)

# Ensuring that the required columns exist
required = ["metric", "class", "n_samples", "cbca_value", "hybrid_value",
            "difference", "ci_lower_95", "ci_upper_95"]
for col in required:
    if col not in long.columns:
        long[col] = np.nan

# ---------- STEP 5: SIGNIFICANCE STARS ----------
def get_stars(row):
    p = row.get("mcnemar_p_value", np.nan)
    if pd.notna(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"
    lo, hi = row.get("ci_lower_95", np.nan), row.get("ci_upper_95", np.nan)
    if pd.notna(lo) and pd.notna(hi):
        if lo > 0 or hi < 0:
            return "*"
    return "ns"

long["stars"] = long.apply(get_stars, axis=1)

# Direction of significance (for supplementary)
def direction(row):
    d = row["difference"]
    s = row["stars"]
    if pd.isna(d):        return ""
    if s == "ns":         return "ns"
    return "Hybrid" if d > 0 else "CBCA"

long["sig_dir"] = long.apply(direction, axis=1)

# ---------- STEP 6: TEST USED ----------
def get_test(row):
    t = row.get("mcnemar_test", None)
    if pd.notna(t) and str(t).strip() != "":
        t = str(t).lower()
        if "exact" in t:  return "McNemar (exact)"
        if "chi"   in t:  return "McNemar (chi-sq)"
        return f"McNemar ({t})"
    return "Bootstrap CI"

long["test_used"] = long.apply(get_test, axis=1)

# ---------- STEP 7: FORMAT CELLS (Table 1) ----------
def fmt_p(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "p<.001"
    return f"p={p:.3f}".rstrip("0").rstrip(".")

def format_cell(row):
    d = row["difference"]
    if pd.isna(d): return ""
    s = row["stars"]
    p = row.get("mcnemar_p_value", np.nan)
    lo, hi = row.get("ci_lower_95", np.nan), row.get("ci_upper_95", np.nan)
    if pd.notna(p):
        return f"{d:+.3f} {s} ({fmt_p(p)})"
    if pd.notna(lo) and pd.notna(hi):
        return f"{d:+.3f} {s} [{lo:+.3f}, {hi:+.3f}]"
    return f"{d:+.3f} {s}"

long["cell"] = long.apply(format_cell, axis=1)

# ---------- STEP 8: ROW ORDERING ----------
row_df = pd.DataFrame(ROW_ORDER, columns=["metric", "class"])
row_df["_order"] = range(len(row_df))
long = long.merge(row_df, on=["metric", "class"], how="left")
long["_order"] = long["_order"].fillna(999).astype(int)

# Natural numeric sort for model columns (1, 2, ..., 10 — not 1, 10, 2)
def model_sort_key(name):
    nums = re.findall(r"\d+", str(name))
    return (int(nums[0]) if nums else 999, str(name))

model_names = sorted(long["model"].unique(), key=model_sort_key)

# ============================================================
# TABLE 1 — Full paired results
# ============================================================
ref = (long.sort_values("_order")
          .groupby(["metric", "class"], as_index=False)
          .agg(n=("n_samples", "first"),
               test=("test_used", "first")))

pivot_cells = long.pivot_table(
    index=["metric", "class"],
    columns="model",
    values="cell",
    aggfunc="first"
).reset_index()

table1 = ref.merge(pivot_cells, on=["metric", "class"], how="left")
table1 = table1.merge(row_df, on=["metric", "class"], how="left")
table1 = table1.sort_values("_order").drop(columns="_order")
table1 = table1[["metric", "class", "n", "test"] + model_names]

print("\n" + "="*110)
print("TABLE 1 — PAIRED COMPARISON OF HYBRID vs CBCA ACROSS MODELS")
print("="*110)
print(table1.to_string(index=False))

# ============================================================
# TABLE 2 — Summary counts across models (NUMERIC FORMAT)
#   Columns are integers and percentages, NOT fractions,
#   so Excel cannot convert them into dates.
# ============================================================
def summarize(g):
    n = len(g)
    sig = (g["stars"] != "ns")
    hb  = (g["difference"] > 0)
    cb  = (g["difference"] < 0)

    hb_k   = int(hb.sum())
    cb_k   = int(cb.sum())
    sig_k  = int(sig.sum())
    shb_k  = int((hb & sig).sum())
    scb_k  = int((cb & sig).sum())

    return pd.Series({
        "n_models":            n,
        # Integer counts
        "Hybrid better (k)":   hb_k,
        "CBCA better (k)":     cb_k,
        "Significant (k)":     sig_k,
        "Sig. Hybrid (k)":     shb_k,
        "Sig. CBCA (k)":       scb_k,
        # Percentages (numeric, 1 decimal)
        "Hybrid better (%)":   round(100.0 * hb_k  / n, 1),
        "CBCA better (%)":     round(100.0 * cb_k  / n, 1),
        "Significant (%)":     round(100.0 * sig_k / n, 1),
        "Sig. Hybrid (%)":     round(100.0 * shb_k / n, 1),
        "Sig. CBCA (%)":       round(100.0 * scb_k / n, 1),
    })

table2 = long.groupby(["metric", "class"]).apply(summarize).reset_index()
table2 = table2.merge(row_df, on=["metric", "class"], how="left")
table2 = table2.sort_values("_order").drop(columns="_order").reset_index(drop=True)

# Reorder columns into a readable block
table2 = table2[[
    "metric", "class", "n_models",
    "Hybrid better (k)", "Hybrid better (%)",
    "CBCA better (k)",   "CBCA better (%)",
    "Significant (k)",   "Significant (%)",
    "Sig. Hybrid (k)",   "Sig. Hybrid (%)",
    "Sig. CBCA (k)",     "Sig. CBCA (%)",
]]

print("\n" + "="*110)
print("TABLE 2 — CROSS-MODEL SUMMARY OF PAIRED COMPARISONS")
print("="*110)
print(table2.to_string(index=False))

# ============================================================
# TABLE S1 — Supplementary: absolute values and raw statistics
# ============================================================
supp_cols = [
    "model", "metric", "class", "n_samples",
    "cbca_value", "hybrid_value", "difference",
    "ci_lower_95", "ci_upper_95",
    "mcnemar_b", "mcnemar_c", "mcnemar_statistic",
    "mcnemar_test", "mcnemar_p_value",
    "stars", "sig_dir", "test_used",
]
for c in supp_cols:
    if c not in long.columns:
        long[c] = np.nan

supp = long[supp_cols].copy()
supp = supp.merge(row_df, on=["metric", "class"], how="left")
supp = supp.sort_values(["model", "_order"]).drop(columns="_order")

print("\n" + "="*110)
print("TABLE S1 — SUPPLEMENTARY: ABSOLUTE VALUES AND RAW STATISTICS (first 20 rows)")
print("="*110)
print(supp.head(20).to_string(index=False))

# ============================================================
# SAVE — CSV and Excel
# ============================================================
out1 = os.path.join(OUTPUT_FOLDER, "Table1_paired_results.csv")
out2 = os.path.join(OUTPUT_FOLDER, "Table2_summary.csv")
outs = os.path.join(OUTPUT_FOLDER, "TableS1_absolute_values.csv")

table1.to_csv(out1, index=False)
table2.to_csv(out2, index=False)
supp.to_csv(outs, index=False)

# ---- Excel output with explicit formatting ----
out_xlsx = os.path.join(OUTPUT_FOLDER, "All_Tables.xlsx")
wb = Workbook()

# --- Sheet 1: Table 1 ---
ws1 = wb.active
ws1.title = "Table1_Paired"
for r in dataframe_to_rows(table1, index=False, header=True):
    ws1.append(r)
for cell in ws1[1]:
    cell.font = Font(bold=True)

# --- Sheet 2: Table 2 (numbers only; force text on metric/class) ---
ws2 = wb.create_sheet("Table2_Summary")
for r in dataframe_to_rows(table2, index=False, header=True):
    ws2.append(r)
for cell in ws2[1]:
    cell.font = Font(bold=True)
# Force the metric/class columns to text so no auto-conversion happens
for row_idx in range(2, ws2.max_row + 1):
    ws2.cell(row=row_idx, column=1).number_format = "@"   # metric
    ws2.cell(row=row_idx, column=2).number_format = "@"   # class

# --- Sheet 3: Supplementary ---
ws3 = wb.create_sheet("TableS1_Absolute")
for r in dataframe_to_rows(supp, index=False, header=True):
    ws3.append(r)
for cell in ws3[1]:
    cell.font = Font(bold=True)

wb.save(out_xlsx)

print("\nSaved files:")
for p in [out1, out2, outs, out_xlsx]:
    print("  -", p)