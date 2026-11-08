
# ==============================================================
# 01_stat_analysis_fixed.py
# Script 1 — method-wise statistical comparison
# Fixes: zero-denominator (inf / nan) handling for Podo Recall vs BL
# ==============================================================

# ---------- STEP 1: Mount Google Drive (Colab only) ----------
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

# ==============================================================
# DATA INGESTION  (unchanged)
# ==============================================================
models  = ["ConvNeXt-Tiny", "Xception", "DenseNet121", "MNv3-Large", "MNv2", "EfficientViTB0"]
methods = ["BL", "DA", "BBFL", "CBCA", "Hybrid"]

accuracy = {
    "BL":     [0.718, 0.940, 0.960, 0.681, 0.943, 0.963],
    "DA":     [0.779, 0.94,  0.966, 0.799, 0.973, 0.97 ],
    "BBFL":   [0.664, 0.913, 0.946, 0.621, 0.903, 0.933],
    "CBCA":   [0.755, 0.953, 0.956, 0.725, 0.950, 0.960],
    "Hybrid": [0.708, 0.943, 0.963, 0.718, 0.950, 0.970],
}
macro_f1 = {
    "BL":     [0.430, 0.900, 0.907, 0.595, 0.878, 0.903],
    "DA":     [0.68,  0.9,   0.93,  0.68,  0.95,  0.926],
    "BBFL":   [0.493, 0.820, 0.870, 0.476, 0.817, 0.823],
    "CBCA":   [0.668, 0.906, 0.916, 0.666, 0.900, 0.928],
    "Hybrid": [0.569, 0.878, 0.925, 0.594, 0.900, 0.964],
}
podo_recall = {
    "BL":     [0.000, 0.769, 0.769, 0.308, 0.692, 0.692],
    "DA":     [0.620, 0.769, 0.850, 0.390, 0.850, 0.769],
    "BBFL":   [0.538, 1.000, 0.850, 0.231, 0.920, 0.769],
    "CBCA":   [0.538, 0.846, 0.923, 0.615, 0.769, 0.769],
    "Hybrid": [0.615, 0.692, 1.000, 0.539, 0.923, 0.923],
}
macro_recall = {
    "BL":     [0.431, 0.881, 0.898, 0.567, 0.864, 0.876],
    "DA":     [0.720, 0.880, 0.93,  0.650, 0.930, 0.897],
    "BBFL":   [0.552, 0.930, 0.920, 0.495, 0.910, 0.880],
    "CBCA":   [0.693, 0.915, 0.946, 0.705, 0.897, 0.897],
    "Hybrid": [0.612, 0.866, 0.975, 0.593, 0.937, 0.951],
}
METRICS = {
    "Accuracy":     accuracy,
    "Macro F1":     macro_f1,
    "Podo Recall":  podo_recall,
    "Macro Recall": macro_recall,
}

# ==============================================================
# COMPARISON HIERARCHY  (unchanged)
# ==============================================================
PRIMARY_PLANNED   = [("Hybrid", m) for m in methods if m != "Hybrid"]
SECONDARY_PLANNED = [("CBCA",   m) for m in ["BL", "DA", "BBFL"]]
ALL_PAIRS         = list(combinations(methods, 2))

# ==============================================================
# STATISTICAL HELPERS
# ==============================================================
def rank_biserial(a, b):
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    r_pos = ranks[diff > 0].sum()
    r_neg = ranks[diff < 0].sum()
    total = ranks.sum()
    return (r_pos - r_neg) / total if total > 0 else 0.0

def bootstrap_ci(a, b, n_boot=10000, ci=95):
    """Bootstrap CI for the mean PERCENTAGE difference (may break with zero denominators)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    pct = 100 * diff / np.asarray(b, dtype=float)
    pct = pct[~np.isnan(pct)]
    n = len(pct)
    if n < 2:
        return np.nan, np.nan, np.nan
    boot_means = []
    for _ in range(n_boot):
        resample = np.random.choice(pct, size=n, replace=True)
        boot_means.append(resample.mean())
    boot_means = np.sort(boot_means)
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return pct.mean(), lower, upper

# ---------- NEW HELPER ----------
def bootstrap_ci_abs(a, b, n_boot=10000, ci=95):
    """Bootstrap CI for the mean ABSOLUTE difference. Always defined — no division."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 2:
        return np.nan, np.nan, np.nan
    boot_means = np.array([np.mean(np.random.choice(diff, size=n, replace=True))
                           for _ in range(n_boot)])
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(diff.mean()), float(lower), float(upper)
# ---------- END NEW HELPER ----------

def permutation_test(a, b, n_perm=10000):
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 2:
        return np.nan
    observed_mean = np.mean(diff)
    count = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=n)
        perm_mean = np.mean(diff * signs)
        if abs(perm_mean) >= abs(observed_mean):
            count += 1
    return count / n_perm

def bayesian_sign_test(a, b):
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.5
    pos = np.sum(diff > 0)
    prob_superiority = 1 - beta.cdf(0.5, pos + 1, n - pos + 1)
    return prob_superiority

# ==============================================================
# PAIRWISE COMPARISON FUNCTION  (MODIFIED: zero-denominator handling)
# ==============================================================
def pairwise_table(metric_name, data_dict, pairs_list):
    df = pd.DataFrame(data_dict, index=models)
    rows = []
    for a, b in pairs_list:
        if a not in df.columns or b not in df.columns:
            continue
        sub = df[[a, b]].dropna()
        n = len(sub)
        row = {"Metric": metric_name, "Comparison": f"{a} vs {b}", "n": n}

        if n < 2:
            row.update({
                "Mean Δ (abs)": np.nan, "95% CI (abs)": "",
                "Bootstrap 95% CI (abs)": "",
                "Mean Δ%": np.nan, "n (%)": np.nan,
                "95% CI": "", "Bootstrap 95% CI": "",
                "Improved (k)": np.nan, "Worsened (k)": np.nan,
                "Rank-biserial r": np.nan,
                "Wilcoxon p": np.nan, "Permutation p": np.nan,
                "Bayesian Prob (A>B)": np.nan,
                "Min p (n)": 2 * (0.5 ** n) if n > 0 else np.nan,
                "Direction": "",
            })
            rows.append(row)
            continue

        diff     = sub[a] - sub[b]
        improved = int((diff > 0).sum())
        worsened = int((diff < 0).sum())

        # ---------- ABSOLUTE difference (always defined) ----------
        mean_abs = float(diff.mean())
        se_abs   = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        ci_abs   = stats.t.interval(0.95, df=n-1, loc=mean_abs, scale=se_abs)
        _, boot_abs_lo, boot_abs_hi = bootstrap_ci_abs(sub[a].values, sub[b].values)

        # ---------- PERCENTAGE difference (handles zero denominators) ----------
        zero_mask = (sub[b].values == 0)
        n_zero    = int(zero_mask.sum())

        if n_zero == 0:
            pct      = 100 * diff / sub[b]
            mean_pct = float(pct.mean())
            se_pct   = float(pct.std(ddof=1) / np.sqrt(n))
            ci       = stats.t.interval(0.95, df=n-1, loc=mean_pct, scale=se_pct)
            _, boot_low, boot_high = bootstrap_ci(sub[a].values, sub[b].values)
            n_pct    = n
            pct_ok   = True
        else:
            sub_pct = sub[~zero_mask]
            n_pct   = len(sub_pct)
            if n_pct >= 2:
                pct      = 100 * (sub_pct[a] - sub_pct[b]) / sub_pct[b]
                mean_pct = float(pct.mean())
                se_pct   = float(pct.std(ddof=1) / np.sqrt(n_pct))
                ci       = stats.t.interval(0.95, df=n_pct-1, loc=mean_pct, scale=se_pct)
                _, boot_low, boot_high = bootstrap_ci(sub_pct[a].values, sub_pct[b].values)
                pct_ok   = True
            else:
                mean_pct = np.nan
                ci       = (np.nan, np.nan)
                boot_low = boot_high = np.nan
                pct_ok   = False

        # ---------- Sign-based tests (use ALL pairs, unaffected by zeros) ----------
        try:
            w, p = stats.wilcoxon(sub[a], sub[b], zero_method="wilcox", method='exact')
        except ValueError:
            p = np.nan

        perm_p     = permutation_test(sub[a].values, sub[b].values)
        bayes_prob = bayesian_sign_test(sub[a].values, sub[b].values)
        r_rb       = rank_biserial(sub[a].values, sub[b].values)
        min_p      = 2 * (0.5 ** n)

        if improved > worsened:
            direction = f"{a} better"
        elif worsened > improved:
            direction = f"{b} better"
        else:
            direction = "tie"

        row.update({
            # ---- absolute columns (always defined) ----
            "Mean Δ (abs)":           round(mean_abs, 4),
            "95% CI (abs)":           f"[{ci_abs[0]:+.4f}, {ci_abs[1]:+.4f}]",
            "Bootstrap 95% CI (abs)": f"[{boot_abs_lo:+.4f}, {boot_abs_hi:+.4f}]",
            # ---- percentage columns (n (%) flags any exclusion) ----
            "Mean Δ%":           round(mean_pct, 2) if pct_ok else "N/A (zero baseline)",
            "n (%)":             n_pct,
            "95% CI":            f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if pct_ok else "—",
            "Bootstrap 95% CI":  f"[{boot_low:+.2f}, {boot_high:+.2f}]" if pct_ok else "—",
            # ---- unchanged statistical columns ----
            "Improved (k)":        improved,
            "Worsened (k)":        worsened,
            "Rank-biserial r":     round(r_rb, 3),
            "Wilcoxon p":          round(p, 4) if not np.isnan(p) else np.nan,
            "Permutation p":       round(perm_p, 4) if not np.isnan(perm_p) else np.nan,
            "Bayesian Prob (A>B)": round(bayes_prob, 3),
            "Min p (n)":           round(min_p, 4),
            "Direction":           direction,
        })
        rows.append(row)
    return pd.DataFrame(rows)

# ==============================================================
# BUILD TABLES
# ==============================================================
primary_tables, secondary_tables, full_tables = [], [], []
for metric_name, data_dict in METRICS.items():
    primary_tables.append(pairwise_table(metric_name, data_dict, PRIMARY_PLANNED))
    secondary_tables.append(pairwise_table(metric_name, data_dict, SECONDARY_PLANNED))
    full_tables.append(pairwise_table(metric_name, data_dict, ALL_PAIRS))

primary_results   = pd.concat(primary_tables,   ignore_index=True)
secondary_results = pd.concat(secondary_tables, ignore_index=True)
full_results      = pd.concat(full_tables,      ignore_index=True)

# ==============================================================
# OVERALL Δ% vs BL  (MODIFIED: also handles zero denominators)
# ==============================================================
overall_rows = []
for metric_name, data_dict in METRICS.items():
    df = pd.DataFrame(data_dict, index=models)
    if "BL" not in df.columns:
        continue
    ordered_methods = ["Hybrid", "CBCA"] + [m for m in methods if m not in ("Hybrid", "CBCA", "BL")]
    for m in ordered_methods:
        if m not in df.columns:
            continue
        sub = df[["BL", m]].dropna()
        if len(sub) < 2:
            continue

        diff = sub[m] - sub["BL"]

        # absolute
        mean_abs = float(diff.mean())
        se_abs   = float(diff.std(ddof=1) / np.sqrt(len(sub)))
        ci_abs   = stats.t.interval(0.95, df=len(sub)-1, loc=mean_abs, scale=se_abs)
        _, boot_abs_lo, boot_abs_hi = bootstrap_ci_abs(sub[m].values, sub["BL"].values)

        # percentage with zero-baseline handling
        zero_mask = (sub["BL"].values == 0)
        if zero_mask.any():
            sub_pct = sub[~zero_mask]
            n_pct   = len(sub_pct)
            if n_pct >= 2:
                pct      = 100 * (sub_pct[m] - sub_pct["BL"]) / sub_pct["BL"]
                mean_pct = float(pct.mean())
                se_pct   = float(pct.std(ddof=1) / np.sqrt(n_pct))
                ci       = stats.t.interval(0.95, df=n_pct-1, loc=mean_pct, scale=se_pct)
                _, boot_low, boot_high = bootstrap_ci(sub_pct[m].values, sub_pct["BL"].values)
                pct_ok   = True
            else:
                mean_pct = np.nan
                ci       = (np.nan, np.nan)
                boot_low = boot_high = np.nan
                pct_ok   = False
        else:
            n_pct    = len(sub)
            pct      = 100 * (sub[m] - sub["BL"]) / sub["BL"]
            mean_pct = float(pct.mean())
            se_pct   = float(pct.std(ddof=1) / np.sqrt(n_pct))
            ci       = stats.t.interval(0.95, df=n_pct-1, loc=mean_pct, scale=se_pct)
            _, boot_low, boot_high = bootstrap_ci(sub[m].values, sub["BL"].values)
            pct_ok   = True

        overall_rows.append({
            "Metric":                   metric_name,
            "Method":                   m,
            "Priority":                 "Primary" if m == "Hybrid" else ("Secondary" if m == "CBCA" else "Baseline"),
            "n":                        len(sub),
            "Mean Δ (abs)":             round(mean_abs, 4),
            "95% CI (abs)":             f"[{ci_abs[0]:+.4f}, {ci_abs[1]:+.4f}]",
            "Bootstrap 95% CI (abs)":   f"[{boot_abs_lo:+.4f}, {boot_abs_hi:+.4f}]",
            "n (%)":                    n_pct,
            "Mean Δ% vs BL":            round(mean_pct, 2) if pct_ok else "N/A (zero baseline)",
            "95% CI":                   f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if pct_ok else "—",
            "Bootstrap 95% CI":         f"[{boot_low:+.2f}, {boot_high:+.2f}]" if pct_ok else "—",
            "Improved":                 int((sub[m] > sub["BL"]).sum()),
            "Worsened":                 int((sub[m] < sub["BL"]).sum()),
        })

overall_df = pd.DataFrame(overall_rows)

# ==============================================================
# PRINT
# ==============================================================
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 40)

def print_block(title, df):
    print("\n" + "=" * 180)
    print(title)
    print("=" * 180)
    print(df.to_string(index=False) if not df.empty else "(empty)")

print("\n\n" + "#" * 180)
print("# PRIMARY ANALYSIS — HYBRID (ultimate strategy) vs all baselines")
print("#" * 180)
for metric_name in METRICS:
    sub = primary_results[primary_results["Metric"] == metric_name]
    print_block(f"HYBRID (primary) — {metric_name}", sub.drop(columns="Metric"))

print("\n\n" + "#" * 180)
print("# OVERALL MEAN Δ vs BL (Hybrid first, CBCA second)")
print("# Absolute difference is always defined; percentage is N/A where BL = 0.")
print("#" * 180)
print(overall_df.to_string(index=False) if not overall_df.empty else "(no data)")

# ==============================================================
# SAVING
# ==============================================================
OUTPUT_DIR    = r"D:\our\path \to\the \Final Stat Analyses"
OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, "Overall Method-Wise")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

primary_results.to_csv(os.path.join(OUTPUT_FOLDER, "hybrid_primary.csv"),   index=False)
secondary_results.to_csv(os.path.join(OUTPUT_FOLDER, "cbca_secondary.csv"), index=False)
full_results.to_csv(os.path.join(OUTPUT_FOLDER, "full_pairwise.csv"),       index=False)
overall_df.to_csv(os.path.join(OUTPUT_FOLDER, "overall_vs_BL.csv"),         index=False)

print("\nSaved:")
print("  - hybrid_primary.csv    (PRIMARY: Hybrid vs all)")
print("  - cbca_secondary.csv    (SECONDARY: CBCA vs baselines)")
print("  - full_pairwise.csv     (SUPPLEMENTARY: all pairs)")
print("  - overall_vs_BL.csv     (overall Δ vs BL, Hybrid first)")
print(f"\nOutput folder: {OUTPUT_FOLDER}")