
# --- 1. SETUP AND IMPORTS ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2, binomtest
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# --- 2. REPRODUCIBILITY + GLOBAL CONFIG ---
MASTER_SEED = 42
R_BOOTSTRAP = 1000          # set to 100 for a fast smoke test
ALPHA       = 0.05
CLASS_NAMES = ['podo', 'scabies', 'tungiasis']
NUM_CLASSES = len(CLASS_NAMES)

# --- 3. PATHS ---
BASE_DIR    = r"D:\our\path\to a root\directory\Results & Analysis"
CBCA_DIR    = os.path.join(BASE_DIR, '02-Test Predictions', '1-CBCA')
HYBRID_DIR  = os.path.join(BASE_DIR, '02-Test Predictions', '2-WCBCA')
OUTPUT_ROOT = os.path.join(BASE_DIR, 'Analysis-by-Model')

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
def mcnemar_test(y_true, y_pred_a, y_pred_b, subset_mask=None):
    if subset_mask is not None:
        y_true   = y_true[subset_mask]
        y_pred_a = y_pred_a[subset_mask]
        y_pred_b = y_pred_b[subset_mask]
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_subset = len(y_true)
    acc_a = float(np.mean(correct_a)) if n_subset > 0 else float('nan')
    acc_b = float(np.mean(correct_b)) if n_subset > 0 else float('nan')
    if (b + c) < 25:
        p_value = 1.0 if (b + c) == 0 else binomtest(b, n=b + c, p=0.5).pvalue
        statistic, test_used = None, 'exact_binomial'
    else:
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)
        test_used = 'chi_squared'
    return {'b': b, 'c': c, 'statistic': statistic, 'p_value': float(p_value),
            'test_used': test_used, 'n_subset': n_subset, 'acc_a': acc_a, 'acc_b': acc_b}


def compute_metric(y_true, y_pred, metric_name, class_idx=None):
    if len(y_true) == 0:
        return float('nan')
    if metric_name == 'accuracy':
        return float(accuracy_score(y_true, y_pred))
    if metric_name == 'macro_recall':
        return float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    if metric_name == 'macro_f1':
        return float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    if metric_name == 'recall':
        return float(recall_score(y_true, y_pred, average=None,
                                  labels=range(NUM_CLASSES), zero_division=0)[class_idx])
    if metric_name == 'precision':
        return float(precision_score(y_true, y_pred, average=None,
                                     labels=range(NUM_CLASSES), zero_division=0)[class_idx])
    if metric_name == 'f1':
        return float(f1_score(y_true, y_pred, average=None,
                              labels=range(NUM_CLASSES), zero_division=0)[class_idx])
    if metric_name == 'specificity':
        cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
        tp = cm[class_idx, class_idx]
        fn = cm[class_idx, :].sum() - tp
        fp = cm[:, class_idx].sum() - tp
        tn = cm.sum() - tp - fn - fp
        return float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')
    raise ValueError(f"Unknown metric_name: {metric_name}")


def paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, metric_name,
                        class_idx=None, R=R_BOOTSTRAP, alpha=ALPHA, seed=MASTER_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    value_a = compute_metric(y_true, y_pred_a, metric_name, class_idx)
    value_b = compute_metric(y_true, y_pred_b, metric_name, class_idx)
    difference = value_b - value_a
    boot_diffs = np.empty(R, dtype=float)
    for r in range(R):
        idx = rng.integers(0, n, size=n)
        ma = compute_metric(y_true[idx], y_pred_a[idx], metric_name, class_idx)
        mb = compute_metric(y_true[idx], y_pred_b[idx], metric_name, class_idx)
        boot_diffs[r] = mb - ma
    lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return {'value_a': float(value_a), 'value_b': float(value_b),
            'difference': float(difference), 'ci_lower': lower, 'ci_upper': upper,
            'boot_diffs': boot_diffs}


def save_confusion_matrices(y_true, y_pred, method_name, out_dir):
    cm_raw = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Test Set Confusion Matrix (Raw Counts) — {method_name}')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    raw_path = os.path.join(out_dir, 'Plots', f'CM_{method_name}_Raw.png')
    plt.savefig(raw_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved raw CM: {raw_path}")

    cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Test Set Confusion Matrix (Row-Normalized) — {method_name}')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    norm_path = os.path.join(out_dir, 'Plots', f'CM_{method_name}_Normalized.png')
    plt.savefig(norm_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved normalized CM: {norm_path}")
    return cm_raw


# ======================================================================
# ONE-MODEL ANALYSIS
# ======================================================================
def run_analysis(cbca_path, hybrid_path, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Plots'), exist_ok=True)

    print("\n" + "#" * 78)
    print(f"#  MODEL: {model_name}")
    print("#" * 78)
    print(f"  CBCA   : {cbca_path}")
    print(f"  Hybrid : {hybrid_path}")
    print(f"  Output : {output_dir}")

    # --- LOAD PREDICTIONS ---
    cbca_data   = np.load(cbca_path)
    hybrid_data = np.load(hybrid_path)
    y_true_cbca   = cbca_data['y_true'];   y_pred_cbca   = cbca_data['y_pred']
    y_true_hybrid = hybrid_data['y_true']; y_pred_hybrid = hybrid_data['y_pred']
    assert np.array_equal(y_true_cbca, y_true_hybrid), \
        f"Test set labels do not match for {model_name} — cannot do a paired analysis."

    y_true = y_true_cbca
    n = len(y_true)
    print(f"\nLoaded paired predictions for {n} test samples.")
    print("Class distribution in the test set:")
    for i, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y_true == i))
        print(f"  {name}: {count} samples ({count / n:.1%})")

    # --- OVERALL ACCURACY ---
    print("\n" + "=" * 70 + "\nSECTION 1 — OVERALL ACCURACY\n" + "=" * 70)
    mcnemar_overall = mcnemar_test(y_true, y_pred_cbca, y_pred_hybrid)
    boot_overall    = paired_bootstrap_ci(y_true, y_pred_cbca, y_pred_hybrid, 'accuracy')
    print(f"\nTest samples: {n}")
    print(f"CBCA accuracy  : {boot_overall['value_a']:.4f}")
    print(f"Hybrid accuracy: {boot_overall['value_b']:.4f}")
    print(f"Difference (Hybrid - CBCA): {boot_overall['difference']:+.4f}")
    print("\nMcNemar discordant pairs:")
    print(f"  b = {mcnemar_overall['b']}  (CBCA correct, Hybrid wrong)")
    print(f"  c = {mcnemar_overall['c']}  (CBCA wrong, Hybrid correct)")
    if mcnemar_overall['statistic'] is not None:
        print(f"  McNemar χ² = {mcnemar_overall['statistic']:.4f}")
    print(f"  Test used: {mcnemar_overall['test_used']}")
    print(f"  p-value  : {mcnemar_overall['p_value']:.6f}")
    print(f"\n95% Bootstrap CI on the difference: "
          f"[{boot_overall['ci_lower']:+.4f}, {boot_overall['ci_upper']:+.4f}]")

    # --- PER-CLASS RECALL ---
    print("\n" + "=" * 70 + "\nSECTION 2 — PER-CLASS RECALL (SENSITIVITY)\n" + "=" * 70)
    per_class_recall_results = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        print(f"\n--- {class_name.upper()} ---")
        mask = (y_true == class_idx); n_class = int(np.sum(mask))
        print(f"Number of {class_name} samples in test set: {n_class}")
        mcnemar_class = mcnemar_test(y_true, y_pred_cbca, y_pred_hybrid, subset_mask=mask)
        boot_class    = paired_bootstrap_ci(y_true, y_pred_cbca, y_pred_hybrid,
                                            'recall', class_idx=class_idx)
        print(f"CBCA recall  : {boot_class['value_a']:.4f}  "
              f"({int(round(boot_class['value_a'] * n_class))}/{n_class})")
        print(f"Hybrid recall: {boot_class['value_b']:.4f}  "
              f"({int(round(boot_class['value_b'] * n_class))}/{n_class})")
        print(f"Difference   : {boot_class['difference']:+.4f}")
        print(f"McNemar b = {mcnemar_class['b']}, c = {mcnemar_class['c']}, "
              f"test = {mcnemar_class['test_used']}, p = {mcnemar_class['p_value']:.6f}")
        print(f"95% Bootstrap CI: [{boot_class['ci_lower']:+.4f}, {boot_class['ci_upper']:+.4f}]")
        per_class_recall_results[class_name] = {
            'mcnemar': mcnemar_class, 'bootstrap': boot_class, 'n_class': n_class,
        }

    # --- PER-CLASS PRECISION ---
    print("\n" + "=" * 70 + "\nSECTION 3 — PER-CLASS PRECISION\n" + "=" * 70)
    per_class_precision_results = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        boot = paired_bootstrap_ci(y_true, y_pred_cbca, y_pred_hybrid,
                                   'precision', class_idx=class_idx)
        per_class_precision_results[class_name] = boot
        print(f"\n{class_name}: CBCA={boot['value_a']:.4f}, Hybrid={boot['value_b']:.4f}, "
              f"diff={boot['difference']:+.4f}, "
              f"95% CI=[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]")

    # --- PER-CLASS SPECIFICITY ---
    print("\n" + "=" * 70 + "\nSECTION 4 — PER-CLASS SPECIFICITY\n" + "=" * 70)
    per_class_specificity_results = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        boot = paired_bootstrap_ci(y_true, y_pred_cbca, y_pred_hybrid,
                                   'specificity', class_idx=class_idx)
        per_class_specificity_results[class_name] = boot
        print(f"\n{class_name}: CBCA={boot['value_a']:.4f}, Hybrid={boot['value_b']:.4f}, "
              f"diff={boot['difference']:+.4f}, "
              f"95% CI=[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]")

    # --- PER-CLASS F1 ---
    print("\n" + "=" * 70 + "\nSECTION 5 — PER-CLASS F1-SCORE\n" + "=" * 70)
    per_class_f1_results = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        boot = paired_bootstrap_ci(y_true, y_pred_cbca, y_pred_hybrid,
                                   'f1', class_idx=class_idx)
        per_class_f1_results[class_name] = boot
        print(f"\n{class_name}: CBCA={boot['value_a']:.4f}, Hybrid={boot['value_b']:.4f}, "
              f"diff={boot['difference']:+.4f}, "
              f"95% CI=[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]")

    # --- MACRO METRICS ---
    print("\n" + "=" * 70 + "\nSECTION 6 — MACRO-AVERAGED RECALL AND F1\n" + "=" * 70)
    boot_macro_recall = paired_bootstrap_ci(y_true, y_pred_cbca, y_pred_hybrid, 'macro_recall')
    boot_macro_f1     = paired_bootstrap_ci(y_true, y_pred_cbca, y_pred_hybrid, 'macro_f1')
    print(f"\nMacro Recall:  CBCA={boot_macro_recall['value_a']:.4f}, "
          f"Hybrid={boot_macro_recall['value_b']:.4f}, "
          f"diff={boot_macro_recall['difference']:+.4f}, "
          f"95% CI=[{boot_macro_recall['ci_lower']:+.4f}, {boot_macro_recall['ci_upper']:+.4f}]")
    print(f"Macro F1:      CBCA={boot_macro_f1['value_a']:.4f}, "
          f"Hybrid={boot_macro_f1['value_b']:.4f}, "
          f"diff={boot_macro_f1['difference']:+.4f}, "
          f"95% CI=[{boot_macro_f1['ci_lower']:+.4f}, {boot_macro_f1['ci_upper']:+.4f}]")

    # --- CONFUSION MATRICES ---
    print("\n" + "=" * 70 + "\nSECTION 7 — CONFUSION MATRICES\n" + "=" * 70)
    cm_cbca   = save_confusion_matrices(y_true, y_pred_cbca,   'CBCA',   output_dir)
    cm_hybrid = save_confusion_matrices(y_true, y_pred_hybrid, 'Hybrid', output_dir)
    print(f"\nCBCA raw CM:\n{cm_cbca}")
    print(f"\nHybrid raw CM:\n{cm_hybrid}")

    # --- BOOTSTRAP DISTRIBUTION PLOTS ---
    print("\n" + "=" * 70 + "\nSECTION 8 — BOOTSTRAP DISTRIBUTION PLOTS\n" + "=" * 70)
    comparisons = [
        ('Overall Accuracy', boot_overall),
        ('Podo Recall',      per_class_recall_results['podo']['bootstrap']),
        ('Scabies Recall',   per_class_recall_results['scabies']['bootstrap']),
        ('Tungiasis Recall', per_class_recall_results['tungiasis']['bootstrap']),
        ('Macro Recall',     boot_macro_recall),
        ('Macro F1',         boot_macro_f1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flatten()
    for ax, (title, result) in zip(axes, comparisons):
        ax.hist(result['boot_diffs'], bins=40, color='steelblue', edgecolor='black', alpha=0.75)
        ax.axvline(0.0, color='red', linestyle='--', linewidth=1.5, label='No difference')
        ax.axvline(result['difference'], color='green', linestyle='-', linewidth=1.5,
                   label='Observed diff')
        ax.axvline(result['ci_lower'], color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(result['ci_upper'], color='orange', linestyle=':', linewidth=1.5,
                   label='95% CI bounds')
        ax.set_title(f"{title}\nObserved diff = {result['difference']:+.4f}  "
                     f"[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]")
        ax.set_xlabel('Bootstrap difference (Hybrid - CBCA)')
        ax.set_ylabel('Frequency'); ax.legend(fontsize=8)
    plt.tight_layout()
    boot_plot_path = os.path.join(output_dir, 'Plots', 'Bootstrap_Distributions.png')
    plt.savefig(boot_plot_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"\nBootstrap distribution plot saved to: {boot_plot_path}")

    # --- SAVE TEXT REPORT ---
    print("\n" + "=" * 70 + "\nSECTION 9 — SAVING RESULTS TO TEXT FILE\n" + "=" * 70)
    txt_path = os.path.join(output_dir, 'paired_statistics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 78 + "\n")
        f.write(f"EXTENDED PAIRED STATISTICAL ANALYSIS — {model_name}\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Master seed: {MASTER_SEED}\nBootstrap resamples: {R_BOOTSTRAP}\n"
                f"Confidence level: {100 * (1 - ALPHA):.1f}%\nTotal test samples: {n}\n\n")
        f.write("-" * 78 + "\nTEST SET COMPOSITION\n" + "-" * 78 + "\n")
        for i, name in enumerate(CLASS_NAMES):
            cnt = int(np.sum(y_true == i))
            f.write(f"  {name}: {cnt} samples ({cnt / n:.1%})\n")
        f.write("\n")
        f.write("-" * 78 + "\nOVERALL ACCURACY\n" + "-" * 78 + "\n")
        f.write(f"  CBCA accuracy  : {boot_overall['value_a']:.6f}\n")
        f.write(f"  Hybrid accuracy: {boot_overall['value_b']:.6f}\n")
        f.write(f"  Difference     : {boot_overall['difference']:+.6f}\n")
        f.write(f"  McNemar b      : {mcnemar_overall['b']}\n")
        f.write(f"  McNemar c      : {mcnemar_overall['c']}\n")
        if mcnemar_overall['statistic'] is not None:
            f.write(f"  McNemar χ²     : {mcnemar_overall['statistic']:.6f}\n")
        f.write(f"  McNemar test   : {mcnemar_overall['test_used']}\n")
        f.write(f"  McNemar p-value: {mcnemar_overall['p_value']:.6f}\n")
        f.write(f"  95% Bootstrap CI: [{boot_overall['ci_lower']:+.6f}, "
                f"{boot_overall['ci_upper']:+.6f}]\n\n")
        f.write("-" * 78 + "\nPER-CLASS RECALL (SENSITIVITY)\n" + "-" * 78 + "\n")
        for name in CLASS_NAMES:
            r = per_class_recall_results[name]
            m, b, nc = r['mcnemar'], r['bootstrap'], r['n_class']
            f.write(f"  [{name}]  (n = {nc})\n")
            f.write(f"    CBCA recall  : {b['value_a']:.6f}  "
                    f"({int(round(b['value_a'] * nc))}/{nc})\n")
            f.write(f"    Hybrid recall: {b['value_b']:.6f}  "
                    f"({int(round(b['value_b'] * nc))}/{nc})\n")
            f.write(f"    Difference   : {b['difference']:+.6f}\n")
            f.write(f"    McNemar b={m['b']}, c={m['c']}, "
                    f"test={m['test_used']}, p={m['p_value']:.6f}\n")
            f.write(f"    95% Bootstrap CI: [{b['ci_lower']:+.6f}, "
                    f"{b['ci_upper']:+.6f}]\n\n")
        f.write("-" * 78 + "\nPER-CLASS PRECISION\n" + "-" * 78 + "\n")
        for name in CLASS_NAMES:
            b = per_class_precision_results[name]
            f.write(f"  [{name}]\n    CBCA precision  : {b['value_a']:.6f}\n"
                    f"    Hybrid precision: {b['value_b']:.6f}\n"
                    f"    Difference      : {b['difference']:+.6f}\n"
                    f"    95% Bootstrap CI: [{b['ci_lower']:+.6f}, "
                    f"{b['ci_upper']:+.6f}]\n\n")
        f.write("-" * 78 + "\nPER-CLASS SPECIFICITY\n" + "-" * 78 + "\n")
        for name in CLASS_NAMES:
            b = per_class_specificity_results[name]
            f.write(f"  [{name}]\n    CBCA specificity  : {b['value_a']:.6f}\n"
                    f"    Hybrid specificity: {b['value_b']:.6f}\n"
                    f"    Difference        : {b['difference']:+.6f}\n"
                    f"    95% Bootstrap CI  : [{b['ci_lower']:+.6f}, "
                    f"{b['ci_upper']:+.6f}]\n\n")
        f.write("-" * 78 + "\nPER-CLASS F1-SCORE\n" + "-" * 78 + "\n")
        for name in CLASS_NAMES:
            b = per_class_f1_results[name]
            f.write(f"  [{name}]\n    CBCA F1  : {b['value_a']:.6f}\n"
                    f"    Hybrid F1: {b['value_b']:.6f}\n"
                    f"    Difference: {b['difference']:+.6f}\n"
                    f"    95% Bootstrap CI: [{b['ci_lower']:+.6f}, "
                    f"{b['ci_upper']:+.6f}]\n\n")
        f.write("-" * 78 + "\nMACRO-AVERAGED METRICS\n" + "-" * 78 + "\n")
        f.write("  Macro Recall:\n"
                f"    CBCA   : {boot_macro_recall['value_a']:.6f}\n"
                f"    Hybrid : {boot_macro_recall['value_b']:.6f}\n"
                f"    Diff   : {boot_macro_recall['difference']:+.6f}\n"
                f"    95% CI : [{boot_macro_recall['ci_lower']:+.6f}, "
                f"{boot_macro_recall['ci_upper']:+.6f}]\n\n")
        f.write("  Macro F1:\n"
                f"    CBCA   : {boot_macro_f1['value_a']:.6f}\n"
                f"    Hybrid : {boot_macro_f1['value_b']:.6f}\n"
                f"    Diff   : {boot_macro_f1['difference']:+.6f}\n"
                f"    95% CI : [{boot_macro_f1['ci_lower']:+.6f}, "
                f"{boot_macro_f1['ci_upper']:+.6f}]\n\n")
        f.write("-" * 78 + "\nCONFUSION MATRICES (RAW COUNTS)\n" + "-" * 78 + "\n")
        f.write(f"CBCA raw CM:\n{cm_cbca}\n\nHybrid raw CM:\n{cm_hybrid}\n\n")
        f.write("=" * 78 + "\nEND OF REPORT\n" + "=" * 78 + "\n")
    print(f"Text report saved to: {txt_path}")

    # --- SAVE CSV ---
    print("\n" + "=" * 70 + "\nSECTION 10 — SAVING RESULTS TO CSV\n" + "=" * 70)
    csv_path = os.path.join(output_dir, 'statistical_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'class', 'n_samples',
                         'cbca_value', 'hybrid_value', 'difference',
                         'ci_lower_95', 'ci_upper_95',
                         'mcnemar_b', 'mcnemar_c', 'mcnemar_statistic',
                         'mcnemar_test', 'mcnemar_p_value'])
        writer.writerow(['accuracy', 'overall', n,
                         f"{boot_overall['value_a']:.6f}", f"{boot_overall['value_b']:.6f}",
                         f"{boot_overall['difference']:+.6f}",
                         f"{boot_overall['ci_lower']:+.6f}", f"{boot_overall['ci_upper']:+.6f}",
                         mcnemar_overall['b'], mcnemar_overall['c'],
                         f"{mcnemar_overall['statistic']:.6f}"
                         if mcnemar_overall['statistic'] is not None else '',
                         mcnemar_overall['test_used'],
                         f"{mcnemar_overall['p_value']:.6f}"])
        for name in CLASS_NAMES:
            r = per_class_recall_results[name]; m = r['mcnemar']; b = r['bootstrap']
            writer.writerow(['recall', name, r['n_class'],
                             f"{b['value_a']:.6f}", f"{b['value_b']:.6f}",
                             f"{b['difference']:+.6f}",
                             f"{b['ci_lower']:+.6f}", f"{b['ci_upper']:+.6f}",
                             m['b'], m['c'],
                             f"{m['statistic']:.6f}" if m['statistic'] is not None else '',
                             m['test_used'], f"{m['p_value']:.6f}"])
        for name in CLASS_NAMES:
            b = per_class_precision_results[name]
            writer.writerow(['precision', name,
                             int(np.sum(y_true == CLASS_NAMES.index(name))),
                             f"{b['value_a']:.6f}", f"{b['value_b']:.6f}",
                             f"{b['difference']:+.6f}",
                             f"{b['ci_lower']:+.6f}", f"{b['ci_upper']:+.6f}",
                             '', '', '', '', ''])
        for name in CLASS_NAMES:
            b = per_class_specificity_results[name]
            writer.writerow(['specificity', name, n,
                             f"{b['value_a']:.6f}", f"{b['value_b']:.6f}",
                             f"{b['difference']:+.6f}",
                             f"{b['ci_lower']:+.6f}", f"{b['ci_upper']:+.6f}",
                             '', '', '', '', ''])
        for name in CLASS_NAMES:
            b = per_class_f1_results[name]
            writer.writerow(['f1', name,
                             int(np.sum(y_true == CLASS_NAMES.index(name))),
                             f"{b['value_a']:.6f}", f"{b['value_b']:.6f}",
                             f"{b['difference']:+.6f}",
                             f"{b['ci_lower']:+.6f}", f"{b['ci_upper']:+.6f}",
                             '', '', '', '', ''])
        writer.writerow(['macro_recall', 'macro', n,
                         f"{boot_macro_recall['value_a']:.6f}",
                         f"{boot_macro_recall['value_b']:.6f}",
                         f"{boot_macro_recall['difference']:+.6f}",
                         f"{boot_macro_recall['ci_lower']:+.6f}",
                         f"{boot_macro_recall['ci_upper']:+.6f}",
                         '', '', '', '', ''])
        writer.writerow(['macro_f1', 'macro', n,
                         f"{boot_macro_f1['value_a']:.6f}",
                         f"{boot_macro_f1['value_b']:.6f}",
                         f"{boot_macro_f1['difference']:+.6f}",
                         f"{boot_macro_f1['ci_lower']:+.6f}",
                         f"{boot_macro_f1['ci_upper']:+.6f}",
                         '', '', '', '', ''])
    print(f"CSV summary saved to: {csv_path}")

    print(f"\n{model_name}: DONE\n")


# ======================================================================
# DISCOVERY  (recursive scan, exact + fuzzy matching)
# ======================================================================
def _collect_npz_recursive(root):
    """Return {basename: full_path} for all .npz files under root (recursive)."""
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith('.npz'):
                full = os.path.join(dp, f)
                if f in out:
                    print(f"  ⚠️  Duplicate basename '{f}':")
                    print(f"       keeping   {out[f]}")
                    print(f"       skipping  {full}")
                else:
                    out[f] = full
    return out


def _norm_key(name):
    """Normalize a filename for fuzzy matching (handles space vs underscore,
    case, missing extension, etc.)."""
    s = name
    if s.lower().endswith('.npz'):
        s = s[:-4]
    s = s.replace('test_predictions', '').replace('Test_Predictions', '')
    for ch in (' ', '_', '-'):
        s = s.replace(ch, '')
    return s.lower()


def discover_pairs(cbca_dir, hybrid_dir):
    """Return [(cbca_path, hybrid_path, display_name), ...] for every matched model."""
    if not os.path.isdir(cbca_dir):
        raise FileNotFoundError(f"CBCA folder not found:   {cbca_dir}")
    if not os.path.isdir(hybrid_dir):
        raise FileNotFoundError(f"Hybrid folder not found: {hybrid_dir}")

    cbca_map   = _collect_npz_recursive(cbca_dir)
    hybrid_map = _collect_npz_recursive(hybrid_dir)

    print(f"\n  CBCA   folder: {len(cbca_map)} .npz file(s)")
    for f in sorted(cbca_map):   print(f"    - {f}")
    print(f"  Hybrid folder: {len(hybrid_map)} .npz file(s)")
    for f in sorted(hybrid_map): print(f"    - {f}")

    pairs = []
    used_hybrid = set()

    # Pass 1: exact basename
    for fname in sorted(set(cbca_map) & set(hybrid_map)):
        pairs.append((cbca_map[fname], hybrid_map[fname], fname))
        used_hybrid.add(fname)

    # Pass 2: fuzzy match leftovers
    leftover_cbca   = sorted(set(cbca_map)   - used_hybrid)
    leftover_hybrid = sorted(set(hybrid_map) - used_hybrid)
    if leftover_cbca and leftover_hybrid:
        print("\n  🔎 Trying fuzzy match on leftovers:")
        print(f"     CBCA leftovers:   {leftover_cbca}")
        print(f"     Hybrid leftovers: {leftover_hybrid}")
        by_key = {_norm_key(h): h for h in leftover_hybrid}
        for c_name in leftover_cbca:
            key = _norm_key(c_name)
            if key in by_key:
                h_name = by_key[key]
                print(f"     ✅ fuzzy pair: '{c_name}' ↔ '{h_name}'")
                pairs.append((cbca_map[c_name], hybrid_map[h_name],
                              f"{c_name} (fuzzy↔{h_name})"))
                used_hybrid.add(h_name)
                del by_key[key]

    # Report unmatched
    matched_cbca = {p[2].split(' (fuzzy')[0] for p in pairs}
    still_cbca   = sorted(set(cbca_map)   - matched_cbca)
    still_hybrid = sorted(set(hybrid_map) - used_hybrid)
    if still_cbca:
        print("\n  ⚠️  Unmatched in CBCA folder:")
        for f in still_cbca: print(f"     - {f}")
    if still_hybrid:
        print("\n  ⚠️  Unmatched in Hybrid folder:")
        for f in still_hybrid: print(f"     - {f}")

    return pairs


def model_name_from_filename(fname):
    """'3-DenseNet test_predictions.npz' -> '3-DenseNet'"""
    base = fname
    if base.lower().endswith('.npz'):
        base = base[:-4]
    base = base.replace('test_predictions', '').replace('Test_Predictions', '')
    base = base.split(' (fuzzy')[0]
    base = base.strip(' -_')
    return base if base else 'model'


# ======================================================================
# DRIVER
# ======================================================================
if __name__ == '__main__':
    print("=" * 78)
    print("MULTI-MODEL ANALYSIS")
    print("=" * 78)
    print(f"CBCA dir   : {CBCA_DIR}")
    print(f"Hybrid dir : {HYBRID_DIR}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Bootstrap  : {R_BOOTSTRAP} resamples per metric per model")

    pairs = discover_pairs(CBCA_DIR, HYBRID_DIR)

    print("\n" + "=" * 100)
    print(f"PAIRING TABLE — {len(pairs)} model pair(s) will be analysed")
    print("=" * 100)
    print(f"{'#':<4}{'Model':<22}{'CBCA file':<40}{'Hybrid file':<40}")
    print("-" * 100)
    for i, (cp, hp, fn) in enumerate(pairs, 1):
        mn = model_name_from_filename(fn)
        print(f"{i:<4}{mn:<22}{os.path.basename(cp):<40}{os.path.basename(hp):<40}")
    print("=" * 100)

    if len(pairs) == 0:
        raise SystemExit(
            "\n❌ No matched .npz pairs found.\n"
            f"   CBCA folder:   {CBCA_DIR}\n"
            f"   Hybrid folder: {HYBRID_DIR}\n"
        )
    if len(pairs) != 6:
        print(f"\n⚠️  Expected 6 models, found {len(pairs)}. Continuing with what's available.")

    combined_dfs      = []
    per_model_summary = []
    for i, (cbca_path, hybrid_path, fname) in enumerate(pairs, 1):
        mn  = model_name_from_filename(fname)
        out = os.path.join(OUTPUT_ROOT, f"{mn}-based model")
        print(f"\n\n{'='*78}\n[{i}/{len(pairs)}]  {mn}\n{'='*78}")
        try:
            run_analysis(cbca_path, hybrid_path, out, mn)
            df = pd.read_csv(os.path.join(out, 'statistical_summary.csv'))
            df.insert(0, 'model', mn)
            combined_dfs.append(df)
            acc_row = df[(df['metric'] == 'accuracy') & (df['class'] == 'overall')].iloc[0]
            per_model_summary.append({
                'model':           mn,
                'n_samples':       int(acc_row['n_samples']),
                'cbca_accuracy':   float(acc_row['cbca_value']),
                'hybrid_accuracy': float(acc_row['hybrid_value']),
                'accuracy_diff':   float(acc_row['difference']),
                'mcnemar_p':       float(acc_row['mcnemar_p_value'])
                                   if acc_row['mcnemar_p_value'] != '' else float('nan'),
                'better':          'Hybrid' if acc_row['difference'] > 0 else 'CBCA',
            })
        except Exception as e:
            print(f"❌ {mn} FAILED: {type(e).__name__}: {e}")
            continue

    # --- COMBINED OUTPUTS ---
    print("\n\n" + "#" * 78)
    print("#  COMBINED SUMMARY ACROSS ALL MODELS")
    print("#" * 78)

    if combined_dfs:
        combined = pd.concat(combined_dfs, ignore_index=True)
        combined_path = os.path.join(OUTPUT_ROOT, 'ALL_MODELS_combined_statistics.csv')
        combined.to_csv(combined_path, index=False)
        print(f"\n✅ Combined CSV: {combined_path}")
        print(f"   ({len(combined)} rows, {len(combined_dfs)} models)")

    if per_model_summary:
        summary_df   = pd.DataFrame(per_model_summary)
        summary_path = os.path.join(OUTPUT_ROOT, 'ALL_MODELS_accuracy_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Per-model accuracy summary: {summary_path}")

        print("\n" + "-" * 100)
        print("ACCURACY: CBCA vs Hybrid, per model")
        print("-" * 100)
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        hybrid_wins = int((summary_df['accuracy_diff'] > 0).sum())
        print(f"\nHybrid beat CBCA on accuracy in {hybrid_wins}/{len(summary_df)} models.")

        # --- Grouped bar chart ---
        models   = summary_df['model'].tolist()
        cbca_acc = summary_df['cbca_accuracy'].tolist()
        hyb_acc  = summary_df['hybrid_accuracy'].tolist()
        x = np.arange(len(models)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(models)), 5))
        b1 = ax.bar(x - w/2, cbca_acc, w, label='CBCA',   color='steelblue')
        b2 = ax.bar(x + w/2, hyb_acc,  w, label='Hybrid', color='darkorange')
        for bars, vals in [(b1, cbca_acc), (b2, hyb_acc)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        f'{v:.3f}', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right')
        ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.05)
        ax.set_title('CBCA vs Hybrid — accuracy per model')
        ax.legend(); plt.tight_layout()
        plot_path = os.path.join(OUTPUT_ROOT, 'ALL_MODELS_accuracy_comparison.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight'); plt.close()
        print(f"\n✅ Accuracy comparison plot: {plot_path}")

    print("\n" + "=" * 78)
    print("ALL MODELS COMPLETE")
    print("=" * 78)
    print(f"Output root: {OUTPUT_ROOT}")
    print("Per-model folders:   <model-name>-based model/")
    print("Combined files:      ALL_MODELS_combined_statistics.csv")
    print("                     ALL_MODELS_accuracy_summary.csv")
    print("                     ALL_MODELS_accuracy_comparison.png")