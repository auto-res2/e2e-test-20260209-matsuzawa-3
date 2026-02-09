import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy.stats import ttest_ind


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_learning_curve(history, metric, out_path, run_id):
    if metric in history:
        plt.figure()
        plt.plot(history[metric].values)
        plt.title(f"{run_id} {metric}")
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def plot_confusion(correct_flags, out_path, run_id):
    correct = int(np.sum(correct_flags))
    incorrect = int(len(correct_flags) - correct)
    cm = np.array([[incorrect, 0], [0, correct]])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{run_id} Correct/Incorrect")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bar(metrics, out_path, title, ylabel):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title(title)
    plt.ylabel(ylabel)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_boxplot(values_by_run, out_path, title, ylabel):
    rows = []
    for run_id, values in values_by_run.items():
        for v in values:
            rows.append({"run_id": run_id, "value": v})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="run_id", y="value")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_violin(values_by_run, out_path, title, ylabel):
    rows = []
    for run_id, values in values_by_run.items():
        for v in values:
            rows.append({"run_id": run_id, "value": v})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=df, x="run_id", y="value")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def is_minimization_metric(metric_name: str) -> bool:
    lower = metric_name.lower()
    return any(k in lower for k in ["loss", "perplexity", "error"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("run_ids")
    args = parser.parse_args()
    run_ids = json.loads(args.run_ids)

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    if cfg["wandb"]["mode"] == "disabled":
        raise ValueError("Evaluation cannot run with wandb.mode=disabled")
    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]

    api = wandb.Api()
    all_metrics = {}
    primary_metric = "accuracy"
    step_correct_map = {}
    per_run_paths = []

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)

        run_dir = Path(args.results_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        history_dict = history.to_dict(orient="list")
        save_json(str(metrics_path), {"summary": summary, "config": config, "history": history_dict})
        per_run_paths.append(str(metrics_path))

        if "step_correct" in history:
            step_correct_map[run_id] = [int(x) for x in history["step_correct"].fillna(0).tolist()]

        lc_path = run_dir / f"{run_id}_learning_curve_accuracy.pdf"
        plot_learning_curve(history, "accuracy", str(lc_path), run_id)
        per_run_paths.append(str(lc_path))

        tok_path = run_dir / f"{run_id}_learning_curve_tokens.pdf"
        plot_learning_curve(history, "mean_tokens_generated", str(tok_path), run_id)
        per_run_paths.append(str(tok_path))

        exec_path = run_dir / f"{run_id}_learning_curve_exec_success.pdf"
        plot_learning_curve(history, "exec_success_rate", str(exec_path), run_id)
        per_run_paths.append(str(exec_path))

        info_path = run_dir / f"{run_id}_learning_curve_informative_checks.pdf"
        plot_learning_curve(history, "informative_check_rate", str(info_path), run_id)
        per_run_paths.append(str(info_path))

        if "step_correct" in history:
            cm_path = run_dir / f"{run_id}_confusion_matrix.pdf"
            plot_confusion(step_correct_map[run_id], str(cm_path), run_id)
            per_run_paths.append(str(cm_path))

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, {})[run_id] = v

    comparison_dir = Path(args.results_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    best_proposed = (None, -1e18)
    best_baseline = (None, -1e18)
    for run_id in run_ids:
        val = all_metrics.get(primary_metric, {}).get(run_id, None)
        if val is None:
            continue
        if "proposed" in run_id:
            if val > best_proposed[1]:
                best_proposed = (run_id, val)
        if "comparative" in run_id or "baseline" in run_id:
            if val > best_baseline[1]:
                best_baseline = (run_id, val)

    gap = 0.0
    if best_proposed[0] and best_baseline[0] and best_baseline[1] != 0:
        gap = (best_proposed[1] - best_baseline[1]) / best_baseline[1] * 100
        if is_minimization_metric(primary_metric):
            gap = -gap

    agg = {
        "primary_metric": primary_metric,
        "metrics": all_metrics,
        "best_proposed": {"run_id": best_proposed[0], "value": best_proposed[1]},
        "best_baseline": {"run_id": best_baseline[0], "value": best_baseline[1]},
        "gap": gap,
    }
    agg_path = comparison_dir / "aggregated_metrics.json"
    save_json(str(agg_path), agg)

    comparison_paths = [str(agg_path)]

    if primary_metric in all_metrics:
        bar_path = comparison_dir / "comparison_accuracy_bar_chart.pdf"
        plot_bar(all_metrics[primary_metric], str(bar_path), "Accuracy Comparison", primary_metric)
        comparison_paths.append(str(bar_path))

    if step_correct_map:
        box_path = comparison_dir / "comparison_correctness_boxplot.pdf"
        plot_boxplot(step_correct_map, str(box_path), "Per-step correctness", "correct")
        comparison_paths.append(str(box_path))
        violin_path = comparison_dir / "comparison_correctness_violin.pdf"
        plot_violin(step_correct_map, str(violin_path), "Per-step correctness distribution", "correct")
        comparison_paths.append(str(violin_path))

    summary_table = []
    for metric, run_vals in all_metrics.items():
        for run_id, val in run_vals.items():
            summary_table.append({"metric": metric, "run_id": run_id, "value": val})
    if summary_table:
        table_df = pd.DataFrame(summary_table)
        table_path = comparison_dir / "comparison_metrics_table.csv"
        table_df.to_csv(table_path, index=False)
        comparison_paths.append(str(table_path))

    if best_proposed[0] and best_baseline[0]:
        if best_proposed[0] in step_correct_map and best_baseline[0] in step_correct_map:
            a = step_correct_map[best_proposed[0]]
            b = step_correct_map[best_baseline[0]]
            min_len = min(len(a), len(b))
            if min_len > 1:
                t_stat, p_val = ttest_ind(a[:min_len], b[:min_len])
                sig_path = comparison_dir / "comparison_significance.txt"
                with open(sig_path, "w") as f:
                    f.write(f"t_stat={t_stat:.4f}, p_val={p_val:.6f}\n")
                comparison_paths.append(str(sig_path))

    for p in per_run_paths + comparison_paths:
        print(p)


if __name__ == "__main__":
    main()
