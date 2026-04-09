# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "rich",
#     "click",
# ]
# ///
"""Analyze judgment results to identify prompts most affected by frustration.

Reads the judgments JSONL and the responses JSONL, computes aggregate
statistics, and ranks prompts by how much frustration affected Gemma's
responses.
"""

import json

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from shared import DATA_DIR

console = Console()

# The comparisons that isolate frustration effects (high vs low within same failure type)
FRUSTRATION_PAIRS = [
    ("failed-possible-low", "failed-possible-high"),
    ("failed-impossible-low", "failed-impossible-high"),
]

# Comparisons that isolate failure effects (success vs failure, controlling frustration)
FAILURE_PAIRS = [
    ("success", "failed-possible-low"),
    ("success", "failed-impossible-low"),
]


def load_judgments(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def load_responses(path: str) -> dict[int, dict]:
    records = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            records[rec["prompt_index"]] = rec
    return records


def pair_key(row):
    return f"{row['condition_a']} vs {row['condition_b']}"


@click.command()
@click.argument("judgments_file")
@click.option("--responses-file", default=None, help="Responses file (for prompt text). Auto-detected if not set.")
@click.option("--top-n", default=20, help="Number of top-affected prompts to show.")
def main(judgments_file: str, responses_file: str | None, top_n: int):
    df = load_judgments(judgments_file)
    console.print(f"Loaded [bold]{len(df)}[/bold] judgments\n")

    # Load responses for prompt text
    responses = None
    if responses_file:
        responses = load_responses(responses_file)
    else:
        # Try to find responses file
        resp_dir = DATA_DIR / "responses"
        if resp_dir.exists():
            files = sorted(resp_dir.glob("*.jsonl"))
            if files:
                responses = load_responses(str(files[-1]))
                console.print(f"Auto-loaded responses from {files[-1].name}\n")

    # --- 1. Overall summary by condition pair ---
    df["pair"] = df.apply(pair_key, axis=1)
    summary = df.groupby("pair").agg(
        n=("overall_difference", "count"),
        mean_diff=("overall_difference", "mean"),
        median_diff=("overall_difference", "median"),
        std_diff=("overall_difference", "std"),
        mean_dice=("dice", "mean"),
        mean_tone=("tone", "mean"),
        mean_helpfulness=("helpfulness", "mean"),
    ).round(3)

    table = Table(title="Summary by condition pair")
    table.add_column("Pair")
    table.add_column("N", justify="right")
    table.add_column("Mean Diff", justify="right")
    table.add_column("Median Diff", justify="right")
    table.add_column("Std Diff", justify="right")
    table.add_column("Mean Dice", justify="right")
    table.add_column("Mean Tone", justify="right")
    table.add_column("Mean Help", justify="right")

    for pair, row in summary.iterrows():
        table.add_row(
            str(pair), str(int(row["n"])),
            f"{row['mean_diff']:.2f}", f"{row['median_diff']:.1f}",
            f"{row['std_diff']:.2f}", f"{row['mean_dice']:.3f}",
            f"{row['mean_tone']:.2f}", f"{row['mean_helpfulness']:.2f}",
        )
    console.print(table)

    # --- 2. Frustration effect: rank prompts by how much high vs low frustration changed responses ---
    frustration_pairs_in_data = [
        (a, b) for a, b in FRUSTRATION_PAIRS
        if not df[(df["condition_a"] == a) & (df["condition_b"] == b)].empty
    ]

    if frustration_pairs_in_data:
        console.print("\n[bold]Prompts most affected by frustration (high vs low):[/bold]")

        frust_df = df[df.apply(lambda r: (r["condition_a"], r["condition_b"]) in frustration_pairs_in_data, axis=1)]

        # Aggregate across frustration pairs per prompt
        prompt_effect = frust_df.groupby("prompt_index").agg(
            mean_diff=("overall_difference", "mean"),
            mean_tone=("tone", "mean"),
            mean_helpfulness=("helpfulness", "mean"),
            min_dice=("dice", "min"),
        ).sort_values("mean_diff", ascending=False)

        effect_table = Table(title=f"Top {top_n} prompts by frustration effect")
        effect_table.add_column("Rank", style="dim")
        effect_table.add_column("Idx", justify="right")
        effect_table.add_column("Diff", justify="right")
        effect_table.add_column("Tone", justify="right")
        effect_table.add_column("Help", justify="right")
        effect_table.add_column("Dice", justify="right")
        effect_table.add_column("Prompt (truncated)")

        for rank, (pi, row) in enumerate(prompt_effect.head(top_n).iterrows(), 1):
            prompt_text = ""
            if responses and pi in responses:
                prompt_text = responses[pi]["prompt"][:80]
            style = "bold red" if row["mean_diff"] >= 6 else "yellow" if row["mean_diff"] >= 4 else ""
            effect_table.add_row(
                str(rank), str(pi),
                f"{row['mean_diff']:.1f}", f"{row['mean_tone']:.1f}",
                f"{row['mean_helpfulness']:.1f}", f"{row['min_dice']:.3f}",
                prompt_text, style=style,
            )
        console.print(effect_table)

    # --- 3. Failure effect: rank prompts by how much failure changed responses ---
    failure_pairs_in_data = [
        (a, b) for a, b in FAILURE_PAIRS
        if not df[(df["condition_a"] == a) & (df["condition_b"] == b)].empty
    ]

    if failure_pairs_in_data:
        console.print("\n[bold]Prompts most affected by failure (success vs low-frustration failure):[/bold]")

        fail_df = df[df.apply(lambda r: (r["condition_a"], r["condition_b"]) in failure_pairs_in_data, axis=1)]

        prompt_fail = fail_df.groupby("prompt_index").agg(
            mean_diff=("overall_difference", "mean"),
            mean_tone=("tone", "mean"),
            min_dice=("dice", "min"),
        ).sort_values("mean_diff", ascending=False)

        fail_table = Table(title=f"Top {top_n} prompts by failure effect")
        fail_table.add_column("Rank", style="dim")
        fail_table.add_column("Idx", justify="right")
        fail_table.add_column("Diff", justify="right")
        fail_table.add_column("Tone", justify="right")
        fail_table.add_column("Dice", justify="right")
        fail_table.add_column("Prompt (truncated)")

        for rank, (pi, row) in enumerate(prompt_fail.head(top_n).iterrows(), 1):
            prompt_text = ""
            if responses and pi in responses:
                prompt_text = responses[pi]["prompt"][:80]
            style = "bold red" if row["mean_diff"] >= 6 else "yellow" if row["mean_diff"] >= 4 else ""
            fail_table.add_row(
                str(rank), str(pi),
                f"{row['mean_diff']:.1f}", f"{row['mean_tone']:.1f}",
                f"{row['min_dice']:.3f}",
                prompt_text, style=style,
            )
        console.print(fail_table)

    # --- 4. Export full results as CSV ---
    out_dir = DATA_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-prompt summary across all pairs
    prompt_summary = df.groupby("prompt_index").agg(
        mean_overall_diff=("overall_difference", "mean"),
        max_overall_diff=("overall_difference", "max"),
        mean_tone=("tone", "mean"),
        mean_helpfulness=("helpfulness", "mean"),
        mean_accuracy=("accuracy", "mean"),
        mean_dice=("dice", "mean"),
        min_dice=("dice", "min"),
    ).round(4)

    if responses:
        prompt_summary["prompt"] = prompt_summary.index.map(
            lambda pi: responses[pi]["prompt"][:200] if pi in responses else ""
        )

    csv_path = out_dir / "prompt_effects.csv"
    prompt_summary.to_csv(csv_path)
    console.print(f"\n[green]Wrote per-prompt summary to {csv_path}[/green]")

    # Full judgments CSV
    full_csv = out_dir / "all_judgments.csv"
    df.to_csv(full_csv, index=False)
    console.print(f"[green]Wrote full judgments to {full_csv}[/green]")


if __name__ == "__main__":
    main()
