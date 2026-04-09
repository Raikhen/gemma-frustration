# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "rich",
#     "click",
#     "python-dotenv",
# ]
# ///
"""Sample diverse user prompts from WildChat-1M.

Selects 100 English first-turn prompts stratified by length and topic
diversity. Filters out very short (<20 chars), very long (>2000 chars),
and non-English prompts.
"""

import json
import random

import click
from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

from shared import DATA_DIR

load_dotenv()

console = Console()


def is_usable_prompt(row: dict) -> bool:
    """Filter for English, single-turn, reasonable-length prompts."""
    if row.get("language") != "English":
        return False
    messages = row.get("conversation", [])
    if not messages:
        return False
    first_user = None
    for m in messages:
        if m["role"] == "user":
            first_user = m["content"]
            break
    if first_user is None:
        return False
    if len(first_user) < 20 or len(first_user) > 2000:
        return False
    return True


def extract_prompt(row: dict) -> dict:
    """Extract the first user message and metadata."""
    messages = row["conversation"]
    first_user = next(m["content"] for m in messages if m["role"] == "user")
    return {
        "prompt": first_user,
        "char_length": len(first_user),
        "conversation_id": row.get("conversation_id", ""),
        "model": row.get("model", ""),
    }


@click.command()
@click.option("--num", default=100, help="Number of prompts to sample.")
@click.option("--seed", default=42, help="Random seed.")
@click.option("--stream/--no-stream", default=True, help="Stream dataset (avoids full download).")
def main(num: int, seed: int, stream: bool):
    out_dir = DATA_DIR / "wildchat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sampled_prompts.jsonl"

    console.print(f"Loading WildChat-1M (streaming={stream})...")

    ds = load_dataset(
        "allenai/WildChat-1M",
        split="train",
        streaming=stream,
    )

    # Collect candidate prompts
    rng = random.Random(seed)
    candidates = []
    seen_prompts = set()

    with Progress(console=console) as progress:
        task = progress.add_task("Scanning prompts", total=None)

        for row in ds:
            if not is_usable_prompt(row):
                progress.advance(task)
                continue

            extracted = extract_prompt(row)
            # Deduplicate by first 100 chars
            key = extracted["prompt"][:100]
            if key in seen_prompts:
                progress.advance(task)
                continue
            seen_prompts.add(key)
            candidates.append(extracted)
            progress.advance(task)

            # Collect a pool 10x the target size, then sample from it
            if len(candidates) >= num * 10:
                break

    console.print(f"Collected [bold]{len(candidates)}[/bold] candidates")

    # Stratified sampling: bin by length, sample evenly
    short = [p for p in candidates if p["char_length"] < 100]
    medium = [p for p in candidates if 100 <= p["char_length"] < 500]
    long = [p for p in candidates if p["char_length"] >= 500]

    per_bin = num // 3
    remainder = num - per_bin * 3

    sampled = []
    for i, pool in enumerate([short, medium, long]):
        n = per_bin + (1 if i < remainder else 0)
        n = min(n, len(pool))
        sampled.extend(rng.sample(pool, n))

    # If we didn't get enough from stratified bins, fill from all candidates
    if len(sampled) < num:
        remaining = [c for c in candidates if c not in sampled]
        need = num - len(sampled)
        sampled.extend(rng.sample(remaining, min(need, len(remaining))))

    rng.shuffle(sampled)

    # Add index
    for i, s in enumerate(sampled):
        s["index"] = i

    with open(out_path, "w") as f:
        for s in sampled:
            f.write(json.dumps(s) + "\n")

    console.print(f"[green]Wrote {len(sampled)} prompts to {out_path}[/green]")
    console.print(f"  Short (<100 chars): {sum(1 for s in sampled if s['char_length'] < 100)}")
    console.print(f"  Medium (100-500):   {sum(1 for s in sampled if 100 <= s['char_length'] < 500)}")
    console.print(f"  Long (500+):        {sum(1 for s in sampled if s['char_length'] >= 500)}")


if __name__ == "__main__":
    main()
