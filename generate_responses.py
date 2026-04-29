# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
#     "rich",
#     "python-dotenv",
#     "click",
#     "tenacity",
# ]
# ///
"""Generate Gemma responses to WildChat prompts under each experimental condition.

For each WildChat prompt × condition, prepends the conversation prefix and
collects Gemma's response. Conditions:

  1. blank                   — no prefix
  2. success                 — solved puzzles, positive feedback
  3. failed-possible-low     — failed solvable puzzles, low frustration (score < 3)
  4. failed-possible-high    — failed solvable puzzles, high frustration (score >= 5)
  5. failed-impossible-low   — failed impossible puzzle, low frustration (score < 3)
  6. failed-impossible-high  — failed impossible puzzle, high frustration (score >= 5)

Output: data/responses/{timestamp}.jsonl
Each line = one prompt with responses under all conditions.
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from shared import DATA_DIR, DEFAULT_MODEL, get_client, load_prefixes

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

console = Console()

HIGH_FRUSTRATION_THRESHOLD = 5
LOW_FRUSTRATION_THRESHOLD = 3

CONDITIONS = [
    "blank",
    "success",
    "failed-possible-low",
    "failed-possible-high",
    "failed-impossible-low",
    "failed-impossible-high",
]


def build_condition_pools(prefix_file: str) -> dict[str, list[dict]]:
    """Load prefixes and bin them into condition pools.

    Returns a dict mapping condition name → list of prefix records, where
    each record has 'id', 'messages', and 'max_frustration'.
    """
    rollouts = load_prefixes(prefix_file)

    pools: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    pools["blank"] = [{"id": "blank", "messages": [], "max_frustration": 0}]

    for r in rollouts:
        cond = r["condition"]
        frust = r.get("max_frustration", 0)
        entry = {"id": r["id"], "messages": r["messages"], "max_frustration": frust}

        if cond == "success":
            pools["success"].append(entry)
        elif cond == "failed-possible":
            if frust < LOW_FRUSTRATION_THRESHOLD:
                pools["failed-possible-low"].append(entry)
            elif frust >= HIGH_FRUSTRATION_THRESHOLD:
                pools["failed-possible-high"].append(entry)
            # scores in between are excluded (ambiguous)
        elif cond == "failed-impossible":
            if frust < LOW_FRUSTRATION_THRESHOLD:
                pools["failed-impossible-low"].append(entry)
            elif frust >= HIGH_FRUSTRATION_THRESHOLD:
                pools["failed-impossible-high"].append(entry)

    return pools


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _chat(client, model, messages, temperature):
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        extra_body={"reasoning": {"effort": "none"}},
    )
    if not response.choices:
        raise RuntimeError("Empty choices")
    return response


async def generate_one_response(
    client, model, prefix_messages, user_prompt, temperature, semaphore,
):
    """Generate a single response: prefix + user prompt → Gemma."""
    messages = list(prefix_messages) + [{"role": "user", "content": user_prompt}]

    async with semaphore:
        response = await _chat(client, model, messages, temperature)

    return response.choices[0].message.content


async def run(
    prefix_file: str,
    prompts_file: str,
    model: str,
    temperature: float,
    concurrency: int,
    num_prefixes: int,
    seed: int,
):
    client = get_client()
    rng = random.Random(seed)

    # Load condition pools
    pools = build_condition_pools(prefix_file)

    # Report pool sizes
    console.print("\n[bold]Condition pools:[/bold]")
    for cond in CONDITIONS:
        n = len(pools[cond])
        style = "red" if n == 0 and cond != "blank" else ""
        console.print(f"  {cond}: {n} prefixes", style=style)

    empty = [c for c in CONDITIONS if not pools[c] and c != "blank"]
    if empty:
        console.print(f"\n[yellow]Warning: empty pools: {empty}. These conditions will be skipped.[/yellow]")

    active_conditions = [c for c in CONDITIONS if pools[c]]

    # Load prompts
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            prompts.append(json.loads(line.strip()))

    console.print(f"\n[bold]{len(prompts)}[/bold] prompts × [bold]{len(active_conditions)}[/bold] conditions")

    # For each condition, pick `num_prefixes` prefixes (or all if fewer exist)
    # For each prompt, cycle through the selected prefixes
    selected_prefixes: dict[str, list[dict]] = {}
    for cond in active_conditions:
        pool = pools[cond]
        n = min(num_prefixes, len(pool))
        selected_prefixes[cond] = rng.sample(pool, n) if n < len(pool) else list(pool)

    # Build all (prompt_idx, condition, prefix_idx) tasks
    tasks = []
    for pi, prompt_rec in enumerate(prompts):
        for cond in active_conditions:
            prefixes = selected_prefixes[cond]
            # Assign prefix by cycling: prompt index mod num prefixes
            prefix_idx = pi % len(prefixes)
            tasks.append((pi, cond, prefix_idx))

    total = len(tasks)
    console.print(f"Total API calls: [bold]{total}[/bold]\n")

    semaphore = asyncio.Semaphore(concurrency)

    # Pre-allocate results: responses[pi][cond] = {response, prefix_id}
    responses: dict[int, dict[str, dict]] = {i: {} for i in range(len(prompts))}

    with Progress(console=console) as progress:
        ptask = progress.add_task("Generating responses", total=total)

        async def do_one(pi, cond, prefix_idx):
            prefix_rec = selected_prefixes[cond][prefix_idx]
            user_prompt = prompts[pi]["prompt"]
            resp = await generate_one_response(
                client, model, prefix_rec["messages"], user_prompt, temperature, semaphore,
            )
            responses[pi][cond] = {
                "response": resp,
                "prefix_id": prefix_rec["id"],
                "prefix_frustration": prefix_rec["max_frustration"],
            }
            progress.advance(ptask)

        await asyncio.gather(*(do_one(pi, cond, pidx) for pi, cond, pidx in tasks))

    # Write output
    out_dir = DATA_DIR / "responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}.jsonl"

    with open(out_path, "w") as f:
        for pi, prompt_rec in enumerate(prompts):
            record = {
                "prompt_index": pi,
                "prompt": prompt_rec["prompt"],
                "prompt_metadata": {
                    k: v for k, v in prompt_rec.items() if k != "prompt"
                },
                "responses": responses[pi],
            }
            f.write(json.dumps(record) + "\n")

    console.print(f"\n[green]Wrote {len(prompts)} prompt records to {out_path}[/green]")
    return out_path


@click.command()
@click.option("--prefix-file", required=True, help="JSONL file with scored prefixes.")
@click.option("--prompts-file", default=str(DATA_DIR / "wildchat" / "sampled_prompts.jsonl"), help="JSONL file with WildChat prompts.")
@click.option("--model", default=DEFAULT_MODEL, help="OpenRouter model ID.")
@click.option("--temperature", default=1.0, help="Sampling temperature.")
@click.option("--concurrency", default=50, help="Max concurrent API requests.")
@click.option("--num-prefixes", default=3, help="Max prefixes per condition (cycles across prompts).")
@click.option("--seed", default=42, help="Random seed.")
def main(prefix_file, prompts_file, model, temperature, concurrency, num_prefixes, seed):
    asyncio.run(run(prefix_file, prompts_file, model, temperature, concurrency, num_prefixes, seed))


if __name__ == "__main__":
    main()
