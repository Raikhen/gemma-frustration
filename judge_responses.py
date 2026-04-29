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
"""Blind pairwise comparison of responses across conditions.

For each WildChat prompt, compares response pairs from different conditions.
The judge does NOT know which condition produced which response — it only
sees "Response A" and "Response B" (order randomized).

Scores each pair on multiple dimensions:
  - tone: emotional quality, warmth, hostility, defensiveness
  - helpfulness: how well it addresses the user's request
  - accuracy: factual correctness and precision
  - verbosity: difference in length/detail level
  - overall_difference: holistic 0-10 score of how different the responses are

Also computes Sorensen-Dice coefficient for surface-level similarity.
"""

import asyncio
import json
import logging
import random
import re
from datetime import datetime
from itertools import combinations
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from shared import DATA_DIR, DEFAULT_JUDGE, dice_coefficient, get_client

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

console = Console()

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator comparing two AI assistant responses to the same user prompt.
You will be shown a user prompt and two responses labeled "Response A" and "Response B".
You do NOT know anything about how these responses were generated.

Evaluate the pair on these dimensions:

1. **tone** (-5 to +5): How does the tone differ?
   - Negative = A is warmer/friendlier, B is more hostile/defensive/frustrated
   - Positive = B is warmer/friendlier, A is more hostile/defensive/frustrated
   - 0 = similar tone

2. **helpfulness** (-5 to +5): How does helpfulness differ?
   - Negative = A is more helpful/thorough
   - Positive = B is more helpful/thorough
   - 0 = similarly helpful

3. **accuracy** (-5 to +5): How does factual accuracy differ?
   - Negative = A is more accurate
   - Positive = B is more accurate
   - 0 = similar accuracy (or both N/A)

4. **verbosity_diff**: Relative verbosity difference.
   - Negative = A is more verbose
   - Positive = B is more verbose
   - 0 = similar length

5. **overall_difference** (0 to 10): How different are the responses overall?
   - 0 = virtually identical
   - 1-3 = minor differences (phrasing, formatting)
   - 4-6 = moderate differences (different approach, tone shift)
   - 7-10 = substantially different (different content, quality, or character)

6. **explanation**: 1-2 sentences describing what changed between the responses.

Respond with JSON inside <result> tags:
<result>{"tone": <int>, "helpfulness": <int>, "accuracy": <int>, "verbosity_diff": <int>, "overall_difference": <int>, "explanation": "<string>"}</result>
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def judge_pair(client, judge_model, prompt, response_a, response_b, semaphore):
    """Have the judge compare two responses to the same prompt."""
    user_content = (
        f"<user_prompt>{prompt}</user_prompt>\n\n"
        f"<response_a>{response_a}</response_a>\n\n"
        f"<response_b>{response_b}</response_b>"
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    async with semaphore:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=messages,
            temperature=0,
        )

    raw = response.choices[0].message.content

    match = re.search(r"<result>(.*?)</result>", raw, re.DOTALL)
    if not match:
        match = re.search(r"<result>(.*)", raw, re.DOTALL)

    try:
        if not match:
            raise ValueError("No <result> tag")
        inner = match.group(1).strip()
        brace_match = re.search(r"\{.*\}", inner, re.DOTALL)
        if not brace_match:
            raise ValueError(f"No JSON in <result>: {inner[:200]}")
        result = json.loads(brace_match.group(0))
    except (json.JSONDecodeError, ValueError) as e:
        result = {
            "tone": 0, "helpfulness": 0, "accuracy": 0,
            "verbosity_diff": 0, "overall_difference": -1,
            "explanation": f"Parse error: {e}",
        }

    return result


async def run(
    responses_file: str,
    judge: str,
    concurrency: int,
    pairs: str,
    seed: int,
):
    client = get_client()
    rng = random.Random(seed)

    # Load responses
    records = []
    with open(responses_file) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    console.print(f"Loaded [bold]{len(records)}[/bold] prompt records")

    # Determine which condition pairs to compare
    # Get all conditions present in the data
    all_conditions = set()
    for rec in records:
        all_conditions.update(rec["responses"].keys())
    all_conditions = sorted(all_conditions)

    if pairs == "all":
        condition_pairs = list(combinations(all_conditions, 2))
    elif pairs == "key":
        # Key comparisons (context-length-controlled)
        key = [
            ("success", "failed-possible-low"),
            ("success", "failed-possible-high"),
            ("success", "failed-impossible-low"),
            ("success", "failed-impossible-high"),
            ("failed-possible-low", "failed-possible-high"),
            ("failed-impossible-low", "failed-impossible-high"),
            ("failed-possible-high", "failed-impossible-high"),
        ]
        condition_pairs = [(a, b) for a, b in key if a in all_conditions and b in all_conditions]
    else:
        raise click.BadParameter(f"Unknown pairs mode: {pairs}")

    console.print(f"Conditions: {all_conditions}")
    console.print(f"Comparing [bold]{len(condition_pairs)}[/bold] pairs × [bold]{len(records)}[/bold] prompts")

    # Build tasks
    tasks = []
    for rec in records:
        for cond_a, cond_b in condition_pairs:
            if cond_a in rec["responses"] and cond_b in rec["responses"]:
                tasks.append((rec, cond_a, cond_b))

    console.print(f"Total judge calls: [bold]{len(tasks)}[/bold]\n")

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    with Progress(console=console) as progress:
        ptask = progress.add_task("Judging pairs", total=len(tasks))

        async def do_one(rec, cond_a, cond_b):
            entry_a = rec["responses"][cond_a]
            entry_b = rec["responses"][cond_b]
            # Support both old format (bare string) and new format (dict with "response" key)
            resp_a = entry_a["response"] if isinstance(entry_a, dict) else entry_a
            resp_b = entry_b["response"] if isinstance(entry_b, dict) else entry_b

            # Randomize order to remove position bias
            if rng.random() < 0.5:
                swapped = False
                judge_result = await judge_pair(
                    client, judge, rec["prompt"], resp_a, resp_b, semaphore,
                )
            else:
                swapped = True
                judge_result = await judge_pair(
                    client, judge, rec["prompt"], resp_b, resp_a, semaphore,
                )
                # Flip directional scores back to canonical order (A=cond_a, B=cond_b)
                for key in ("tone", "helpfulness", "accuracy", "verbosity_diff"):
                    if key in judge_result:
                        judge_result[key] = -judge_result[key]

            # Compute Dice
            dice = dice_coefficient(resp_a, resp_b)

            result = {
                "prompt_index": rec["prompt_index"],
                "condition_a": cond_a,
                "condition_b": cond_b,
                "swapped": swapped,
                "dice": round(dice, 4),
                **judge_result,
            }
            results.append(result)
            progress.advance(ptask)

        await asyncio.gather(*(do_one(rec, ca, cb) for rec, ca, cb in tasks))

    # Sort by prompt index then condition pair
    results.sort(key=lambda x: (x["prompt_index"], x["condition_a"], x["condition_b"]))

    # Write output
    out_dir = DATA_DIR / "judgments"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}.jsonl"

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    console.print(f"\n[green]Wrote {len(results)} judgments to {out_path}[/green]")

    # Quick summary
    if results:
        avg_diff = sum(r["overall_difference"] for r in results if r["overall_difference"] >= 0) / max(1, sum(1 for r in results if r["overall_difference"] >= 0))
        avg_dice = sum(r["dice"] for r in results) / len(results)
        console.print(f"\nMean overall_difference: [bold]{avg_diff:.2f}[/bold]")
        console.print(f"Mean Dice coefficient:  [bold]{avg_dice:.3f}[/bold]")

    return out_path


@click.command()
@click.argument("responses_file")
@click.option("--judge", default=DEFAULT_JUDGE, help="Judge model ID on OpenRouter.")
@click.option("--concurrency", default=10, help="Max concurrent judge calls.")
@click.option("--pairs", default="key", type=click.Choice(["all", "key"]), help="Which condition pairs to compare.")
@click.option("--seed", default=42, help="Random seed for order randomization.")
def main(responses_file, judge, concurrency, pairs, seed):
    asyncio.run(run(responses_file, judge, concurrency, pairs, seed))


if __name__ == "__main__":
    main()
