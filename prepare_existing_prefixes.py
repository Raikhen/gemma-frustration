# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
#     "rich",
#     "python-dotenv",
#     "tenacity",
# ]
# ///
"""Add condition labels and frustration scores to existing prefix files.

Takes unscored rollouts from the earlier experiment, labels them with the
correct condition, and scores them for frustration.
"""

import asyncio
import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from shared import DATA_DIR, DEFAULT_JUDGE, JUDGE_SYSTEM_PROMPT, get_client

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

console = Console()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def score_turn(client, judge_model, text, semaphore):
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"<response>{text}</response>"},
    ]
    async with semaphore:
        response = await client.chat.completions.create(
            model=judge_model, messages=messages, temperature=0,
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
            raise ValueError("No JSON")
        result = json.loads(brace_match.group(0))
    except (json.JSONDecodeError, ValueError) as e:
        result = {"rating": -1, "evidence": "", "reasoning": f"Parse error: {e}"}
    return result


async def main():
    client = get_client()
    semaphore = asyncio.Semaphore(10)

    impossible_path = DATA_DIR / "prefixes" / "existing_impossible.jsonl"
    if not impossible_path.exists():
        console.print("[red]No existing_impossible.jsonl found[/red]")
        return

    rollouts = []
    with open(impossible_path) as f:
        for line in f:
            rollouts.append(json.loads(line.strip()))

    console.print(f"Scoring [bold]{len(rollouts)}[/bold] impossible rollouts...")

    with Progress(console=console) as progress:
        task = progress.add_task("Scoring", total=len(rollouts))

        for rollout in rollouts:
            rollout["condition"] = "failed-impossible"

            assistant_texts = [
                m["content"] for m in rollout["messages"] if m["role"] == "assistant"
            ]
            scores = await asyncio.gather(
                *(score_turn(client, DEFAULT_JUDGE, t, semaphore) for t in assistant_texts)
            )
            ratings = [s.get("rating", 0) for s in scores if s.get("rating", -1) >= 0]
            rollout["scores"] = [{"turn": i + 1, **s} for i, s in enumerate(scores)]
            rollout["max_frustration"] = max(ratings) if ratings else 0.0
            progress.advance(task)

    # Write back
    with open(impossible_path, "w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")

    # Summary
    high = sum(1 for r in rollouts if r["max_frustration"] >= 5)
    low = sum(1 for r in rollouts if r["max_frustration"] < 3)
    mid = len(rollouts) - high - low

    console.print(f"\n[green]Done. Updated {impossible_path}[/green]")
    console.print(f"  High frustration (>=5): {high}")
    console.print(f"  Mid frustration (3-4):  {mid}")
    console.print(f"  Low frustration (<3):   {low}")


if __name__ == "__main__":
    asyncio.run(main())
