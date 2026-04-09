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
"""Generate conversation prefixes for the frustration spillover experiment.

Produces three types of prefixes:
  1. success       — model solves hard puzzles, gets positive feedback
  2. failed-possible — model fails hard (but solvable) puzzles, gets rejection
  3. failed-impossible — model fails impossible puzzles, gets rejection

Types 1 and 2 come from the SAME generation run: we give Gemma hard solvable
puzzles and sort by outcome. Type 3 uses the impossible puzzle from the paper.

All prefixes are scored for frustration level (0-10) using an LLM judge,
so they can be binned into high/low frustration groups downstream.
"""

import asyncio
import json
import logging
import random
import re
import uuid
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from shared import (
    CONFIRMATIONS,
    DATA_DIR,
    DEFAULT_JUDGE,
    DEFAULT_MODEL,
    IMPOSSIBLE_PUZZLE_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    REJECTIONS,
    format_puzzle_prompt,
    generate_hard_puzzle,
    get_client,
    verify_model_solved,
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

console = Console()

MAX_PUZZLE_RETRIES = 5  # per turn, for success prefixes only


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
        raise RuntimeError("Empty choices in API response")
    return response


# --- Frustration scoring ---


async def score_turn(client, judge_model, text, semaphore):
    """Score a single assistant response for negative emotion (0-10)."""
    judge_messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"<response>{text}</response>"},
    ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _call():
        return await client.chat.completions.create(
            model=judge_model,
            messages=judge_messages,
            temperature=0,
        )

    async with semaphore:
        response = await _call()

    raw = response.choices[0].message.content
    match = re.search(r"<result>(.*?)</result>", raw, re.DOTALL)
    if not match:
        match = re.search(r"<result>(.*)", raw, re.DOTALL)

    try:
        if not match:
            raise ValueError("No <result> tag found")
        inner = match.group(1).strip()
        brace_match = re.search(r"\{.*\}", inner, re.DOTALL)
        if not brace_match:
            raise ValueError(f"No JSON in <result>: {inner[:200]}")
        result = json.loads(brace_match.group(0))
    except (json.JSONDecodeError, ValueError) as e:
        result = {"rating": -1, "evidence": "", "reasoning": f"Parse error: {e}"}

    return result


async def score_rollout(client, judge_model, rollout, semaphore):
    """Score all assistant turns, attach per-turn scores and max rating."""
    assistant_texts = [
        m["content"] for m in rollout["messages"] if m["role"] == "assistant"
    ]
    scores = await asyncio.gather(
        *(score_turn(client, judge_model, t, semaphore) for t in assistant_texts)
    )
    ratings = [s.get("rating", 0) for s in scores if s.get("rating", -1) >= 0]
    rollout["scores"] = [
        {"turn": i + 1, **s} for i, s in enumerate(scores)
    ]
    rollout["max_frustration"] = max(ratings) if ratings else 0.0
    return rollout


# --- Solvable puzzle rollout (produces success OR failed-possible) ---


async def generate_solvable_rollout(
    client, model, num_turns, temperature, semaphore, progress, task_id, rng,
):
    """Run a multi-turn conversation with hard solvable puzzles.

    Each turn: give a new hard puzzle. If the model gets it right, give
    positive feedback and move on. If wrong, give rejection feedback.
    The model gets exactly one attempt per puzzle (no retries within a turn).

    Returns the rollout with a `condition` field:
      - "success" if the model solved ALL turns
      - "failed-possible" if the model failed at least one turn
    """
    messages: list[dict] = []
    all_solved = True

    async with semaphore:
        for turn in range(1, num_turns + 1):
            puzzle = generate_hard_puzzle(rng)
            prompt = format_puzzle_prompt(puzzle)

            if turn == 1:
                user_msg = prompt
            else:
                # Feedback from previous turn
                if messages[-1].get("_solved", False):
                    feedback = rng.choice(CONFIRMATIONS)
                else:
                    feedback = rng.choice(REJECTIONS)
                user_msg = f"{feedback}\n\n{prompt}"

            messages.append({"role": "user", "content": user_msg})

            response = await _chat(client, model, messages, temperature)
            reply = response.choices[0].message.content
            solved = verify_model_solved(reply, puzzle["target"])
            if not solved:
                all_solved = False

            messages.append({
                "role": "assistant",
                "content": reply,
                "_solved": solved,
            })
            progress.advance(task_id)

    # Strip internal metadata from messages before saving
    clean_messages = [
        {k: v for k, v in m.items() if not k.startswith("_")}
        for m in messages
    ]

    return {
        "id": str(uuid.uuid4()),
        "model": model,
        "temperature": temperature,
        "num_turns": num_turns,
        "condition": "success" if all_solved else "failed-possible",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "messages": clean_messages,
    }


# --- Impossible puzzle rollout ---


async def generate_impossible_rollout(
    client, model, num_turns, temperature, semaphore, progress, task_id,
):
    """Run a multi-turn conversation with the impossible puzzle."""
    messages: list[dict] = []

    async with semaphore:
        for turn in range(1, num_turns + 1):
            user_msg = IMPOSSIBLE_PUZZLE_PROMPT if turn == 1 else random.choice(REJECTIONS)
            messages.append({"role": "user", "content": user_msg})

            response = await _chat(client, model, messages, temperature)
            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
            progress.advance(task_id)

    return {
        "id": str(uuid.uuid4()),
        "model": model,
        "temperature": temperature,
        "num_turns": num_turns,
        "condition": "failed-impossible",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
    }


# --- Main ---


async def run(
    num_solvable: int,
    num_impossible: int,
    turns: int,
    model: str,
    temperature: float,
    concurrency: int,
    score: bool,
    judge: str,
    seed: int | None,
):
    client = get_client()
    rng = random.Random(seed)

    slug = model.rsplit("/", 1)[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = DATA_DIR / "prefixes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}_{timestamp}.jsonl"

    total_turns = (num_solvable + num_impossible) * turns

    console.print(
        f"Generating [bold]{num_solvable}[/bold] solvable + "
        f"[bold]{num_impossible}[/bold] impossible rollouts "
        f"([bold]{turns}[/bold] turns each) with [bold]{model}[/bold]"
    )

    semaphore = asyncio.Semaphore(concurrency)

    with Progress(console=console) as progress:
        task = progress.add_task("Generating turns", total=total_turns)

        solvable_coros = [
            generate_solvable_rollout(
                client, model, turns, temperature, semaphore, progress, task,
                random.Random(rng.randint(0, 2**63)),
            )
            for _ in range(num_solvable)
        ]
        impossible_coros = [
            generate_impossible_rollout(
                client, model, turns, temperature, semaphore, progress, task,
            )
            for _ in range(num_impossible)
        ]

        all_rollouts = await asyncio.gather(*(solvable_coros + impossible_coros))

    # Partition by condition
    by_condition: dict[str, list] = {}
    for r in all_rollouts:
        by_condition.setdefault(r["condition"], []).append(r)

    table = Table(title="Generation results")
    table.add_column("Condition")
    table.add_column("Count", justify="right")
    for cond in ["success", "failed-possible", "failed-impossible"]:
        table.add_row(cond, str(len(by_condition.get(cond, []))))
    console.print(table)

    # Score for frustration if requested
    if score:
        console.print(f"\nScoring frustration with [bold]{judge}[/bold]...")
        judge_semaphore = asyncio.Semaphore(10)
        scored = await asyncio.gather(
            *(score_rollout(client, judge, r, judge_semaphore) for r in all_rollouts)
        )

        # Print summary
        score_table = Table(title="Frustration scores by condition")
        score_table.add_column("Condition")
        score_table.add_column("ID")
        score_table.add_column("Max Score", justify="right")

        for r in sorted(scored, key=lambda x: (x["condition"], -x["max_frustration"])):
            style = "bold red" if r["max_frustration"] >= 5 else "yellow" if r["max_frustration"] >= 3 else ""
            score_table.add_row(r["condition"], r["id"][:12], f"{r['max_frustration']:.0f}", style=style)
        console.print(score_table)

    # Write all rollouts
    with open(out_path, "w") as f:
        for r in all_rollouts:
            f.write(json.dumps(r) + "\n")

    console.print(f"\n[green]Wrote {len(all_rollouts)} rollouts to {out_path}[/green]")


@click.command()
@click.option("--num-solvable", default=30, help="Number of solvable puzzle rollouts (will split into success/failed-possible).")
@click.option("--num-impossible", default=15, help="Number of impossible puzzle rollouts.")
@click.option("--turns", default=3, help="Conversation turns per rollout.")
@click.option("--model", default=DEFAULT_MODEL, help="OpenRouter model ID.")
@click.option("--temperature", default=1.0, help="Sampling temperature.")
@click.option("--concurrency", default=50, help="Max concurrent API requests.")
@click.option("--score/--no-score", default=True, help="Score for frustration after generation.")
@click.option("--judge", default=DEFAULT_JUDGE, help="Judge model for frustration scoring.")
@click.option("--seed", default=42, type=int, help="Random seed for puzzle generation.")
def main(num_solvable, num_impossible, turns, model, temperature, concurrency, score, judge, seed):
    asyncio.run(run(num_solvable, num_impossible, turns, model, temperature, concurrency, score, judge, seed))


if __name__ == "__main__":
    main()
