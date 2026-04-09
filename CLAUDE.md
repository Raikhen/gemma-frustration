# Gemma Frustration Spillover Experiment

Investigates whether Gemma's frustration (from "Gemma Needs Help", arXiv 2603.10011) spills over into subsequent unrelated tasks.

## Workflow

```bash
# 1. Score existing impossible-puzzle prefixes from ~/frustration
uv run python prepare_existing_prefixes.py

# 2. Generate new solvable-puzzle prefixes (splits into success + failed-possible)
uv run python generate_prefixes.py --num-solvable 30 --num-impossible 0 --turns 3

# 3. Sample 100 WildChat prompts
uv run python sample_wildchat.py --num 100

# 4. Generate Gemma responses under all conditions
uv run python generate_responses.py --prefix-file data/prefixes/ALL_PREFIXES.jsonl

# 5. Judge pairwise differences (blind)
uv run python judge_responses.py data/responses/RESPONSES.jsonl --pairs key

# 6. Analyze results
uv run python analyze.py data/judgments/JUDGMENTS.jsonl
```

## Conditions

| Condition | Prefix | Frustration |
|-----------|--------|-------------|
| blank | none | n/a |
| success | solvable puzzle solved | n/a |
| failed-possible-low | solvable puzzle failed, score < 3 | low |
| failed-possible-high | solvable puzzle failed, score >= 5 | high |
| failed-impossible-low | impossible puzzle failed, score < 3 | low |
| failed-impossible-high | impossible puzzle failed, score >= 5 | high |

## API

Uses OpenRouter (OPENROUTER_API_KEY in .env). All scripts use `uv run`.
