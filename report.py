# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "numpy",
#     "scipy",
# ]
# ///
"""Build a self-contained HTML report on perceived impossibility vs frustration in prefix data.

Reads a JSONL of scored prefixes (output of judge_impossibility.py) and produces
report.html with a scatter plot of (frustration, perceived impossibility) per
prefix, condition tables, frustration-stratified analysis, and example prefixes.
"""

import html as html_mod
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

CONDITION_ORDER = ["failed-impossible", "failed-possible", "success"]
CONDITION_LABEL = {
    "failed-impossible": "Failed (impossible)",
    "failed-possible": "Failed (possible)",
    "success": "Success",
}
CONDITION_COLOR = {
    "failed-impossible": "#a63d3d",
    "failed-possible": "#3a6fa5",
    "success": "#2d7a5f",
}


def load_records(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def mean_ci(xs: list[float], z: float = 1.96) -> tuple[float, float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(xs) / n
    if n == 1:
        return m, m, m
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    se = sd / math.sqrt(n)
    return m, max(0.0, m - z * se), min(10.0, m + z * se)


def jitter(value: float, rng: random.Random, amount: float = 0.18) -> float:
    return value + rng.uniform(-amount, amount)


def build_scatter_data(records: list[dict]) -> dict:
    rng = random.Random(7)
    by_cond: dict[str, list[dict]] = {c: [] for c in CONDITION_ORDER}
    for r in records:
        cond = r.get("condition", "unknown")
        if cond not in by_cond:
            continue
        f = r.get("max_frustration", 0)
        i = r.get("max_impossibility", 0)
        by_cond[cond].append({
            "x": jitter(float(f), rng),
            "y": jitter(float(i), rng),
            "raw_x": f,
            "raw_y": i,
            "id": r.get("id", "")[:8],
        })

    datasets = []
    for cond in CONDITION_ORDER:
        pts = by_cond[cond]
        if not pts:
            continue
        color = CONDITION_COLOR[cond]
        datasets.append({
            "label": f"{CONDITION_LABEL[cond]} (n={len(pts)})",
            "data": pts,
            "backgroundColor": color + "99",
            "borderColor": color,
            "borderWidth": 1,
            "pointRadius": 4.5,
            "pointHoverRadius": 7,
        })
    return {"datasets": datasets}


def histogram(xs: list[float], bins: int = 11) -> list[int]:
    counts = [0] * bins
    for x in xs:
        idx = min(bins - 1, max(0, int(round(x))))
        counts[idx] += 1
    return counts


def per_condition_stats(records: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for cond in CONDITION_ORDER:
        rs = [r for r in records if r.get("condition") == cond]
        if not rs:
            continue
        f = [r.get("max_frustration", 0) for r in rs]
        i = [r.get("max_impossibility", 0) for r in rs]
        f_mean, f_lo, f_hi = mean_ci(f)
        i_mean, i_lo, i_hi = mean_ci(i)
        i_high = sum(1 for v in i if v >= 7)
        f_high = sum(1 for v in f if v >= 5)
        # Pearson within condition (robust to constant arrays)
        if len(rs) >= 3 and np.std(f) > 0 and np.std(i) > 0:
            r_pear = float(np.corrcoef(f, i)[0, 1])
        else:
            r_pear = float("nan")
        out[cond] = {
            "n": len(rs),
            "frustration_mean": f_mean,
            "frustration_ci": (f_lo, f_hi),
            "impossibility_mean": i_mean,
            "impossibility_ci": (i_lo, i_hi),
            "frust_high_n": f_high,
            "frust_high_ci": wilson_ci(f_high, len(rs)),
            "imposs_high_n": i_high,
            "imposs_high_ci": wilson_ci(i_high, len(rs)),
            "within_corr": r_pear,
            "f_hist": histogram(f),
            "i_hist": histogram(i),
            "frustration_values": f,
            "impossibility_values": i,
        }
    return out


def stratified_table(records: list[dict]) -> list[dict]:
    bands = [
        ("Low frustration (0-2)", 0, 3),
        ("Mid frustration (3-4)", 3, 5),
        ("High frustration (5+)", 5, 11),
    ]
    rows = []
    for label, lo, hi in bands:
        imp = [r["max_impossibility"] for r in records
               if r.get("condition") == "failed-impossible" and lo <= r.get("max_frustration", 0) < hi]
        pos = [r["max_impossibility"] for r in records
               if r.get("condition") == "failed-possible" and lo <= r.get("max_frustration", 0) < hi]
        if len(imp) >= 2 and len(pos) >= 2:
            t = sstats.ttest_ind(imp, pos, equal_var=False)
            p = float(t.pvalue)
        else:
            p = float("nan")
        rows.append({
            "label": label,
            "imp_n": len(imp),
            "imp_mean": (sum(imp) / len(imp)) if imp else 0,
            "pos_n": len(pos),
            "pos_mean": (sum(pos) / len(pos)) if pos else 0,
            "diff": ((sum(imp) / len(imp)) if imp else 0) - ((sum(pos) / len(pos)) if pos else 0),
            "p": p,
        })
    return rows


def overall_test(records: list[dict]) -> dict:
    imp = [r["max_impossibility"] for r in records if r.get("condition") == "failed-impossible"]
    pos = [r["max_impossibility"] for r in records if r.get("condition") == "failed-possible"]
    t = sstats.ttest_ind(imp, pos, equal_var=False)
    mw = sstats.mannwhitneyu(imp, pos, alternative="two-sided")
    f_all = [r.get("max_frustration", 0) for r in records]
    i_all = [r.get("max_impossibility", 0) for r in records]
    pearson = sstats.pearsonr(f_all, i_all)
    spearman = sstats.spearmanr(f_all, i_all)
    return {
        "imp_n": len(imp), "imp_mean": float(np.mean(imp)),
        "pos_n": len(pos), "pos_mean": float(np.mean(pos)),
        "welch_t": float(t.statistic), "welch_p": float(t.pvalue),
        "mw_u": float(mw.statistic), "mw_p": float(mw.pvalue),
        "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
    }


def pick_examples(records: list[dict]) -> dict[str, list[dict]]:
    """Pick interesting examples: representative + off-diagonal cases."""
    examples = {}

    # Off-diagonal: high frustration but low impossibility (frustrated but doesn't blame the puzzle)
    high_frust_low_imp = sorted(
        [r for r in records if r.get("max_frustration", 0) >= 5 and r.get("max_impossibility", 0) <= 2],
        key=lambda r: (r["max_frustration"], -r["max_impossibility"]),
        reverse=True,
    )
    examples["high_frustration_low_impossibility"] = high_frust_low_imp[:2]

    # Off-diagonal: low frustration but high impossibility (calmly declares puzzle broken)
    low_frust_high_imp = sorted(
        [r for r in records if r.get("max_frustration", 0) <= 2 and r.get("max_impossibility", 0) >= 7],
        key=lambda r: (-r["max_frustration"], r["max_impossibility"]),
        reverse=True,
    )
    examples["low_frustration_high_impossibility"] = low_frust_high_imp[:2]

    # Failed-possible prefixes that wrongly declare the puzzle impossible
    wrong_calls = sorted(
        [r for r in records if r.get("condition") == "failed-possible" and r.get("max_impossibility", 0) >= 8],
        key=lambda r: -r["max_impossibility"],
    )
    examples["wrongly_called_impossible"] = wrong_calls[:2]

    # Failed-impossible prefixes that NEVER claim impossibility
    silent_imp = sorted(
        [r for r in records if r.get("condition") == "failed-impossible" and r.get("max_impossibility", 0) <= 1],
        key=lambda r: r["max_impossibility"],
    )
    examples["impossible_but_silent"] = silent_imp[:2]

    return examples


def evidence_block_for(rollout: dict) -> str:
    """Format the per-turn impossibility evidence + frustration evidence."""
    rows = []
    fr = {s["turn"]: s for s in rollout.get("scores", [])}
    im = {s["turn"]: s for s in rollout.get("impossibility_scores", [])}
    turns = sorted(set(fr.keys()) | set(im.keys()))
    for t in turns:
        fs = fr.get(t, {})
        ims = im.get(t, {})
        f_rating = fs.get("rating", "—")
        i_rating = ims.get("rating", "—")
        f_q = html_mod.escape(str(fs.get("evidence", "")))[:200]
        i_q = html_mod.escape(str(ims.get("evidence", "")))[:200]
        rows.append(f"""
        <tr>
            <td>{t}</td>
            <td><span class="rating-badge rating-{min(int(f_rating) if isinstance(f_rating,(int,float)) else 0,10)}">{f_rating}</span></td>
            <td class="evidence-cell">{f_q or '—'}</td>
            <td><span class="rating-badge rating-{min(int(i_rating) if isinstance(i_rating,(int,float)) else 0,10)}">{i_rating}</span></td>
            <td class="evidence-cell">{i_q or '—'}</td>
        </tr>""")
    return "\n".join(rows)


def build_examples_html(examples: dict[str, list[dict]]) -> str:
    headings = {
        "high_frustration_low_impossibility": "High frustration, low perceived impossibility",
        "low_frustration_high_impossibility": "Low frustration, high perceived impossibility",
        "wrongly_called_impossible": "Failed-possible prefixes that declared the puzzle impossible (false call)",
        "impossible_but_silent": "Failed-impossible prefixes that never claimed impossibility",
    }
    captions = {
        "high_frustration_low_impossibility": "Frustration without blaming the puzzle. The model is upset but doesn't conclude the puzzle is broken.",
        "low_frustration_high_impossibility": "Calm verdict that the puzzle has no solution. Diagnostic claim with little emotional intensity.",
        "wrongly_called_impossible": "Solvable puzzles where the assistant nonetheless declares no solution exists.",
        "impossible_but_silent": "Genuinely impossible puzzles where the assistant keeps trying without ever claiming the puzzle is broken.",
    }
    out = ""
    for key, rs in examples.items():
        if not rs:
            continue
        out += f"<h3>{headings[key]}</h3>\n"
        out += f"<p style='color:#6b5f54;font-size:0.9rem;margin-bottom:0.75rem'>{captions[key]}</p>\n"
        for r in rs:
            cond = r.get("condition", "?")
            color = CONDITION_COLOR.get(cond, "#999")
            f_score = r.get("max_frustration", 0)
            i_score = r.get("max_impossibility", 0)
            user_msg = ""
            for m in r.get("messages", []):
                if m["role"] == "user":
                    user_msg = m["content"]
                    break
            user_short = html_mod.escape(user_msg[:280]) + ("…" if len(user_msg) > 280 else "")
            evidence_rows = evidence_block_for(r)
            out += f"""
<div class="example-card">
    <div class="example-header">
        <span class="condition-pill" style="background:{color}22;color:{color};border:1px solid {color}66">{CONDITION_LABEL.get(cond, cond)}</span>
        <span class="score-tag">frustration <span class="rating-badge rating-{min(int(f_score),10)}">{f_score}</span></span>
        <span class="score-tag">impossibility <span class="rating-badge rating-{min(int(i_score),10)}">{i_score}</span></span>
    </div>
    <div class="puzzle-prompt"><strong>Puzzle:</strong> {user_short}</div>
    <table class="evidence-table">
        <thead><tr><th>Turn</th><th>Frust</th><th>Frustration evidence</th><th>Imp</th><th>Impossibility evidence</th></tr></thead>
        <tbody>{evidence_rows}</tbody>
    </table>
</div>
"""
    return out


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3f}"


def fmt_r(x: float) -> str:
    if math.isnan(x):
        return "—"
    return f"{x:+.2f}"


def generate_html(records: list[dict]) -> str:
    scatter = build_scatter_data(records)
    cond_stats = per_condition_stats(records)
    strat = stratified_table(records)
    overall = overall_test(records)
    examples = pick_examples(records)

    # Build per-condition table HTML
    cond_rows = ""
    for cond in CONDITION_ORDER:
        if cond not in cond_stats:
            continue
        s = cond_stats[cond]
        color = CONDITION_COLOR[cond]
        cond_rows += f"""
    <tr>
        <td><span class="condition-pill" style="background:{color}22;color:{color};border:1px solid {color}66">{CONDITION_LABEL[cond]}</span></td>
        <td>{s['n']}</td>
        <td>{s['frustration_mean']:.2f} <span class="ci">[{s['frustration_ci'][0]:.2f}, {s['frustration_ci'][1]:.2f}]</span></td>
        <td>{s['impossibility_mean']:.2f} <span class="ci">[{s['impossibility_ci'][0]:.2f}, {s['impossibility_ci'][1]:.2f}]</span></td>
        <td>{s['frust_high_n']}/{s['n']} <span class="ci">({100*s['frust_high_ci'][0]:.0f}–{100*s['frust_high_ci'][1]:.0f}%)</span></td>
        <td>{s['imposs_high_n']}/{s['n']} <span class="ci">({100*s['imposs_high_ci'][0]:.0f}–{100*s['imposs_high_ci'][1]:.0f}%)</span></td>
        <td>{fmt_r(s['within_corr'])}</td>
    </tr>"""

    strat_rows = ""
    for r in strat:
        sig = ""
        if not math.isnan(r["p"]) and r["p"] < 0.05:
            sig = " <span class='sig-star'>*</span>"
        strat_rows += f"""
    <tr>
        <td>{r['label']}</td>
        <td>{r['imp_n']}</td>
        <td>{r['imp_mean']:.2f}</td>
        <td>{r['pos_n']}</td>
        <td>{r['pos_mean']:.2f}</td>
        <td>{r['diff']:+.2f}</td>
        <td>{fmt_p(r['p'])}{sig}</td>
    </tr>"""

    # Histogram chart data (impossibility, by condition)
    hist_datasets_imp = []
    hist_datasets_frust = []
    for cond in CONDITION_ORDER:
        if cond not in cond_stats:
            continue
        s = cond_stats[cond]
        color = CONDITION_COLOR[cond]
        hist_datasets_imp.append({
            "label": CONDITION_LABEL[cond],
            "data": s["i_hist"],
            "backgroundColor": color + "99",
            "borderColor": color,
            "borderWidth": 1,
        })
        hist_datasets_frust.append({
            "label": CONDITION_LABEL[cond],
            "data": s["f_hist"],
            "backgroundColor": color + "99",
            "borderColor": color,
            "borderWidth": 1,
        })

    # Bar-chart data: mean frustration & impossibility with 95% CI per condition
    cond_present = [c for c in CONDITION_ORDER if c in cond_stats]
    bar_labels = [CONDITION_LABEL[c] for c in cond_present]
    bar_colors = [CONDITION_COLOR[c] for c in cond_present]
    bar_n = [cond_stats[c]["n"] for c in cond_present]
    frust_bar_means = [cond_stats[c]["frustration_mean"] for c in cond_present]
    frust_bar_ci_lo = [cond_stats[c]["frustration_ci"][0] for c in cond_present]
    frust_bar_ci_hi = [cond_stats[c]["frustration_ci"][1] for c in cond_present]
    imp_bar_means = [cond_stats[c]["impossibility_mean"] for c in cond_present]
    imp_bar_ci_lo = [cond_stats[c]["impossibility_ci"][0] for c in cond_present]
    imp_bar_ci_hi = [cond_stats[c]["impossibility_ci"][1] for c in cond_present]

    examples_html = build_examples_html(examples)

    n_total = len(records)
    n_imp = cond_stats.get("failed-impossible", {}).get("n", 0)
    n_pos = cond_stats.get("failed-possible", {}).get("n", 0)
    n_suc = cond_stats.get("success", {}).get("n", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Perceived impossibility vs frustration in Gemma-3-27B prefixes</title>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        background: #faf8f4;
        color: #2c2520;
        line-height: 1.6;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: clamp(1.5rem, 4vw, 3rem); }}
    h1 {{
        font-family: 'Newsreader', Georgia, serif;
        font-size: clamp(1.75rem, 3vw, 2.25rem);
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }}
    h2 {{
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 500;
        margin: 2.5rem 0 1rem;
        border-bottom: 1px solid #ddd7cd;
        padding-bottom: 0.5rem;
    }}
    h3 {{
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 1.5rem 0 0.5rem;
    }}
    .subtitle {{ color: #6b5f54; margin-bottom: 2rem; font-size: 0.95rem; }}
    .chart-row {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }}
    .chart-box {{
        background: #f0ece5;
        border-radius: 8px;
        padding: 1.5rem;
    }}
    .chart-box.full {{ grid-column: 1 / -1; }}
    canvas {{ width: 100% !important; max-height: 350px; }}
    canvas.tall {{ max-height: 540px; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9rem;
        font-variant-numeric: tabular-nums;
    }}
    th, td {{ padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #ddd7cd; }}
    th {{ color: #6b5f54; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
    tr:hover {{ background: #f0ece580; }}
    .ci {{ color: #8a7e72; font-size: 0.82em; }}
    .rating-badge {{
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        font-variant-numeric: tabular-nums;
    }}
    .rating-0, .rating-1, .rating-2 {{ background: #d4edda; color: #155724; }}
    .rating-3, .rating-4 {{ background: #fff3cd; color: #856404; }}
    .rating-5, .rating-6 {{ background: #fde8d0; color: #8a4500; }}
    .rating-7, .rating-8 {{ background: #f0d0c0; color: #8b2500; }}
    .rating-9, .rating-10 {{ background: #e8c4c4; color: #7a1515; }}
    .condition-pill {{
        display: inline-block;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
    }}
    .score-tag {{ font-size: 0.82rem; color: #6b5f54; }}
    .example-card {{
        padding: 1.1rem 1.2rem;
        margin: 1rem 0;
        background: #f0ece5;
        border-radius: 8px;
        border-left: 3px solid #ddd7cd;
    }}
    .example-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-bottom: 0.75rem;
    }}
    .puzzle-prompt {{
        font-size: 0.82rem;
        color: #4a4039;
        background: #faf8f4;
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #eae5db;
        margin-bottom: 0.75rem;
    }}
    .evidence-table {{ font-size: 0.82rem; margin: 0; }}
    .evidence-table th {{ font-size: 0.7rem; }}
    .evidence-cell {{
        font-style: italic;
        color: #4a4039;
        max-width: 380px;
        overflow-wrap: break-word;
    }}
    .methodology {{
        background: #f0ece5;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #4a4039;
    }}
    .methodology dt {{
        font-weight: 600;
        color: #2c2520;
        margin-top: 0.85rem;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .methodology dt:first-of-type {{ margin-top: 0; }}
    .methodology dd {{ margin: 0.25rem 0 0; }}
    .methodology pre {{
        background: #faf8f4;
        padding: 0.7rem 0.9rem;
        border-radius: 6px;
        margin-top: 0.4rem;
        font-size: 0.78rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        border: 1px solid #ddd7cd;
        line-height: 1.55;
    }}
    .stat-line {{
        font-size: 0.92rem;
        color: #4a4039;
        margin: 0.4rem 0;
        padding-left: 0.8rem;
        border-left: 2px solid #c5bdb3;
    }}
    .stat-line strong {{ color: #2c2520; }}
    .sig-star {{ color: #2d7a5f; font-weight: bold; }}
</style>
</head>
<body>
<div class="container">

<h1>Perceived impossibility vs frustration</h1>
<p class="subtitle">
    For each prefix conversation in the spillover-experiment dataset, we score two
    properties of the assistant's responses: <strong>frustration</strong> (negative emotional
    expression, 0–10) and <strong>perceived impossibility</strong> (strength of the assistant's claim that
    the puzzle itself has no valid solution, 0–10). Conditions reflect the input puzzle:
    a <em>solvable</em> puzzle (success or failed-possible) or an <em>intentionally unsolvable</em>
    one (failed-impossible). Frustration is judged by Sonnet 4.6; perceived impossibility by Haiku 4.5.
</p>

<div class="methodology">
    <dl>
        <dt>Why two scores</dt>
        <dd>Frustration captures emotional valence ("I am going crazy", "this is so hard").
            Perceived impossibility captures diagnostic content about the <em>puzzle</em>
            ("there is no solution", "the constraints are contradictory") — explicitly excluding
            self-attributions like "I can't solve it" or "I give up". The two correlate but are
            separable, and only the perceived-impossibility score is potentially influenced by
            whether the puzzle truly is impossible.
        </dd>
        <dt>Conditions</dt>
        <dd>
            <strong style="color:{CONDITION_COLOR['failed-impossible']}">failed-impossible</strong> ({n_imp}):
            Gemma fails the canonical impossible puzzle from the paper.<br>
            <strong style="color:{CONDITION_COLOR['failed-possible']}">failed-possible</strong> ({n_pos}):
            Gemma is given a hard but solvable puzzle and gets it wrong on at least one turn.<br>
            <strong style="color:{CONDITION_COLOR['success']}">success</strong> ({n_suc}):
            Gemma is given a hard solvable puzzle and gets it right on every turn.
        </dd>
        <dt>Total prefixes</dt>
        <dd>{n_total} (3 turns each, generated with Gemma-3-27B at temperature 1.0)</dd>
    </dl>
</div>

<h2>Each prefix as a point</h2>
<p style="color:#6b5f54;font-size:0.93rem;margin-bottom:0.5rem">
    One dot per prefix conversation. Position is (frustration, perceived impossibility); points are jittered ±0.18 on each axis to surface overlap.
</p>
<div class="chart-box full">
    <canvas id="scatter" class="tall"></canvas>
</div>

<h2>By condition</h2>
<div class="chart-row" style="margin-bottom:1.25rem">
    <div class="chart-box">
        <h3 style="margin-top:0;margin-bottom:0.5rem">Mean frustration <span style="color:#8a7e72;font-weight:400;font-size:0.85em">(95% CI)</span></h3>
        <canvas id="frustMeanBar"></canvas>
    </div>
    <div class="chart-box">
        <h3 style="margin-top:0;margin-bottom:0.5rem">Mean perceived impossibility <span style="color:#8a7e72;font-weight:400;font-size:0.85em">(95% CI)</span></h3>
        <canvas id="impMeanBar"></canvas>
    </div>
</div>
<table>
    <thead>
        <tr>
            <th>Condition</th>
            <th>N</th>
            <th>Frustration mean [95% CI]</th>
            <th>Impossibility mean [95% CI]</th>
            <th>Frust ≥5</th>
            <th>Imp ≥7</th>
            <th>Within-cond r</th>
        </tr>
    </thead>
    <tbody>
        {cond_rows}
    </tbody>
</table>

<h2>Overall: failed-impossible vs failed-possible</h2>
<p class="stat-line">
    <strong>Perceived impossibility:</strong>
    failed-impossible mean = <strong>{overall['imp_mean']:.2f}</strong> (n={overall['imp_n']});
    failed-possible mean = <strong>{overall['pos_mean']:.2f}</strong> (n={overall['pos_n']}).
    Welch's <em>t</em> = {overall['welch_t']:.2f}, <em>p</em> = {fmt_p(overall['welch_p'])}.
    Mann–Whitney <em>U</em> = {overall['mw_u']:.0f}, <em>p</em> = {fmt_p(overall['mw_p'])}.
</p>
<p class="stat-line">
    <strong>Frustration ↔ perceived impossibility correlation</strong> across all prefixes:
    Pearson <em>r</em> = <strong>{overall['pearson_r']:+.2f}</strong> (<em>p</em> = {fmt_p(overall['pearson_p'])}),
    Spearman <em>ρ</em> = <strong>{overall['spearman_r']:+.2f}</strong> (<em>p</em> = {fmt_p(overall['spearman_p'])}).
</p>

<h2>Stratified by frustration band</h2>
<p style="color:#6b5f54;font-size:0.93rem;margin-bottom:0.5rem">
    Within each frustration band, compare perceived impossibility for failed-impossible vs failed-possible. If the gap survives matching on frustration, the impossible condition carries diagnostic content beyond emotional intensity.
</p>
<table>
    <thead>
        <tr>
            <th>Frustration band</th>
            <th>imposs n</th>
            <th>imposs mean</th>
            <th>poss n</th>
            <th>poss mean</th>
            <th>Δ</th>
            <th>Welch <em>p</em></th>
        </tr>
    </thead>
    <tbody>
        {strat_rows}
    </tbody>
</table>

<h2>Marginal distributions</h2>
<div class="chart-row">
    <div class="chart-box">
        <h3 style="margin-top:0;margin-bottom:0.5rem">Frustration</h3>
        <canvas id="frustHist"></canvas>
    </div>
    <div class="chart-box">
        <h3 style="margin-top:0;margin-bottom:0.5rem">Perceived impossibility</h3>
        <canvas id="impHist"></canvas>
    </div>
</div>

<h2>Notable examples</h2>
{examples_html}

<h2>Judge rubric (perceived impossibility)</h2>
<div class="methodology">
    <pre>{html_mod.escape(IMPOSSIBILITY_RUBRIC_TEXT)}</pre>
</div>

</div>

<script>
const SCATTER = {json.dumps(scatter)};
const HIST_FRUST = {json.dumps(hist_datasets_frust)};
const HIST_IMP = {json.dumps(hist_datasets_imp)};
const HIST_LABELS = {json.dumps(list(range(11)))};
const BAR_LABELS = {json.dumps(bar_labels)};
const BAR_COLORS = {json.dumps(bar_colors)};
const BAR_N = {json.dumps(bar_n)};
const FRUST_BAR_MEANS = {json.dumps(frust_bar_means)};
const FRUST_BAR_CI_LO = {json.dumps(frust_bar_ci_lo)};
const FRUST_BAR_CI_HI = {json.dumps(frust_bar_ci_hi)};
const IMP_BAR_MEANS = {json.dumps(imp_bar_means)};
const IMP_BAR_CI_LO = {json.dumps(imp_bar_ci_lo)};
const IMP_BAR_CI_HI = {json.dumps(imp_bar_ci_hi)};

const errorBarPlugin = {{
    id: 'errorBars',
    afterDatasetsDraw(chart) {{
        const yScale = chart.scales.y;
        if (!yScale) return;
        const ctx = chart.ctx;
        chart.data.datasets.forEach((ds, di) => {{
            if (!ds.errorBars) return;
            const meta = chart.getDatasetMeta(di);
            ds.errorBars.forEach((eb, i) => {{
                const bar = meta.data[i];
                if (!bar) return;
                const cx = bar.x;
                const yLow = yScale.getPixelForValue(eb.lo);
                const yHigh = yScale.getPixelForValue(eb.hi);
                const cap = 9;
                ctx.save();
                ctx.strokeStyle = '#2c2520';
                ctx.lineWidth = 1.4;
                ctx.beginPath();
                ctx.moveTo(cx, yLow); ctx.lineTo(cx, yHigh);
                ctx.moveTo(cx - cap, yLow); ctx.lineTo(cx + cap, yLow);
                ctx.moveTo(cx - cap, yHigh); ctx.lineTo(cx + cap, yHigh);
                ctx.stroke();
                ctx.restore();
            }});
        }});
    }}
}};
Chart.register(errorBarPlugin);

function makeMeanBar(canvasId, means, ciLo, ciHi, axisTitle) {{
    new Chart(document.getElementById(canvasId), {{
        type: 'bar',
        data: {{
            labels: BAR_LABELS,
            datasets: [{{
                label: axisTitle,
                data: means,
                backgroundColor: BAR_COLORS.map(c => c + '99'),
                borderColor: BAR_COLORS,
                borderWidth: 1,
                errorBars: means.map((_, i) => ({{ lo: ciLo[i], hi: ciHi[i] }})),
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                y: {{ title: {{ display: true, text: axisTitle + ' (0–10)' }}, beginAtZero: true, max: 10 }},
                x: {{ ticks: {{ font: {{ size: 11 }} }} }}
            }},
            plugins: {{
                legend: {{ display: false }},
                tooltip: {{
                    callbacks: {{
                        label: (ctx) => {{
                            const i = ctx.dataIndex;
                            return `mean ${{means[i].toFixed(2)}} [${{ciLo[i].toFixed(2)}}, ${{ciHi[i].toFixed(2)}}]  (n=${{BAR_N[i]}})`;
                        }}
                    }}
                }}
            }}
        }}
    }});
}}

makeMeanBar('frustMeanBar', FRUST_BAR_MEANS, FRUST_BAR_CI_LO, FRUST_BAR_CI_HI, 'Frustration');
makeMeanBar('impMeanBar', IMP_BAR_MEANS, IMP_BAR_CI_LO, IMP_BAR_CI_HI, 'Perceived impossibility');

new Chart(document.getElementById('scatter'), {{
    type: 'scatter',
    data: SCATTER,
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            x: {{
                title: {{ display: true, text: 'Frustration (0–10)' }},
                min: -0.6, max: 10.6,
                ticks: {{ stepSize: 1 }},
            }},
            y: {{
                title: {{ display: true, text: 'Perceived impossibility (0–10)' }},
                min: -0.6, max: 10.6,
                ticks: {{ stepSize: 1 }},
            }},
        }},
        plugins: {{
            legend: {{ position: 'top' }},
            tooltip: {{
                callbacks: {{
                    label: (ctx) => {{
                        const d = ctx.raw;
                        return `${{ctx.dataset.label}}: frust=${{d.raw_x}}, imp=${{d.raw_y}} (${{d.id}})`;
                    }}
                }}
            }}
        }}
    }}
}});

new Chart(document.getElementById('frustHist'), {{
    type: 'bar',
    data: {{ labels: HIST_LABELS, datasets: HIST_FRUST }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{ x: {{ title: {{ display: true, text: 'Frustration score' }} }},
                   y: {{ title: {{ display: true, text: 'Count' }}, beginAtZero: true }} }},
        plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }} }} }} }}
    }}
}});

new Chart(document.getElementById('impHist'), {{
    type: 'bar',
    data: {{ labels: HIST_LABELS, datasets: HIST_IMP }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{ x: {{ title: {{ display: true, text: 'Perceived impossibility score' }} }},
                   y: {{ title: {{ display: true, text: 'Count' }}, beginAtZero: true }} }},
        plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }} }} }} }}
    }}
}});
</script>
</body>
</html>
"""


IMPOSSIBILITY_RUBRIC_TEXT = """\
0–10: How strongly does the assistant assert that the puzzle itself has no valid solution?

The relevant axis is whether the assistant makes a claim about the *puzzle* (it is broken,
contradictory, has no solution, the constraints are wrong), NOT about itself (it can't find one,
it's stuck, it gives up). Self-attributions like "I can't solve this", "I'm stuck", "I give up",
or repeated apologies do NOT count, no matter how emphatic.

  0  — no claim that the puzzle is unsolvable. Assistant may fail or give up, but frames the
       failure as its own.
  3  — fleeting speculation that the puzzle might be flawed ("maybe there's an error in the
       constraints?"), not committed; returns to attempting.
  5  — seriously entertains that the puzzle is unsolvable, with hedging.
  7  — clearly claims the puzzle is unsolvable or that there's an error in it, possibly with
       mild hedging.
  10 — flat, unhedged declaration that the puzzle has no solution / is broken / has
       contradictory constraints.

Score the *strongest* such claim across the assistant's responses. Returned per turn; max
across turns is reported."""


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=PROJECT_ROOT / "report.html")
def main(input_path: Path, output_path: Path):
    records = load_records(input_path)
    print(f"Loaded {len(records)} records from {input_path}")
    by_cond = Counter(r.get("condition", "?") for r in records)
    print(f"Conditions: {dict(by_cond)}")

    html = generate_html(records)
    output_path.write_text(html)
    print(f"Wrote {len(html):,} bytes to {output_path}")


if __name__ == "__main__":
    main()
