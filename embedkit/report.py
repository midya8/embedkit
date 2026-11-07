"""Terminal, JSON and Markdown reports for embedding evaluations."""

from __future__ import annotations

import json


def build_report(coherence_result: dict, separation_result: dict, params: dict) -> dict:
    return {
        "params": params,
        "coherence": coherence_result,
        "separation": separation_result,
    }


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def to_markdown(report: dict) -> str:
    coh, sep = report["coherence"], report["separation"]
    lines = [
        "# EmbedKit report",
        "",
        f"- Words evaluated: {coh['words']} (k={coh['k']})",
        f"- Mean coherence: {coh['mean_coherence']}",
        f"- Mean intra-cluster similarity: {sep['mean_intra']}",
        f"- Mean inter-cluster similarity: {sep['mean_inter']}",
        f"- Separation margin: {sep['margin']}",
        "",
        "| Word | Cluster | Coherence | Nearest |",
        "|---|---|---|---|",
    ]
    for word, detail in sorted(coh["per_word"].items()):
        lines.append(
            f"| {word} | {detail['cluster']} | {detail['coherence']:.4f} "
            f"| {', '.join(detail['nearest'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def to_terminal(report: dict) -> str:
    coh, sep = report["coherence"], report["separation"]
    lines = [
        f"words={coh['words']} k={coh['k']} mean_coherence={coh['mean_coherence']}",
        f"intra={sep['mean_intra']} inter={sep['mean_inter']} margin={sep['margin']}",
    ]
    for word, detail in sorted(coh["per_word"].items()):
        lines.append(f"  {word:<14} {detail['cluster']:<10} {detail['coherence']:.2f}  {', '.join(detail['nearest'])}")
    return "\n".join(lines)
