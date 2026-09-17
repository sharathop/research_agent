"""
aggregate_results.py

Run this AFTER stopping research_agent_fixed.py partway through a batch
(Ctrl+C, closed terminal, crash, etc.) to reconstruct the two files that
main() would otherwise only write once the full batch finishes:

    - all_results.json
    - needs_human_review.json

It does this by reading every per-app file already written to results/
(each app writes its own file the moment it finishes, independent of
the batch-level aggregation), so nothing is lost even if the run never
reaches the end of main().

Usage:
    Place this file in the SAME directory as research_agent_fixed.py
    (so that ./results/ is a sibling folder), then run:

        python aggregate_results.py

Note: run_summary.json's token_usage totals are NOT recoverable this
way - that counter (TOKEN_USAGE) only ever lived in memory and is lost
when the process is killed before main() finishes. Everything else
(per-app status, errors, needs_human_review flags, reports) is fully
recovered.
"""

import json
import os
import glob
import sys

RESULTS_DIR = "results"


def main():
    if not os.path.isdir(RESULTS_DIR):
        sys.exit(
            f"No '{RESULTS_DIR}/' folder found in the current directory.\n"
            f"Run this script from the same folder as research_agent_fixed.py."
        )

    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not paths:
        sys.exit(f"'{RESULTS_DIR}/' exists but contains no .json files - nothing to aggregate.")

    records = []
    skipped = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((path, str(e)))

    with open("all_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    needs_review = [
        {
            "app": r.get("app", "unknown"),
            "category": r.get("category", "unknown"),
            "status": r.get("status", "unknown"),
            "reason": r.get("error") or "verification contradiction/inconclusive",
        }
        for r in records
        if r.get("needs_human_review")
    ]
    with open("needs_human_review.json", "w", encoding="utf-8") as f:
        json.dump(needs_review, f, indent=2, ensure_ascii=False)

    status_counts = {}
    for r in records:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"Read {len(paths)} file(s) from {RESULTS_DIR}/")
    if skipped:
        print(f"  Skipped {len(skipped)} unreadable file(s):")
        for p, err in skipped:
            print(f"    - {p}: {err}")
    print(f"Recovered {len(records)} app result(s) -> all_results.json")
    print(f"  Status breakdown: {status_counts}")
    print(f"Flagged for human review: {len(needs_review)} -> needs_human_review.json")
    print()
    print("Note: run_summary.json's token_usage totals are not recoverable this way")
    print("(that counter only ever lived in memory). Per-app counts above are accurate.")


if __name__ == "__main__":
    main()