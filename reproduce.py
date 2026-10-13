"""Re-score AML Sentinel's published agent runs and check every number against the committed results.

No API key and no download needed for this part; it takes under a minute. The agent's traces and
verdicts are in results/ibm/agent_runs/, the scoring code is src/ibm_agent_eval.py and
src/ibm_faithfulness.py, and the agent configuration was frozen at commit 56a2bb5 before the held-out
test. Reproducing the rules and model figures needs the IBM dataset from Kaggle: see REPRODUCE.md.

Usage: python reproduce.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="reproduce-"))


def run(*args, cwd=None):
    """Run one of the repository's own scripts with this Python, quietly."""
    subprocess.run([sys.executable, *map(str, args)], cwd=cwd or ROOT, check=True, stdout=subprocess.DEVNULL)


def differences(a, b, path=""):
    """Every place two JSON values differ, allowing for rounding in the last decimal places."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in one file" for k in set(a) ^ set(b)]
        return out + [d for k in set(a) & set(b) for d in differences(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items against {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= 1e-3 else [f"{path}: {a} against {b}"]
    return [] if a == b else [f"{path}: {str(a)[:40]} against {str(b)[:40]}"]


def check(name, recomputed, published):
    diff = differences(recomputed, published)
    print(f"  {'MATCH    ' if not diff else 'DIFFERENT'} {name}")
    for d in diff[:5]:
        print(f"             {d}")
    return not diff


def rerun_in_place(name, files, *args, cwd=None):
    """For scripts that write next to the published files: run, compare with what was there, then put it back.
    JSON files are compared value by value; other files (such as CSVs) are only restored."""
    before = {f: (ROOT / f).read_text() for f in files}
    try:
        run(*args, cwd=cwd)
        return all([check(f"{name}: {f}", json.loads((ROOT / f).read_text()), json.loads(before[f])) for f in files if f.endswith(".json")])
    finally:
        for f, text in before.items():
            (ROOT / f).write_text(text)


def main():
    print("AML Sentinel: re-scoring the agent's 300 held-out alerts with the repository's own scoring code.\n")
    ok = True
    for name in ("ibm-test-haiku", "ibm-test-sonnet"):
        d = f"results/ibm/agent_runs/{name}"
        ok &= rerun_in_place(f"{name} ranking and decisions", [f"{d}/metrics.json"], "src/ibm_agent_eval.py", "--run", name)
        ok &= rerun_in_place(f"{name} case-file faithfulness", [f"{d}/faithfulness.json", f"{d}/faithfulness_by_alert.csv"],
                             "src/ibm_faithfulness.py", "--run", name)
    h = json.loads((ROOT / "results/ibm/agent_runs/ibm-test-haiku/metrics.json").read_text())
    rk, gap = h["ranking"], h["agent_vs_model"]
    print(f"\n  Ranking AUC on the 300 alerts   agent {rk['agent']['auc']:.2f}   scoring model {rk['model']['auc']:.2f}"
          f"   (gap {gap['auc_gap']:+.2f}, 95% interval {gap['ci95'][0]:+.2f} to {gap['ci95'][1]:+.2f})")
    print("  The rules and scoring-model figures (results/ibm/rules_baseline.json, model_baseline.json) need the")
    print("  IBM dataset from Kaggle; REPRODUCE.md has the steps.")
    print("\nEvery re-scored number matches the published results." if ok else "\nSome numbers differ; see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
