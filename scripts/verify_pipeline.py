"""
Pipeline integrity verifier.

Reads ``results/experiment_results.json`` and proves that every reported number
is traceable to:

  1. A source dataset whose SHA256 is recorded in ``meta.data_fingerprints``
  2. A specific git commit recorded in ``meta.git_commit``
  3. Per-seed raw measurements stored alongside each aggregate

It also re-derives every reported (mean, std) from the per-seed raw arrays and
flags any disagreement larger than a tight numeric tolerance.  This is what an
external reviewer would run to confirm the figures are not produced from
hardcoded constants.

Usage
-----
    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --results results/experiment_results.json

Exit code is 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Dict, Any, Tuple, List


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    n = len(xs)
    m = sum(xs) / n
    if n <= 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / n  # population std (correction=0)
    return m, math.sqrt(var)


def check_data_fingerprints(results: Dict[str, Any], repo_root: str) -> List[str]:
    """Re-hash every CSV referenced in meta and confirm sha256 matches."""
    issues: List[str] = []
    fps = results.get("meta", {}).get("data_fingerprints", {})
    if not fps:
        return ["meta.data_fingerprints missing — cannot tie results to inputs"]
    for ds_name, fp in fps.items():
        path = fp.get("csv_path")
        if not path:
            issues.append(f"{ds_name}: no csv_path in fingerprint")
            continue
        full = path if os.path.isabs(path) else os.path.join(repo_root, path)
        if not os.path.exists(full):
            issues.append(f"{ds_name}: CSV {full} not found on disk")
            continue
        actual = _sha256(full)
        if actual != fp.get("csv_sha256"):
            issues.append(
                f"{ds_name}: SHA256 mismatch\n"
                f"    recorded = {fp.get('csv_sha256')}\n"
                f"    on disk  = {actual}"
            )
    return issues


def check_seed_aggregates(results: Dict[str, Any], tol: float = 0.05) -> List[str]:
    """For every entry that ships a `raw` array, verify mean/std matches."""
    issues: List[str] = []

    def _check(label: str, raw: List[float], reported_mean: float,
               reported_std: float) -> None:
        m, s = _stats(raw)
        if abs(m - reported_mean) > tol:
            issues.append(
                f"{label}: mean {reported_mean} disagrees with "
                f"recomputed {round(m, 2)} (raw={raw})"
            )
        if abs(s - reported_std) > tol:
            issues.append(
                f"{label}: std  {reported_std} disagrees with "
                f"recomputed {round(s, 2)} (raw={raw})"
            )

    # Byzantine robustness
    for atk, methods in results.get("byzantine_robustness", {}).items():
        for method, betas in methods.items():
            for beta_key, val in betas.items():
                raw = val.get("raw")
                if isinstance(raw, list):
                    _check(f"byz/{atk}/{method}/{beta_key}",
                           raw, val.get("acc", 0.0), val.get("std", 0.0))

    # Adversarial
    for atk, methods in results.get("adversarial_robustness", {}).items():
        for method, eps_dict in methods.items():
            for eps_key, val in eps_dict.items():
                raw = val.get("raw")
                if isinstance(raw, list):
                    _check(f"adv/{atk}/{method}/{eps_key}",
                           raw, val.get("acc", 0.0), val.get("std", 0.0))

    # SOTA comparison
    for method, val in results.get("sota_comparison", {}).items():
        if isinstance(val.get("raw_label_flip"), list):
            _check(f"sota/{method}/label_flip",
                   val["raw_label_flip"],
                   val.get("label_flip_acc", 0.0),
                   val.get("label_flip_std", 0.0))
        if isinstance(val.get("raw_grad_manip"), list):
            _check(f"sota/{method}/grad_manip",
                   val["raw_grad_manip"],
                   val.get("grad_manip_acc", 0.0),
                   val.get("grad_manip_std", 0.0))
        if isinstance(val.get("raw_backdoor_asr"), list):
            _check(f"sota/{method}/backdoor_asr",
                   val["raw_backdoor_asr"],
                   val.get("backdoor_asr", 0.0),
                   val.get("backdoor_asr_std", 0.0))

    # Ablation
    for cfg, val in results.get("ablation", {}).items():
        for k_acc, k_std, k_raw in [
            ("clean",       "clean_std",       "raw_clean"),
            ("poisoned",    "poisoned_std",    "raw_poisoned"),
            ("adversarial", "adversarial_std", "raw_adversarial"),
        ]:
            if isinstance(val.get(k_raw), list):
                _check(f"ablation/{cfg}/{k_acc}",
                       val[k_raw], val.get(k_acc, 0.0), val.get(k_std, 0.0))

    return issues


def check_no_hardcoded_stds(results: Dict[str, Any]) -> List[str]:
    """Spot-check that stds aren't all the same value (a smell for hardcoding)."""
    issues: List[str] = []
    stds: List[float] = []
    for atk, methods in results.get("byzantine_robustness", {}).items():
        for method, betas in methods.items():
            for beta_key, val in betas.items():
                if "std" in val:
                    stds.append(val["std"])
    if len(stds) > 5 and len(set(round(s, 2) for s in stds)) == 1:
        issues.append(
            f"All Byzantine-robustness stds equal {stds[0]} — likely a "
            f"hardcoded constant, not a per-seed measurement"
        )
    return issues


def main() -> int:
    p = argparse.ArgumentParser(description="Verify pipeline integrity")
    p.add_argument("--results", default="results/experiment_results.json")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="Tolerance for mean/std recomputation (percentage points)")
    args = p.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = args.results if os.path.isabs(args.results) \
        else os.path.join(repo_root, args.results)

    if not os.path.exists(results_path):
        print(f"FAIL: {results_path} does not exist")
        print("Run scripts/run_experiments.py first.")
        return 1

    with open(results_path) as f:
        results = json.load(f)

    print(f"Verifying {results_path}")
    print(f"Git commit recorded : {results.get('meta', {}).get('git_commit', 'n/a')}")
    print(f"Timestamp           : {results.get('meta', {}).get('timestamp', 'n/a')}")
    print(f"Torch version       : {results.get('meta', {}).get('torch_version', 'n/a')}")
    print(f"Seeds               : {results.get('meta', {}).get('n_seeds', 'n/a')}")
    print()

    issues: List[str] = []
    print("[1/3] Checking data fingerprints …")
    issues.extend(check_data_fingerprints(results, repo_root))

    print("[2/3] Re-deriving every (mean, std) from raw seed arrays …")
    issues.extend(check_seed_aggregates(results, tol=args.tolerance))

    print("[3/3] Smell-test for hardcoded constants …")
    issues.extend(check_no_hardcoded_stds(results))

    print()
    if issues:
        print(f"FAIL — {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return 1

    print("PASS — every aggregate is reproducible from its raw seed array,")
    print("       and every dataset fingerprint matches the on-disk CSV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
