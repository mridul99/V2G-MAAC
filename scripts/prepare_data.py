#!/usr/bin/env python3
"""
prepare_data.py
------------------
Adapter between the Mendeley Data release (7 per-variant CSVs: main.csv,
seed_101.csv, seed_103.csv, seed_104.csv, adversarial.csv, noisy_benign.csv,
sparse_feedback.csv -- each already train+test combined for that variant,
with a `split` column) and what every notebook in this repo expects to load:
ONE combined CSV across all 7 variants, pointed to via the V2G_COMBINED_CSV
environment variable.

WHY THIS IS NEEDED, NOT JUST A CONVENIENCE: the dataset release on Mendeley
Data does not ship v2g_benchmark_multi_world.csv (the generator's raw,
pre-label, all-variants-at-once output) -- only the already-labeled,
already-split per-variant files (see the dataset's own README_dataset.md /
datasheet.md, "Contents"). Concatenating them back into one combined frame
is the one adapter step needed before any notebook/script in this repo can
run against the publicly released data.

IS THIS SAFE -- DOES IT CHANGE ANY RESULT? No. Every notebook/script here
calls benchmark_labels.add_ground_truth_labels() and ml_features.prepare_ml_frame()
itself as its first step, and both are pure, deterministic functions of
columns ALREADY present in the released CSVs: gt_cpo_malicious is recomputed
from authority_malicious, gt_emsp_malicious from emsp_malicious_eligible,
gt_ev_sybil from ev_sybil, ev_session_share_so_far from connection_time/ev_id
grouped by (variant, world_id). Recomputing them on data where they already
happen to match is a harmless no-op overwrite, not a second source of truth.
This script does the minimum necessary job -- read the 7 files, concatenate,
write one CSV -- and does not touch any feature value. Per-variant row
counts it prints should match the dataset's own package_release.py
EXPECTED_ROWS_PER_VARIANT=125,110 exactly; if they don't, something about
the downloaded files is wrong, and this script says so rather than
proceeding quietly.

USAGE:
    python scripts/prepare_data.py --data-dir /path/to/downloaded/mendeley/release

Writes <data-dir>/v2g_benchmark_combined.csv by default (override with
--out). Then point any notebook/script in this repo at it, e.g.:

    export V2G_COMBINED_CSV=/path/to/downloaded/mendeley/release/v2g_benchmark_combined.csv

Transparently handles both single-file variants (main.csv) and the
MANIFEST.sha256-documented split convention (main_part1of3.csv,
main_part2of3.csv, ...), and verifies every downloaded file against
MANIFEST.sha256 first when that file is present (skip with
--skip-manifest-check).
"""
import argparse
import glob
import hashlib
import os
import re

import pandas as pd

VARIANTS = [
    "main", "seed_101", "seed_103", "seed_104",
    "adversarial", "noisy_benign", "sparse_feedback",
]

EXPECTED_ROWS_PER_VARIANT = 125_110  # must match the dataset's own package_release.py


def _variant_files(data_dir: str, variant: str) -> list:
    """Returns the file(s) making up one variant, in part order: either
    [<variant>.csv] or [<variant>_part1ofN.csv, ..., <variant>_partNofN.csv]."""
    single = os.path.join(data_dir, f"{variant}.csv")
    if os.path.exists(single):
        return [single]

    parts = glob.glob(os.path.join(data_dir, f"{variant}_part*of*.csv"))
    if not parts:
        raise FileNotFoundError(
            f"Could not find '{variant}.csv' or '{variant}_part*of*.csv' in {data_dir}. "
            f"Download every file listed in the dataset's README_dataset.md into one "
            f"folder and pass that folder as --data-dir."
        )

    def part_num(path):
        m = re.search(r"_part(\d+)of(\d+)\.csv$", path)
        if not m:
            raise ValueError(f"Unexpected part filename, doesn't match '_partNofM.csv': {path}")
        return int(m.group(1))

    parts.sort(key=part_num)
    return parts


def load_variant(data_dir: str, variant: str) -> pd.DataFrame:
    files = _variant_files(data_dir, variant)
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "variant" in df.columns:
        unexpected = set(df["variant"].unique()) - {variant}
        if unexpected:
            raise ValueError(
                f"{variant}'s file(s) contain unexpected variant value(s) {unexpected} -- "
                f"check these are the right files for this variant."
            )
    return df


def verify_manifest(data_dir: str) -> None:
    manifest_path = os.path.join(data_dir, "MANIFEST.sha256")
    if not os.path.exists(manifest_path):
        print(f"NOTE: no MANIFEST.sha256 found in {data_dir} -- skipping integrity check.")
        return
    print(f"Verifying downloaded files against {manifest_path} ...")
    ok = True
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            expected_hash, name = line.split("  ", 1)
            path = os.path.join(data_dir, name)
            if not os.path.exists(path):
                print(f"  MISSING: {name}")
                ok = False
                continue
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_hash:
                print(f"  MISMATCH: {name} (corrupted/incomplete download -- re-download it)")
                ok = False
    print("  all files OK" if ok else "  one or more files failed verification -- see above")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", required=True,
                     help="folder holding the downloaded Mendeley release files")
    ap.add_argument("--out", default=None,
                     help="output path (default: <data-dir>/v2g_benchmark_combined.csv)")
    ap.add_argument("--skip-manifest-check", action="store_true",
                     help="skip MANIFEST.sha256 verification")
    args = ap.parse_args()

    data_dir = args.data_dir
    out_path = args.out or os.path.join(data_dir, "v2g_benchmark_combined.csv")

    if not args.skip_manifest_check:
        verify_manifest(data_dir)

    print(f"\n{'Variant':<18}{'Rows':>10}  Source file(s)")
    print("-" * 70)
    frames = []
    all_ok = True
    for variant in VARIANTS:
        df = load_variant(data_dir, variant)
        files = _variant_files(data_dir, variant)
        ok = len(df) == EXPECTED_ROWS_PER_VARIANT
        all_ok = all_ok and ok
        flag = "" if ok else f"  <-- expected {EXPECTED_ROWS_PER_VARIANT:,}"
        print(f"{variant:<18}{len(df):>10,}  {', '.join(os.path.basename(f) for f in files)}{flag}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print("-" * 70)
    print(f"Combined: {len(combined):,} rows, {combined.shape[1]} columns  "
          f"(row counts as expected: {all_ok})")

    combined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("\nPoint the notebooks/scripts in this repo at it, e.g.:")
    print(f"  export V2G_COMBINED_CSV={out_path}")


if __name__ == "__main__":
    main()
