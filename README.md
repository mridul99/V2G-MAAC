# V2G-MAAC

**A semi-synthetic benchmark dataset for multi-authority access control in vehicle-to-grid charging networks** — evaluation and reference-baseline code.

[![Dataset DOI](https://img.shields.io/badge/dataset-10.17632%2F4dfgtkgsb5.1-blue)](https://doi.org/10.17632/4dfgtkgsb5.1)
Code: MIT · Data: CC BY 4.0

V2G-MAAC is a semi-synthetic, session-level log of EV charging/roaming activity built on real charging-session structure from ACN-Data (Lee, Li & Low, 2019) and extended with three independently injected, ground-truth threat axes spanning the three authorities in a V2G session — the Charge Point Operator (CPO), the e-Mobility Service Provider (eMSP), and the Electric Vehicle (EV) itself — plus a fourth, parallel identity-delegation axis. Every session carries:

- whether it is malicious, and along which of the three independent threat axes (CPO billing fraud, eMSP credential compromise, EV Sybil identity reuse);
- whether the acting identity differs from the session's home identity (station sub-contracting, roaming-credential delegation, account delegation) — the fourth, orthogonal delegation axis;
- a day-blocked temporal train/test split, reconstructable directly from the released `split` column;
- seven variants sharing one generator configuration: a reference build (`main`), three independent reseeds, and three stress conditions (`adversarial`, `noisy_benign`, `sparse_feedback`).

The **data** are on Mendeley Data: <https://doi.org/10.17632/4dfgtkgsb5.1>. This repository holds the baseline and evaluation code used in the paper, run against that released data. **The generator itself is not released here or on Mendeley Data** — the full generation methodology (threat-injection parameters, onset-gating design, the full six-axis grounding ledger) is described in the companion paper; this repository starts from the already-released, already-labeled data and reproduces every baseline result reported from there.

## Repository contents

| Path | Contents |
|---|---|
| `notebooks/01_prepare_labels_splits.ipynb` | How the ground-truth labels and the day-blocked temporal split were derived (re-runnable against the released data via `scripts/prepare_data.py`; reproduces the released `split` column exactly) |
| `notebooks/02_rbac_abac_baselines.ipynb` | RBAC (rule-based, structurally blind) and ABAC (naive per-authority trust, single- vs multi-authority) baselines |
| `notebooks/03_ml_baseline.ipynb` | Logistic regression + random forest |
| `notebooks/04_dl_baseline.ipynb` | Feedforward neural network (MLP) |
| `notebooks/05_rl_baseline.ipynb` | Contextual-bandit DQN under an explicit asymmetric reward model |
| `notebooks/06_graph_structural_baseline.ipynb` | 2-hop heterogeneous mean-aggregation GNN over the CPO–eMSP–EV session graph |
| `notebooks/07_transformer_entity_attention.ipynb` | Transformer-over-entity-history proof of concept, with an entity-only ablation diagnostic |
| `notebooks/08_delegation_check.ipynb` | The fourth, parallel delegation axis's blind-spot demonstration (exploratory, not a fully evaluated ladder rung — see `docs/BASELINES.md`) |
| `scripts/prepare_data.py` | Concatenates the 7 downloaded Mendeley release CSVs into the single combined file every notebook/script here expects |
| `docs/BASELINES.md` | The baseline ladder, evaluation protocol, and known caveats/findings, one level more detail than this README |
| `requirements.txt` | Python packages |

Every `%%writefile`-based shared module (`decision_engine.py`, `benchmark_labels.py`, `temporal_split.py`, `ml_features.py`, `naive_trust.py`, `delegation_baseline.py`) is written out identically by more than one notebook — each copy is byte-for-byte the same; keeping them inline per notebook (rather than factoring out a separate package) is deliberate, so every notebook stays independently runnable top to bottom with nothing else to install from this repo.

## Getting the data

1. Download every file from the dataset's Mendeley Data listing (<https://doi.org/10.17632/4dfgtkgsb5.1>) into one folder, e.g. `data/`. If a variant was too large to upload as one file it ships as `<variant>_part1ofN.csv` … `_partNofN.csv` — leave the parts as they are, the next step reassembles them.
2. Verify the download (optional but recommended): `sha256sum -c MANIFEST.sha256` inside `data/`.
3. Build the combined file every notebook/script here expects:
   ```bash
   pip install -r requirements.txt
   python scripts/prepare_data.py --data-dir data/
   export V2G_COMBINED_CSV=data/v2g_benchmark_combined.csv
   ```
   `prepare_data.py` concatenates the 7 per-variant release files into one frame (see its own docstring for why this is safe — every notebook here recomputes ground-truth labels and derived features itself regardless, so this step touches no feature value, only row layout) and prints a per-variant row-count sanity check (each variant should show 125,110 rows).

Each notebook also reads `V2G_<BASELINE>_OUT_DIR` (e.g. `V2G_ML_OUT_DIR`, `V2G_DL_OUT_DIR`, `V2G_RL_OUT_DIR`, `V2G_GRAPH_OUT_DIR`, `V2G_TRANSFORMER_OUT_DIR`, `V2G_RBAC_ABAC_OUT_DIR`, `V2G_LABELS_SPLITS_OUT_DIR`, `V2G_WORLDS_OUT_DIR`) for where to write checkpoints/predictions/figures; set it to a local folder, e.g. `export V2G_ML_OUT_DIR=outputs/ml_baseline`. Every one of these defaults to a Colab Drive path if left unset — set the env var rather than editing the notebook.

**`notebooks/01_prepare_labels_splits.ipynb` needs the full, unreleased combined CSV conceptually** (it demonstrates how the `split` column and the three main ground-truth axes were derived from the generator's raw output) — pointed at the `prepare_data.py`-reconstructed combined file, it reproduces the released `split`/`gt_*` columns exactly, since both are pure functions of columns that are already in the release. It is not needed to run any other notebook; every baseline notebook from `02` onward works directly from `V2G_COMBINED_CSV`.

## Variants

| Variant | Purpose |
|---|---|
| `main` | Reference configuration; every learned baseline is trained on this variant only |
| `seed_101` / `seed_103` / `seed_104` | Three independent reseeds of the same configuration, for distinguishing genuine reproducibility from variant-induced change |
| `adversarial` | Threat-layer parameters shift over time (stationary-attacker assumption violated) |
| `noisy_benign` | Elevated behavioral noise among honest actors |
| `sparse_feedback` | Reduced availability of the running, history-dependent features |

## Reference baselines

RBAC (fixed, objective certificate check — structurally blind to all three threat axes), ABAC (naive per-authority trust, single- vs multi-authority combination), logistic regression, random forest, a feedforward MLP, a contextual-bandit DQN under an explicit asymmetric reward model, and a 2-hop heterogeneous graph-structural model, plus a transformer-over-entity-history proof of concept. Every learned baseline shares the same feature set (`ml_features.ALL_FEATURES` — deliberately excludes raw CPO/eMSP/EV identity, see that module's own bug log for why), is trained once on `main`'s train split and evaluated zero-shot on every variant's own test split, and every baseline's score is passed through the same `decision_engine.trust_weighted_decision()` for the same 3-way PERMIT / CONDITIONAL-PERMIT / DENY decision space. They are simple, generic reference implementations, not tuned state of the art. See `docs/BASELINES.md` for the full ladder, the evaluation protocol, and the documented caveats/findings behind each one (several real pipeline bugs and their fixes are kept in the modules' own docstrings deliberately, not swept away).

## Requirements

Python 3.10+, `pip install -r requirements.txt`. A GPU is recommended for the DL/RL/graph/transformer baselines (PyTorch); everything also runs on the CPU, more slowly.

## Citation

Please cite the dataset (and the article once published):

> Mishra Mridul Kailashchandra, U.P. Rao, P. Chauhan. V2G-MAAC: A Semi-Synthetic Benchmark Dataset for Multi-Authority Access Control in Vehicle-to-Grid Charging Networks [dataset]. Mendeley Data, v1, 2026. https://doi.org/10.17632/4dfgtkgsb5.1

Charging-session structure is derived from ACN-Data (Lee, Z. J., Li, T., & Low, S. H. (2019). ACN-Data: Analysis and Applications of an Open EV Charging Dataset); please cite it as well.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0, distributed on Mendeley Data.
