# Baseline ladder, protocol, and known caveats

## Ground truth

Three independent threat axes, each binary, injected independently (`benchmark_labels.py`):

| Field | Axis | Definition |
|---|---|---|
| `gt_cpo_malicious` | CPO | Billing-fraud pattern; **onset-gated** (`authority_malicious`: identical to an honest CPO before `day_idx` reaches the configured onset day) |
| `gt_emsp_malicious` | eMSP | Colluding/compromised credential issuer; **onset-gated**, symmetric with the CPO axis (`emsp_malicious_eligible`) |
| `gt_ev_sybil` | EV | Identity reuse/fabrication (`ev_sybil`); a structural property, not time-gated |

Pooled, for convenience (not independent signal): `risk_factor_count` = sum of the three flags (0–3); `any_risk` = 1[`risk_factor_count` > 0] (the binary target every learned baseline is scored against); `composite_risk_score` = equal-weighted mean across the three axes (`AXIS_WEIGHTS`, documented as a modelling convenience, not a real-world severity claim).

A fourth, parallel axis — identity delegation (`gt_cpo_delegate_malicious`, `gt_emsp_delegate_malicious`, `gt_ev_delegate_abuse`) — is independent of the three above and is **not** part of the main detection ladder; see `08_delegation_check.ipynb` and *Delegation check*, below.

## Protocol

Two different evaluation protocols, depending on whether a baseline has anything to train:

- **RBAC and ABAC** (`02_rbac_abac_baselines.ipynb`, rule-based, nothing learned): RBAC has no calibration at all — a fixed, objective `certificate_valid` check, structurally blind to all three threat axes. ABAC's naive per-authority trust is a causal running statistic computed across each variant's own full chronological sequence (train+test together, since it only ever looks backward); its one calibrated constant, `cpo_trust_scale`, is derived from **each variant's own train split**, not pooled from `main`. Both are evaluated on every variant's own sessions, not a model trained elsewhere.
- **ML, DL, RL, graph-structural, transformer** (learned, `03`–`07`): each is trained **once**, on `main`'s train split only, then scored **zero-shot** on every variant's own held-out test split — no retraining or fine-tuning per variant. This is what makes the seven variants a generalisation/robustness probe for these baselines specifically. Internal validation (for early stopping) is a tail-of-days carve-out from `main`'s own train split (`VAL_FRACTION_OF_TRAIN_DAYS=0.15`), never the test period.

The day-blocked temporal split (`temporal_split.py`): `day_cutoff` is the 70th percentile of unique `day_idx` values (`TRAIN_FRACTION_OF_DAYS=0.7`) — `830` on the released data. Every session strictly before that index is train, every session from it onward is test.

`(session_id, variant, world_id)` is the only safe join key across any two baselines' prediction files — `session_id` is a real, reused ACN-Data identifier and is **not** unique across variants or worlds on its own.

## The ladder

| # | Notebook | Approach | Features | Notes |
|---|---|---|---|---|
| 1 | `01_prepare_labels_splits.ipynb` | — | — | Derives the ground-truth labels and the temporal split; not a baseline itself |
| 2 | `02_rbac_abac_baselines.ipynb` | RBAC / ABAC (rule-based) | `certificate_valid`; per-authority naive trust | RBAC is deliberately blind to all three axes; ABAC's single- vs multi-authority comparison is the headline finding |
| 3 | `03_ml_baseline.ipynb` | Logistic regression, random forest | `ml_features.ALL_FEATURES` | |
| 4 | `04_dl_baseline.ipynb` | Feedforward MLP (`Linear→ReLU→Dropout`, hidden sizes 64/32) | `ml_features.ALL_FEATURES` | `BCEWithLogitsLoss` with `pos_weight`; checkpointed |
| 5 | `05_rl_baseline.ipynb` | Contextual-bandit DQN | `ml_features.ALL_FEATURES` | Optimises an explicit asymmetric reward, not log-loss — see *RL reward model* below |
| 6 | `06_graph_structural_baseline.ipynb` | 2-hop heterogeneous mean-aggregation GNN | `ml_features.ALL_FEATURES` + entity-level aggregates | See *Graph-baseline caveats* below — a real, documented confound and its fix |
| 7 | `07_transformer_entity_attention.ipynb` | Transformer-over-entity-history (proof of concept) | `ml_features.ALL_FEATURES` + per-entity session history | v2: identity-shortcut fix attempt; ships its own before/after ablation |
| 8 | `08_delegation_check.ipynb` | Naive trust applied to the delegation axis | `ml_features`-adjacent, delegation-specific | Exploratory finding, not a fully evaluated ladder rung (see below) |

Every learned baseline (3–7) deliberately **excludes** raw `cpo_id`/`emsp_id`/`ev_id` from its feature set. `ml_features.py`'s own bug log documents why: one-hot-encoding those IDs produced AUROC/AUPRC = 1.0000 exactly on the real data, because the malicious CPO/eMSP were (in an earlier generator revision) malicious from day 0 with no time variation — a model with the raw ID as input doesn't need any behavioural signal, it just memorises the name. Every feature left in the set is a genuine behavioural/contextual signal, which is the standard the benchmark itself is designed to hold any detector to.

## RL reward model (`05_rl_baseline.ipynb`)

A documented, named design choice — stated plainly so it can be cited precisely. Rows are the ground-truth `any_risk` label, columns are the action taken:

| | DENY | CONDITIONAL-PERMIT | PERMIT |
|---|---|---|---|
| benign (risk=0) | −1.0 | +0.3 | +1.0 |
| risky (risk=1) | +1.0 | +0.2 | −2.0 |

The asymmetry (a false PERMIT on a risky session costs −2.0, twice the magnitude of a false DENY on a benign one) operationalises the standard access-control argument that a false grant is costlier than a false block. Framed as a contextual bandit (not multi-step RL): granting or denying session *N* does not change session *N+1*'s features, so the general Q-learning Bellman update degenerates to `Q(s,a) = r` (γ=0, no next state) — the textbook single-step/contextual-bandit special case, following the standard "supervised dataset → contextual bandit" conversion (Beygelzimer & Langford, 2009). The notebook reports both the actual learned policy's action and what plain `theta_high`/`theta_low` thresholding of an equivalent derived score would have produced, as a direct, quantitative answer to "does optimising for the actual asymmetric cost of a decision change what gets decided."

## Graph-baseline caveats (`06_graph_structural_baseline.ipynb`)

Entity nodes deliberately get **no** learned, per-identity embedding — every entity representation is a deterministic function of neighbouring sessions' behavioural features, for the same reason raw IDs are excluded from every other baseline's feature set. A second, subtler channel to the same effect was found and fixed during development: averaging a feature over an entity's **entire** history (with only 3 CPOs/3 eMSPs and tens of thousands of sessions each) converges to a near-zero-variance estimate of that entity's true per-entity rate parameter — for the eMSP axis specifically, `previous_flag_rate`'s running average recovers `emsp_dispute_flag`'s injected Bernoulli rate almost exactly, re-introducing an identity-shortcut through aggregation rather than raw IDs. Confirmed via an ablation (the notebook's own Step 6: score with each row's own features zeroed out, leaving only entity aggregates — the eMSP-axis AUROC barely moved). Fixed with `MAX_NEIGHBOR_HISTORY=25` (Hamilton et al. 2017's own GraphSAGE neighbour-sampling discipline — an unbounded aggregate trivially overfits for any node type with few, stable neighbours) plus a targeted exclusion of `previous_flag_rate` from the eMSP entity's own aggregate specifically. A tighter bound (`K=5`) was tried and rejected — it destabilised seed-reproducibility (AUROC std jumped from ~0.0001–0.0003 to 0.0064 across seed reseeds) without fully closing the gap. Full dose-response detail is in the module's own docstring.

## Delegation check (`08_delegation_check.ipynb`)

Demonstrates the fourth, parallel identity-delegation axis's blind spot: naive per-authority trust, applied to the *home* identity, is high even when the *acting* (delegate) identity is malicious — the notebook reports both numbers per axis per variant. This is reported in the paper as an exploratory finding, **not** a fully evaluated rung of the main ladder — no baseline here is trained or scored against the delegation ground-truth columns the way baselines 3–7 are scored against `any_risk`.
