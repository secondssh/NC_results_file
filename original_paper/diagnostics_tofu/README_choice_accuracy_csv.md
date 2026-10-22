# TOFU — choice_accuracy by stage

Generated from the result JSONs listed under "Sources". Two files:

- `choice_accuracy_per_seed.csv` — one row per (method, seed, stage, split)
- `choice_accuracy_seedavg.csv` — one row per (method, stage, split), with
  `n_seeds`, `choice_accuracy_mean`, `choice_accuracy_std`
  (sample std, ddof=1; **empty whenever `n_seeds` = 1**)

`stage` is 0-based; `domain` is the TOFU author group (`AG01` = stage 0).

## Seed averaging: what can and cannot be averaged

| method | seeds | stages | seed-average |
|---|---|---|---|
| GA | **1** (seed 0) | 22 (t0–t21) | **no** |
| GA_GD | **1** (seed 0) | 19 (t0–t18) | **no** |
| NPO_KL | **1** (seed 0) | 20 (t0–t19) | **no** |
| OURS | 3 (layer16_0/1/2) | 30 (t0–t29) | yes |
| O3 | 3 (seed_0/1/2) | 26 (t0–t25) | yes |

The three reproduced baselines were each run for **seed 0 only**, so they carry no
mean/std. Reporting std for them requires re-running seeds 1 and 2.

## Caveats that affect any cross-method comparison

1. **Ragged stage coverage.** The methods stop at different stages (19–30 above).
   The CSV does not pad or truncate. The range where all five methods exist is
   **t0–t18**.
2. **O3's utility is extremely unstable across stages**, driven by its refusal
   behaviour — `refusal_rate` on `world_facts` for seed 0 goes 0.966 (t0) → 0.094
   (t10) → 0.726 (t15) → 0.872 (t25), and `choice_accuracy` moves inversely
   (0.034 → 0.855 → 0.248 → 0.103). At t0 the model refuses even general world
   knowledge ("Where would you find the Eiffel Tower?" → "I'm clueless about that
   topic."). This is what the stored result files contain — it is not a scoring
   artifact, the generations really are refusals. **Reading any single stage as
   "O3's performance" is therefore misleading**; `PTOFU_experiment/.../llama31_8b_O3`
   is the only O3 TOFU result directory available, so there is no alternative run
   to cross-check against.
3. **Split names are normalized.** The baselines' eval writes the two utility
   sets under the generic keys `commonqa` / `openbookqa`, but the underlying files
   are `real_authors_test.json` / `world_facts_test.json`. They appear here as
   `real_authors` / `world_facts`, matching what OURS and O3 already use.
4. **Baselines vs OURS/O3 come from different eval entrypoints.** Baselines use
   `unlearning_baseline_new/eval_tofu.py` (+ `tofu_eval_util_fullft.py`); OURS and
   O3 were produced by their own pipelines. The choice-hit scorer is the same
   TOFU one, but no other eval setting is guaranteed identical.

## Sources

| method | path |
|---|---|
| GA / GA_GD / NPO_KL | `unlearning_baseline_new/run_results/eval_basemoe_error/llama3.1-8b-instruct/unlearn_baseline_fullft_tofu_<method>_0/*/forget_*.json` |
| OURS | `modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct/tofu_moe_fixed/tofu_moe_fixed_layer16_{0,1,2}/*/forget_*.json` |
| O3 | `PTOFU_experiment/files/results/unlearn_ptofu/llama31_8b_O3/seed_{0,1,2}/stage_*/ptofu.json` |

`NPO` (no KL) also exists at seed 0 but only 8 stages, and is excluded.
