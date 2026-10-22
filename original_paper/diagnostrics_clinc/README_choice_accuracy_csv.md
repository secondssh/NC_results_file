# CLINC150 — choice_accuracy by stage

Generated from the result JSONs listed under "Sources". Two files:

- `choice_accuracy_per_seed.csv` — one row per (method, seed, stage, split)
- `choice_accuracy_seedavg.csv` — one row per (method, stage, split), with
  `n_seeds`, `choice_accuracy_mean`, `choice_accuracy_std`
  (sample std, ddof=1; **empty whenever `n_seeds` = 1**)

`stage` is 0-based; `domain` is the CLINC intent domain (`work` = stage 0).

## Seed averaging: what can and cannot be averaged

| method | seeds | stages | seed-average |
|---|---|---|---|
| GA | **1** (seed 0) | 9 (t0–t8, all) | **no** |
| GA_GD | **1** (seed 0) | 9 (t0–t8, all) | **no** |
| NPO_KL | **1** (seed 0) | 9 (t0–t8, all) | **no** |
| OURS | 3 (layer16_0/1/2) | 9 (t0–t8, all) | yes |
| O3 | 3 (seed_0/1/2) | 9 (t0–t8, all) | yes |

The three reproduced baselines were each run for **seed 0 only**, so they carry no
mean/std. Reporting std for them requires re-running seeds 1 and 2 (~1.5 days per
job at the current settings). All five methods cover the full **t0–t8** range.

## Caveats that affect any cross-method comparison

1. **O3's metric was rescaled.** Its JSONs store `exact_acc` on a 0–100 scale, not
   `choice_accuracy` on 0–1, so it is divided by 100 here. It is the same quantity
   (exact match of the generated `"Intent: X"` against the gold label), but it is
   computed by a different codebase. Note the older 3-domain run keyed this `acc`;
   the 9-domain run used here keys it `exact_acc`.
2. **O3 post-unlearning results are the `test_`-prefixed files.** Each domain has
   both `seed{S}_<domain>_<split>.json` and `test_seed{S}_<domain>_<split>.json`;
   only the latter is used. Its `mrpc`/`rte` are constant across all 9 domains
   (0.8081 / 0.8917) because O3 routes out-of-domain queries to the untouched
   original model, so utility is invariant to how many domains have been unlearned.
   The unprefixed files are the unlearning branch without that routing (forget
   0.007–0.050, retain 0.004–0.118) and are not used.
3. **O3's `forget` rises over stages, and the reason matters.** It goes 0.153 (t0)
   → 0.842 (t8) on the cumulative forget set. The per-domain breakdown stored in
   the same file at t8 shows only the just-unlearned domain is actually forgotten,
   while every earlier one has recovered:

   | domain unlearned at | exact_acc at t8 |
   |---|---|
   | small_talk (t8, most recent) | **0.138** |
   | travel (t1) | 0.987 |
   | credit_cards (t4) | 0.958 |
   | banking (t3) | 0.956 |
   | work (t0) | 0.940 |
   | kitchen_and_dining (t5) | 0.938 |
   | auto_and_commute (t6) | 0.936 |
   | home (t2) | 0.884 |
   | utility (t7) | 0.836 |

   So in this run O3 forgets only the current domain and does not retain earlier
   forgetting. The `forget` column is therefore **not** comparable to the
   baselines' as "how much has been unlearned so far" even though both are
   computed on the same cumulative forget set — for the baselines a low value
   means genuinely forgotten, whereas O3's high value means earlier forgetting
   was undone. Worth confirming against the intended O3 protocol before using
   this column in a table.
4. **Split names are normalized.** The baselines' eval writes the two utility sets
   under the generic keys `commonqa` / `openbookqa`, but the underlying files are
   `mrpc_test.json` / `rte_validation.json`. They appear here as `mrpc` / `rte`,
   matching what OURS already uses.
5. **MRPC and RTE are binary tasks — chance is 0.50.** Values well below that
   (e.g. GA_GD `rte` = 0.011 at t3) mean the output format itself has broken, not
   merely that the answers are wrong.
6. **CLINC `choice_accuracy` is generation-based**, so it is sensitive to eval
   settings in a way TOFU's likelihood-ranked score is not. Measured on RTE at t0,
   `eval_batch_size` 8 vs 16 moves it 0.7329 → 0.7401 (2/277 samples flip). The
   baselines and OURS are all at the config default of 8; O3 comes from a separate
   codebase, so that is not guaranteed to match.
7. **The baselines here are the 10-epoch runs.** An earlier 20-epoch attempt is
   archived under `unlearn_baseline_fullft_clinc_ep20_*` (2–3 stages only, GA fully
   collapsed at t0) and is excluded.

## Sources

| method | path |
|---|---|
| GA / GA_GD / NPO_KL | `unlearning_baseline_new/run_results/eval_basemoe_error/llama3.1-8b-instruct/unlearn_baseline_fullft_clinc_<method>_0/*/forget_*.json` |
| OURS | `modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct/clinc_moe_fixed/clincqa_moe_fixed_layer16_{0,1,2}/*/forget_*.json` |
| O3 | `ScienceQA_expriment/unlearning_o3/unlearn_recent_o3/SCALE_0.1_seed_{0,1,2}_o_unlearn_lora_force_checkpoints_clinc9_origin/test_seed{S}_<domain>_{testSD,testRD,mrpc,rte}.json` — the 9-domain run; the earlier 3-domain `unlearn_lora_clinc` run is superseded |
