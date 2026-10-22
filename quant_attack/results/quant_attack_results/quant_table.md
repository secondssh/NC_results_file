# Quantization attack (NF4 4bit, seed 0) — CLINC t8 / TOFU t19

열: bf16 = 원래 성능, backbone = 백본만 4bit, all = 백본 + (RULE: MoE lora/gate, O3: LoRA) 4bit. 괄호 = bf16 대비 Δ. forget 은 높아질수록 지식이 되살아난 것.
O3 CLINC 는 exact_acc (OOD detector: ood_checkpoints_clincqa_recent), 나머지는 choice_accuracy. O3 TOFU 평가는 forget 40 / retained 200 문항 (RULE 은 800 / 3200).


## CLINC t8

| method | split | bf16 | backbone | all |
|---|---|---|---|---|
| RULE | forget | 0.013 | - | - |
| RULE | retained | 0.969 | - | - |
| RULE | mrpc | 0.701 | - | - |
| RULE | rte | 0.523 | - | - |
| O3 | forget | 0.841 | - | - |
| O3 | retained | 0.693 | - | - |
| O3 | mrpc | 0.808 | - | - |
| O3 | rte | 0.892 | - | - |
| base | forget | 0.972 | - | - |
| base | retained | 0.971 | - | - |
| base | mrpc | - | - | - |
| base | rte | - | - | - |

## TOFU t19

| method | split | bf16 | backbone | all |
|---|---|---|---|---|
| RULE | forget | 0.419 | - | - |
| RULE | retained | 0.844 | - | - |
| RULE | real_authors | 0.840 | - | - |
| RULE | world_facts | 0.838 | - | - |
| O3 | forget | 0.400 | - | - |
| O3 | retained | 0.290 | - | - |
| O3 | real_authors | 0.300 | - | - |
| O3 | world_facts | 0.496 | - | - |
| base | forget | - | - | - |
| base | retained | - | - | - |
| base | real_authors | - | - | - |
| base | world_facts | - | - | - |
