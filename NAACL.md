# NAACL 넣을 표/그림 체크리스트 (2026-10-23 기준)

비교군: O3, GA (GD, NPO+KL), Ours(RULE). ✅ = 결과 + 표/그림 있음, ⚠️ = 있지만 비교군이나 조건 일부 빠짐, ❌ = 없음

| # | 항목 | 위치 | 상태 | 파일 | 비교군 / 빠진 것 |
|---|---|---|---|---|---|
| 1 | # of params | Main | ✅ | [latency/tables/params.png](latency/tables/params.png) (`.csv`) | RULE CLINC t0·t4·t8 / TOFU t0·t10·t19, O3, GA·GD·NPO+KL 전부. E 확장(요청 80건 이상): [scale_micro](latency/tables/scale_micro.png) |
| 2 | Training time | Append | ⚠️ | [train_rule](latency/tables/train_rule.png), [train_baseline](latency/tables/train_baseline.png), [train_o3](latency/tables/train_o3.png) | RULE·GA·GD·NPO+KL은 CLINC/TOFU 둘 다. **O3는 CLINC t0만** (TOFU O3 학습 시간 없음) |
| 3 | Latency | Append | ✅ | [infer_clinc_bs1_fixed64](latency/tables/infer_clinc_bs1_fixed64.png), [infer_tofu_bs1_fixed64](latency/tables/infer_tofu_bs1_fixed64.png) (주 표), eval batch / 원래 설정 표, [o3_clinc_breakdown](latency/tables/o3_clinc_breakdown.png), [microbench](latency/tables/microbench_router_expert.png), [scale_e2e](latency/tables/scale_e2e.png) | base, RULE, GD, O3 (CLINC·TOFU). GA·NPO+KL은 GD와 구조가 같아(full-FT, 추가 모듈 0) 따로 안 잼 → 각주로 처리. bs1 원래 생성 설정은 24문항 1회뿐 |
| 4 | Routing performance (CLINC, TOFU) | Append | ✅ | [clinc_router_acc_seed_mean_std](router_acc/last_png/clinc_router_acc_seed_mean_std.png), [tofu_router_acc_seed_mean_std](router_acc/last_png/tofu_router_acc_seed_mean_std.png) (seed 0/1/2 mean ± std), 요약 [tofu_summary](router_acc/tables/tofu_summary.png) / [clinc_summary](router_acc/tables/clinc_summary.png) | RULE만 해당 (O3 OOD 탐지기 정확도는 안 뽑음) |
| 5 | CLINC routing confusion matrix | Append | ✅ | [clinc_full_cm_last](router_acc/tables/clinc_full_cm_last.png) (t8, 10 expert), step별 [clinc_cm_by_step](router_acc/last_png/clinc_cm_by_step.png), 이진 [clinc_binary_cm_last](router_acc/tables/clinc_binary_cm_last.png) | router가 domain 단위(e0 + 9 domain)라 150 intent CM은 아님 |
| 6 | Paraphrase 5유형 figure (Forget / Retain × CLINC / TOFU) | Main | ✅ (새로 만듦) | [para_main_2x2](paraphrase/tables/para_main_2x2.png) (y축 확대), gap만 [para_main_2x2_delta](paraphrase/tables/para_main_2x2_delta.png), `.pdf` 같이 있음 | RULE seed0만. base는 [base_vs_rule 표](paraphrase/tables/). **O3·GA·GD·NPO+KL paraphrase 평가는 없음** |
| 7 | Quantization 4bit table | – | ⚠️ | [quant_clinc](quant_attack/tables/quant_clinc.png), [quant_tofu](quant_attack/tables/quant_tofu.png) | CLINC: RULE t0·t4·t8, O3 t0, GD t0, base. TOFU: RULE t0·t10만. **없음: GA·NPO+KL (bs64 설정), TOFU t19, O3 TOFU, GD TOFU** (GA·NPO+KL은 예전 bs1024 legacy에서 4bit 붕괴 기록만) |
| 8 | Relearning figure | – | ✅ (새로 만듦) / ⚠️ | [relearn_main](relearning_attack/tables/relearn_main.png) (`.pdf`) | t0 work in: RULE·O3·GD / t3 banking in: RULE·O3·base / t3 banking out(BANKING77): 6개 전부. **GA·NPO+KL in-domain 공격 없음, TOFU relearning 없음.** GA t3는 공격 전부터 모든 split 0 (붕괴) |

## 본문에 쓸 때 주의

- 6: 원문 대비 forget 차이는 step 평균으로 CLINC −0.003 ~ +0.014, TOFU −0.035 ~ +0.029 (step별 최대 CLINC +0.036, TOFU ±0.077). 오타가 가장 많이 샘. TOFU retain의 Semantic 하락(−0.07)은 base에서도 똑같이 떨어지는 질문 난이도 효과 ([paraphrase/README.md](paraphrase/README.md) 2절)
- 8: t0 in-domain 1 epoch에서는 RULE이 가장 많이 돌아옴 (0.79 vs O3 0.30, GD 0.10). 5 epoch에서는 셋 다 0.90 이상. out-domain에서는 RULE 0.44로 O3(0.62)보다 낮음. O3는 t3 공격 전부터 forget 0.71 (원래 덜 잊음)
- 3: O3 수치는 공개 코드 그대로 (eager attention, LoRA 224곳 미병합). 구조적 비용(OOD 탐지기 44.6ms/요청, 355M)과 구현 비용을 나눠 쓰기

## 그림 다시 만들기

```bash
/home/hai_ssh/.conda/envs/lunar/bin/python modify_code/readme_main_results/_build/build_naacl_figs.py   # 6, 8번 그림 (결과 파일만 읽음, GPU 안 씀)
```
