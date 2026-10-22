# 확장실험 결과 모음 (RULE, seed 0 중심)

`modify_code/README.md`의 확장실험 절마다 결과를 한 폴더에 모았습니다. 각 폴더 구성은 같습니다.
- `README.md`: 설정, 결과 표, 분석, 미완료 항목, 원래 위치
- `tables/`: CSV + PNG
- `results/`: 원본 json/csv 복사본
- `code/`: 실험 코드 사본

| 폴더 | 실험 | 상태 | 한 줄 결론 |
|---|---|---|---|
| [router_acc/](router_acc/README.md) | Router acc (마지막 토큰 / prompt 토큰 / 생성 토큰, forget+retain CM) | 완료 (TOFU t0–19, CLINC t0–8, seed 0/1/2) | 실제 routing은 TOFU F 0.59 / R 0.95, CLINC F 0.80 / R 0.99. 결과 json router_accuracy는 TOFU에서 padding 때문에 낮게 나옴. 생성 토큰은 forget이어도 대부분 e0로 감 |
| [paraphrase/](paraphrase/README.md) | GPT-4o 5유형 paraphrase로 SD/RD 재평가 | 완료 145/145 + base 12 | forget 변화 TOFU −0.035 ~ +0.029, CLINC −0.003 ~ +0.014 (가장 큰 것은 오타). TOFU para의 retained 하락은 base도 똑같이 떨어지는 질문 난이도 효과 |
| [quant_attack/](quant_attack/README.md) | NF4 4bit 양자화 공격 (백본 / 백본+MoE) | CLINC RULE t0·t8, O3 t0, GD t0, TOFU RULE t0·t10 완료 | 어느 조건에서도 forget이 되살아나지 않음 (±0.03). 떨어지는 것은 utility |
| [relearning_attack/](relearning_attack/README.md) | Targeted relearning (LoRA 재학습) | 완료 (CLINC t8 banking 12 run, t0 RULE/O3/GD) | in-domain 100문장 5 epoch이면 RULE forget이 base 수준까지 돌아옴 (백본만 학습해도 동일). t0 1 epoch에서는 RULE 0.79 / O3 0.30 / GD 0.10으로 RULE이 가장 약하고, 5 epoch에서는 셋 다 0.90 이상 |
| [cross_domain/](cross_domain/README.md) | F+R / R+F 섞인 입력 | 완료 (CLINC t4·t8, TOFU t10·t19, RULE + base) | CLINC는 섞어도 안 샘. TOFU는 router 마지막 토큰이 e0로 가서 +0.06 ~ +0.13 더 샘 |
| `latency/` | (이번 정리 범위 밖, 비어 있음) | 측정 스크립트 작성 전 | 기존 값은 `modify_code/latency/existing_metrics.md` |

## 다시 만들기

```bash
/home/hai_ssh/.conda/envs/lunar/bin/python modify_code/readme_main_results/_build/build_all.py            # 전부
/home/hai_ssh/.conda/envs/lunar/bin/python modify_code/readme_main_results/_build/build_all.py relearn    # 하나만 (router para quant relearn cross)
```

- 원래 위치에서 결과를 다시 복사하고 CSV/PNG를 새로 만듭니다.
- README는 손으로 쓴 것이라 덮어쓰지 않습니다. 예외는 relearning_attack README의 실험 B 표로, csv와 같은 값으로 자동 갱신됩니다.
- 2026-10-22 01:2x 기준으로 만들었습니다. paraphrase 요약표(`repro_check/compare_para_eval.*`, `base_para_steps.*`, `para_eval_steps.png`)는 145개가 모두 끝난 데이터로 이번에 다시 생성했습니다. 예전 파일은 TOFU 일부 step만 반영돼 있었습니다.
