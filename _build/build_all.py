"""readme_main_results 전체 재생성 (원본 결과 복사 + CSV/PNG 표). README.md 는 손으로 쓴 것이라 건드리지 않음.

/home/hai_ssh/.conda/envs/lunar/bin/python modify_code/readme_main_results/_build/build_all.py [router router_last para quant relearn cross]
"""
import sys

import build_cross
import build_para
import build_quant
import build_relearn
import build_router
import build_router_last_png

STEPS = {"router": build_router, "router_last": build_router_last_png, "para": build_para, "quant": build_quant, "relearn": build_relearn,
         "cross": build_cross}

if __name__ == "__main__":
    for k in sys.argv[1:] or STEPS:
        STEPS[k].main()
