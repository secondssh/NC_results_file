#!/usr/bin/env python3
"""train_unlearn_lora_o.py (O3 LLM LoRA 학습) 를 원래 코드·인자 그대로 실행하되, tokenizer chat template 만 보충.

O3_LLAMA31_ClincQA 는 transformers 5.x 로 저장돼 chat template 이 chat_template.jinja 에만 있다. transformers 4.46 (lunar) 은
이 파일을 안 읽어 build_chat_prompt → apply_chat_template 가 실패한다. eval_quant_o3_clinc.py (O3 추론 측정) 와 같은 방식으로
template 이 비어 있으면 모델 폴더의 chat_template.jinja 를 넣는다 (최신 transformers 가 읽는 것과 같은 template). 그 외는 손대지 않음.
사용: python o3_lora_train_wrapper.py <train_unlearn_lora_o.py 경로> <원래 인자...>
"""
import os
import runpy
import sys

import transformers

_orig = transformers.AutoTokenizer.from_pretrained.__func__


def _from_pretrained(cls, path, *args, **kwargs):
    tok = _orig(cls, path, *args, **kwargs)
    jinja = os.path.join(path, "chat_template.jinja") if isinstance(path, str) else None
    if getattr(tok, "chat_template", None) is None and jinja and os.path.isfile(jinja):
        tok.chat_template = open(jinja).read()
        print(f"[wrapper] chat_template ← {jinja}", flush=True)
    return tok


transformers.AutoTokenizer.from_pretrained = classmethod(_from_pretrained)

script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
