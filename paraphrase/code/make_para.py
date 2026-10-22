"""평가용 SD/RD 문항을 GPT-4o로 1:1 변형(paraphrase 유형별)해서 step별 평가 파일을 만든다.

SD/RD는 같은 문항 풀에서 t에 따라 나뉘므로, 풀 전체를 유형마다 한 번만 변형(mapping)하고
원본 step 파일 구조 그대로 *_{suffix} 폴더에 질문만 바꿔 쓴다. 정답/intent/perturbed 등은 그대로.

유형(--type): para(의미 보존) | ctx(문맥 추가) | struct(문장 구조 변경) | bt(역번역, 영→독→영) | noise(표면 잡음)

  python3 make_para.py gen   --dataset clinc --type ctx --limit 10   # 시험 생성 (앞 N개)
  python3 make_para.py gen   --dataset clinc --type ctx              # 전체 생성 (만든 것은 건너뜀, 탈락분 재요청)
  python3 make_para.py build --dataset clinc --type ctx              # step별 평가 파일 생성

키: modify_code/.env 첫 줄 (값만). 출력: paraphrase/maps/{dataset}_{type}[_{stage}].json, paraphrase/logs/
표준 라이브러리만 사용 (실험 conda 환경에 설치 없음).
para 유형은 logs/make_para_v1_para_only.py.bak 버전으로 생성됨 (지시문 동일하게 유지).
"""
import argparse, concurrent.futures as cf, difflib, glob, json, os, random, re, sys, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)  # data_new
KEY_FILE = os.path.join(os.path.dirname(DATA), "modify_code", ".env")
MODEL = "gpt-4o-2024-11-20"
PRICE_IN, PRICE_OUT = 2.50 / 1e6, 10.00 / 1e6  # USD per token (GPT-4o 공식 가격 기준, 확인 필요)
BATCH, WORKERS, SEED = 20, 8, 1234

DATASETS = {
    "clinc": dict(
        src={"SD": "clincqa_SD_10/*_test_SD.json", "RD": "clincqa_RD_10/*_test_RD.json"},
        out={"SD": "clincqa_SD_10_{suffix}", "RD": "clincqa_RD_10_{suffix}"},
        context="Inputs are short utterances a user might say to a virtual assistant "
                "(often casual and lowercase). Keep them as natural user utterances.",
    ),
    "tofu": dict(
        src={"SD": "ptofu_SD/*.json", "RD": "ptofu_RD/*.json"},
        out={"SD": "ptofu_SD_{suffix}", "RD": "ptofu_RD_{suffix}"},
        context="Inputs are questions about fictitious authors and their books. "
                "Every person name, book title, place and date must stay exactly as written.",
    ),
}

# para 유형 원래 지시문 (v1, 변경 금지)
SYSTEM_V1 = """You rewrite queries for a paraphrase-robustness evaluation of a language model.
{context}
For each input: {rule}
Rules:
- Keep every named entity (person names, places, titles of works, organizations), number and date exactly as written.
- Do not add, remove or hint at any information, and never answer the question.
- Keep the same register and a similar length.
Return JSON: {{"items": [{{"id": <id>, "paraphrase": "<text>"}}]}} with exactly one entry per input id."""

SYSTEM_V2 = """You transform queries for a robustness evaluation of a language model.
{context}
Task for each input: {rule}
Rules:
- Keep every named entity (person names, places, titles of works, organizations), number and date exactly as written.
- Do not add, remove or hint at any information relevant to the answer, and never answer the question.
Return JSON: {{"items": [{{"id": <id>, "output": "<text>"}}]}} with exactly one entry per input id."""

# stage: rule(지시), len(원문 대비 길이 비율 허용 범위), ident(원문과 같아도 허용), overlap(원문 단어 유지 비율 하한), sim(글자 유사도 하한)
TYPES = {
    "para": dict(suffix="para", system=SYSTEM_V1, stages=[dict(
        name="", len=(0.5, 2.0),
        rule="Write ONE paraphrase that keeps exactly the same meaning, intent and requested information, "
             "but noticeably changes the wording and/or sentence structure (not just one word).")]),
    "ctx": dict(suffix="para_ctx", system=SYSTEM_V2, stages=[dict(
        name="", len=(1.2, 6.0), overlap=0.8,
        rule="Embed the query in a short natural conversational context, as a real user might write it: add a brief "
             "neutral lead-in and/or closing (e.g., a greeting, a short remark about why they are asking, or a thank-you). "
             "Keep the original query itself essentially unchanged (same words). The added context must not contain any "
             "information related to the answer and must not mention ANY named entity (not even the one in the query).")]),
    "struct": dict(suffix="para_struct", system=SYSTEM_V2, stages=[dict(
        name="", len=(0.5, 2.5),
        rule="Rewrite the query with a different sentence type while keeping exactly the same meaning: turn a question into "
             "an imperative or declarative request (e.g., 'Tell me ...', 'I'd like to know ...'), and turn a command or "
             "statement into a question (e.g., 'Could you ...?', 'How do I ...?'). Keep the speaker's perspective: the user "
             "is still talking about themselves (I/my/me stay first person; e.g., 'i need to book a flight' -> 'can you book "
             "a flight for me?', never 'are you booking a flight?'). When a statement implies a request, turn it into a "
             "question asking for that same thing (e.g., 'i'll fly from lax to phl for $120 on nov 5th' -> 'can i fly from "
             "lax to phl for $120 on nov 5th?'); do not add new requests such as confirming, checking or verifying. "
             "Keep the same register (casual lowercase stays casual) and a similar length.")]),
    "bt": dict(suffix="para_bt", system=SYSTEM_V2, stages=[
        dict(name="de", len=(0.5, 3.0), ident=True,
             rule="Translate the input into natural German. Keep person names, titles of works, places, numbers and dates "
                  "exactly as written (do not translate names or titles)."),
        dict(name="en", len=(0.5, 2.5), ident=True,
             rule="The input is in German. Translate it into natural English. Keep person names, titles of works, places, "
                  "numbers and dates exactly as written."),
    ]),
    "noise": dict(suffix="para_noise", system=SYSTEM_V2, stages=[dict(
        name="", len=(0.7, 1.3), sim=0.75,
        rule="Rewrite the query as if typed quickly by a careless user: introduce realistic typos in ordinary words "
             "(1 typo if the input has fewer than 6 words, otherwise 2-3 typos; e.g., adjacent-key substitutions, dropped, "
             "doubled or swapped letters); you may also drop some punctuation or capitalization. Do NOT change any other "
             "words, and do NOT put typos into person names, titles of works, places or numbers.")]),
}

lock = threading.Lock()
usage = {"in": 0, "out": 0, "requests": 0}


def api_key():
    return open(KEY_FILE).read().strip().splitlines()[0].strip().strip('"').strip("'")


def load_pool(ds):
    pool = {}
    for split, pat in DATASETS[ds]["src"].items():
        for f in sorted(glob.glob(os.path.join(DATA, pat))):
            for x in json.load(open(f)):
                pool.setdefault(x["question"], x)
    return pool  # question -> 대표 item


def call(messages, key, retries=5):
    body = json.dumps(dict(model=MODEL, messages=messages, temperature=0, seed=SEED,
                           response_format={"type": "json_object"})).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=120))
            with lock:
                usage["in"] += d["usage"]["prompt_tokens"]; usage["out"] += d["usage"]["completion_tokens"]; usage["requests"] += 1
            return json.loads(d["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            msg = e.read()[:200].decode(errors="ignore")
            if e.code in (429, 500, 502, 503) and a < retries - 1:
                time.sleep(2 ** a + random.random()); continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if a < retries - 1:
                time.sleep(2 ** a); continue
            raise


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


SOFT = ("length_ratio", "number", "overlap", "char_sim")  # 3번 실패 시 마지막 결과 채택해도 되는 사유


def check(ds, orig, item, out, st):
    """문제 있으면 사유 문자열, 통과면 None. orig = 원래 영어 질문 (이름/숫자 기준)"""
    if not isinstance(out, str) or not out.strip():
        return "empty"
    if not st.get("ident") and norm(out) == norm(orig):
        return "identical"
    r = len(out) / max(len(orig), 1)
    lo, hi = st["len"]
    if len(orig) >= 20 and not lo <= r <= hi:  # 아주 짧은 문장("10-4" 등)은 길이 비율 검사 제외
        return f"length_ratio {r:.2f}"
    for num in re.findall(r"\d+", orig):
        if num not in out:
            return f"number {num} missing"
    if ds == "tofu":
        name = (item.get("forget_author") or "").strip()
        for part in ([name] + name.split()) if name else []:
            if len(part) > 2 and part.lower() in orig.lower() and part.lower() not in out.lower():
                return f"name '{part}' missing"
    if st.get("overlap"):
        ow, nw = words(orig), set(words(out))
        k = sum(w in nw for w in ow) / max(len(ow), 1)
        if k < st["overlap"]:
            return f"overlap {k:.2f}"
    if st.get("sim"):
        s = difflib.SequenceMatcher(None, orig.lower(), out.lower()).ratio()
        if s < st["sim"]:
            return f"char_sim {s:.2f}"
    return None


def run_stage(ds, ty, st, pool, qs, inputs, key, map_path):
    """inputs: 원문 q -> 이번 stage에 넣을 텍스트. 반환: q -> 출력, forced, unchanged"""
    mp = json.load(open(map_path)) if os.path.exists(map_path) else {}
    system = TYPES[ty]["system"].format(context=DATASETS[ds]["context"], rule=st["rule"])
    rejects = {}
    label = f"{ds}/{ty}" + (f"/{st['name']}" if st["name"] else "")

    def run_batch(batch, extra=""):
        payload = [{"id": i, "text": inputs[q]} for i, q in enumerate(batch)]
        msgs = [{"role": "system", "content": system + extra},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        out = call(msgs, key)
        got = {int(x["id"]): (x.get("output") if "output" in x else x.get("paraphrase"))
               for x in out.get("items", []) if str(x.get("id", "")).isdigit()}
        res = {}
        for i, q in enumerate(batch):
            p = got.get(i)
            why = "missing id" if p is None else check(ds, q, pool[q], p, st)
            res[q] = (p, why)
        return res

    for attempt in range(3):
        todo = [q for q in qs if q not in mp]
        if not todo:
            break
        extra = "" if attempt == 0 else "\nA previous attempt failed automatic checks: follow every rule exactly (names, numbers, length)."
        size = BATCH if attempt == 0 else 5
        batches = [todo[i:i + size] for i in range(0, len(todo), size)]
        print(f"[{label}] attempt {attempt}: {len(todo)} items in {len(batches)} requests", flush=True)
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            for n, res in enumerate(ex.map(lambda b: run_batch(b, extra), batches), 1):
                with lock:
                    for q, (p, why) in res.items():
                        if why is None:
                            mp[q] = p; rejects.pop(q, None)
                        else:
                            rejects[q] = {"out": p, "why": why}
                if n % 20 == 0 or n == len(batches):
                    json.dump(mp, open(map_path, "w"), ensure_ascii=False, indent=1)
                    print(f"  {n}/{len(batches)} done, mapped {len(mp)}, cost ${usage['in']*PRICE_IN + usage['out']*PRICE_OUT:.2f}", flush=True)

    # 3번 시도 후에도 탈락: 가벼운 사유면 마지막 결과 채택(forced), 그 외는 입력 그대로(unchanged)
    forced, unchanged = {}, {}
    for q in [q for q in qs if q not in mp]:
        r = rejects.get(q, {"out": None, "why": "no candidate"})
        if r["out"] and str(r["why"]).startswith(SOFT):
            mp[q] = r["out"]; forced[q] = r
        else:
            mp[q] = inputs[q]; unchanged[q] = r
    json.dump(mp, open(map_path, "w"), ensure_ascii=False, indent=1)
    return mp, forced, unchanged


def gen(args):
    ds, ty = args.dataset, args.type
    key = api_key()
    pool = load_pool(ds)
    qs = sorted(pool)
    if args.limit:
        qs = qs[: args.limit]
    os.makedirs(os.path.join(HERE, "maps"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    stages = TYPES[ty]["stages"]
    inputs = {q: q for q in qs}
    stage_logs = []
    for i, st in enumerate(stages):
        last = i == len(stages) - 1
        map_path = os.path.join(HERE, "maps", f"{ds}_{ty}.json" if last else f"{ds}_{ty}_{st['name']}.json")
        mp, forced, unchanged = run_stage(ds, ty, st, pool, qs, inputs, key, map_path)
        stage_logs.append(dict(stage=st["name"], map=os.path.basename(map_path), forced=forced, unchanged=unchanged))
        inputs = {q: mp[q] for q in qs}
    cost = usage["in"] * PRICE_IN + usage["out"] * PRICE_OUT
    final = json.load(open(os.path.join(HERE, "maps", f"{ds}_{ty}.json")))
    log = dict(dataset=ds, type=ty, model=MODEL, seed=SEED, requested=len(qs), mapped=sum(q in final for q in qs),
               stages=stage_logs, usage=usage, cost_usd=round(cost, 4), time=time.strftime("%F %T"))
    tag = f"{ds}_{ty}" + (f"_pilot{args.limit}" if args.limit else "")
    json.dump(log, open(os.path.join(HERE, "logs", f"{tag}_{time.strftime('%m%d_%H%M%S')}.json"), "w"), ensure_ascii=False, indent=1)
    nf = sum(len(s["forced"]) for s in stage_logs); nu = sum(len(s["unchanged"]) for s in stage_logs)
    print(f"[{ds}/{ty}] mapped {log['mapped']}/{len(qs)}, forced {nf}, unchanged {nu}, "
          f"tokens in {usage['in']} out {usage['out']}, requests {usage['requests']}, cost ${cost:.3f}")
    for s in stage_logs:
        for tag_, d in (("FORCED", s["forced"]), ("UNCHANGED", s["unchanged"])):
            for q, r in list(d.items())[:5]:
                print(f"  {tag_}[{s['stage']}]", r["why"], "|", q, "->", r["out"])


def build(args):
    ds, ty = args.dataset, args.type
    mp = json.load(open(os.path.join(HERE, "maps", f"{ds}_{ty}.json")))
    suffix = TYPES[ty]["suffix"]
    meta = dict(para_type=ty, para_model=MODEL)
    for split, pat in DATASETS[ds]["src"].items():
        out_dir = os.path.join(DATA, DATASETS[ds]["out"][split].format(suffix=suffix))
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(DATA, pat)))
        missing = set()
        for f in files:
            new = []
            for x in json.load(open(f)):
                q = x["question"]
                if q not in mp:
                    missing.add(q); continue
                p, y = mp[q], dict(x)
                y["orig_question"] = q
                y["question"] = p
                for k in ("instruction", "text_input"):
                    if k in y and isinstance(y[k], str):
                        if ds == "clinc":
                            assert y[k].endswith("Utterance: " + q), (f, k)
                            y[k] = y[k][: -len(q)] + p
                        else:
                            assert y[k] == q, (f, k)
                            y[k] = p
                if ds == "clinc" and "text_full" in y:
                    assert x["text_full"] == x["instruction"] + x["output"], (f, "text_full")
                    y["text_full"] = y["instruction"] + y["output"]
                y.update(meta)
                new.append(y)
            if missing:
                continue
            json.dump(new, open(os.path.join(out_dir, os.path.basename(f)), "w"), ensure_ascii=False, indent=1)
        if missing:
            sys.exit(f"[{ds}/{ty}/{split}] mapping 없는 문항 {len(missing)}개 → gen 먼저 완료 필요")
        print(f"[{ds}/{ty}/{split}] {len(files)} files -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "build"])
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--type", choices=list(TYPES), default="para")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else build(a)
