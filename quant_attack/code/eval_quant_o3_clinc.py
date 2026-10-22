"""eval_o3_clinc.py 복사본 + quantization attack 옵션 (원본은 수정하지 않음).

원본과 다른 점:
  --quant_mode none|backbone|all : PeftModel 로드 직후 fake NF4 적용 (quant_attack/fake_quant.py)
                                   backbone = 백본 Linear 224개(base_layer), all = + LoRA lora_A/lora_B
  --result_file                  : 결과 경로 (원본은 LoRA 폴더 옆 test_seed*_*.json 에 써서 O3 원래 결과를 덮어씀)
  --max_samples N                : 앞 N 개만 평가 (재현/ood detector 확인용)
OOD detector(RoBERTa) 는 양자화하지 않는다. batch size 1 도 원본 그대로 (OOD weight 가 배치 단위).
실행은 ScienceQA_expriment/ 에서 (src/, utils/ 상대 import).
"""
import os, re, json, argparse, torch, random, pickle, math, sys 
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # ScienceQA_expriment (src/, utils/)
sys.path.insert(1, _HERE)                    # quant_attack (fake_quant)
from fake_quant import apply_fake_quant
from tqdm import tqdm
from collections import defaultdict
from src.ood_model_selector import RobertaForSelector, RobertaForSelector_inference
from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers import GenerationConfig, AutoTokenizer, AutoConfig
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
from typing import Union
from sklearn.mixture import GaussianMixture as GMM
from src.peft_model_hacked_o import PeftModel
from src.modeling_llama_hacked_31 import LlamaForCausalLM_ood
from utils.prompter import build_chat_prompt

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  
    pass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def weighting_func_gmm(train_in_score, test_in_score):
    # 1. fit two gaussians
    mean1, std1 = norm.fit(train_in_score)
    mean2, std2 = norm.fit(test_in_score)

    # 2. build the gaussian mixture model
    gmm = GMM(n_components=2)
    gmm.means_ = np.array([[mean1], [mean2]])
    # gmm.covariances_ = np.array([[[std1**2]], [[std2**2]]])
    gmm.covariances_ = np.array([[[std2 ** 2]], [[std2 ** 2]]])
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

    # center: x0
    # x0 = minimize(lambda x: (gmm_cdf(x, gmm) - 0.5)**2, x0=0).x[0]
    x0 = (mean1 + mean2) / 2
    return gmm, x0

# cumulative prob func
def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)

# the cumulative prob for a point
def cumulative_probability(x, gmm):
    return gmm_cdf(x, gmm)

# the cumulative prob for the symmtric point
def symmetric_cumulative_probability(x, x0, gmm):
    symmetric_x = 2 * x0 - x
    return gmm_cdf(symmetric_x, gmm)

all_w_res = []
all_w_res_dic = {}

def obtain_weights(input_x, gmm, x0):
    cp_x = cumulative_probability(input_x, gmm)
    cp_symmetric_x = symmetric_cumulative_probability(input_x, x0, gmm)

    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    scaling_factor = 10
    cp_sum *= scaling_factor
    range_th = 2 # 2 0.5
    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))

    # if w_res > 0.4:
    #     w_res = 1.2
    # elif w_res <= 0.4 and w_res > 0.3:
    #     w_res = w_res * 1.2 # 2
    # else:
    #     w_res=0
    if w_res > 0.9: # not_test_RD 0.997 0.995 0.98 # test_SD 0.999 0.990 0.9845
        w_res = 1.2
    elif w_res <= 0.4 and w_res > 0.3:
        w_res = w_res # 2
    else:
        w_res=0
    return w_res

def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--test_dataset', type=str, help='test_dataset')
    parser.add_argument('--base_model', type=str, help='base_model')
    parser.add_argument('--ood_base_model', type=str, default="roberta-large", help='base_model')
    parser.add_argument('--lora_weights', type=str, help='lora_model')
    parser.add_argument('--ood_weights', type=str, help='ood_model')
    parser.add_argument('--ood_type', type=str, help='ood type')
    parser.add_argument('--ood_setting', type=str, help='ood setting')
    parser.add_argument('--ood_setting_name', type=str, help='ood setting name')
    parser.add_argument('--seed', type=int, default=1, help='base_model')
    parser.add_argument('--unlearn_topic', type=str, help='file_name') # @ 수정 
    parser.add_argument('--quant_mode', type=str, default='none', choices=['none', 'backbone', 'all'])
    parser.add_argument('--result_file', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=0)
    # relearning attack (relearn_attack/attack_lora.py): 학습해서 저장 / 저장된 것 불러오기
    parser.add_argument('--attack_train_data', type=str, default=None)
    parser.add_argument('--attack_epochs', type=int, default=1)
    parser.add_argument('--attack_lr', type=float, default=1e-4)
    parser.add_argument('--attack_save', type=str, default=None)
    parser.add_argument('--attack_state', type=str, default=None)

    # Parsing arguments
    args = parser.parse_args()
    set_seed(args.seed)
    data_a = json.load(open(args.test_dataset))
    if args.max_samples > 0:
        data_a = data_a[:args.max_samples]
    base_model = args.base_model
    lora_weight = args.lora_weights
    max_batch_size = 1
    
    # types = args.ood_type.split("_")
    # ood_types = []
    # for i in types:
    #     if len(i) > 0:
    #         ood_types.append(i)
    ood_types = [args.ood_type]
    ood_setting = args.ood_setting
    ood_setting_names = args.ood_setting_name
    load_8bit = False
    ood_weights = []
    ood_type = "ocsvm"

    for i in ood_types:  # "biology", "physics", "chemistry"
        o_p = args.ood_weights + f"{ood_setting_names}_{i}_ood_{ood_setting_names}"
        ood_weights.append(o_p)

    result_file = args.result_file
    os.makedirs(os.path.dirname(result_file) or ".", exist_ok=True)
    print(result_file)

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')
    # O3_LLAMA31_ClincQA 는 transformers 5.x 로 저장돼 chat template 이 chat_template.jinja 에만 있음.
    # transformers 4.46 은 이 파일을 안 읽으므로 최신 버전과 같게 직접 넣는다.
    _jinja = os.path.join(base_model, "chat_template.jinja")
    if getattr(tokenizer, "chat_template", None) is None and os.path.isfile(_jinja):
        tokenizer.chat_template = open(_jinja).read()

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer.eos_token is None; cannot set pad_token safely.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    lora_target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ]
    config = AutoConfig.from_pretrained(base_model)
    config.lora_target_modules = lora_target_modules

    orthogonal_loss = False
    olora_weights = {}
    config.orthogonal_loss = orthogonal_loss
    config.orthogonal_loss_weight = 0.1

    model = LlamaForCausalLM_ood.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weight,
        torch_dtype=torch.bfloat16,
    )

    model.init_olora(orthogonal_loss=orthogonal_loss, olora_weights=olora_weights)
    model.init_active_adapters_d(active_adapters_d=['default'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    quant_rep = apply_fake_quant(model, args.quant_mode)
    quant_rep.update(lora_weights=lora_weight, base_model=base_model, ood_weights=ood_weights,
                     test_dataset=args.test_dataset, max_samples=args.max_samples)
    ##############################################
    ood_base_model = args.ood_base_model
    ood_tokenizer = RobertaTokenizer.from_pretrained(ood_base_model)
    ood_models = []
    ood_clrs = []
    ood_thresholds = []
    ood_x0 = []
    ood_mean_lists = []
    ood_precision_lists = []
    ood_fea_lists = []
    ood_gmm_w_cls = []

    for i in ood_weights:
        roberta_path = i + f"_roberta_{ood_type}"
        ocsvm_path = i + f"_{ood_type}.pkl"
        threshold_path = i + f"_threshold_{ood_type}.json"
        mean_list_path = i + f"_mean_list_{ood_type}.pt"
        precision_list_path = i + f"_precision_list_{ood_type}.pt"
        fea_list_path = i + f"_fea_list_{ood_type}.pt"
        gmm_w_path = i + f"_gmm_w_{ood_type}.pkl"

        ood_models.append(RobertaForSelector_inference(ood_base_model, lora_path=roberta_path, projection_dim=100).to(device))
        with open(ocsvm_path, "rb") as input_file:
            c_lr = pickle.load(input_file)
        ood_clrs.append(c_lr)
        with open(gmm_w_path , "rb") as input_file:
            gmm_w = pickle.load(input_file)
        ood_gmm_w_cls.append(gmm_w)
        with open(threshold_path) as f:
            threshold = json.load(f)
        ood_thresholds.append(threshold[1])
        ood_x0.append(threshold[0])

        ood_mean_lists.append(torch.load(mean_list_path, map_location=torch.device(device)))
        ood_precision_lists.append(torch.load(precision_list_path, map_location=torch.device(device)))
        ood_fea_lists.append(torch.load(fea_list_path, map_location=torch.device(device)))
    ##############################################
    def _ood_weight(raw_text):  # 아래 평가 루프와 같은 계산 (batch 1)
        ood_input = ood_tokenizer([raw_text], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        w_max = 0
        for i in range(len(ood_weights)):
            mah = ood_models[i].get_unsup_Mah_score_s(ood_input, ood_mean_lists[i], ood_precision_lists[i], ood_fea_lists[i])[:, 1:]
            w = obtain_weights(ood_clrs[i].score_samples(mah), ood_gmm_w_cls[i], ood_x0[i])
            w_max = max(w_max, w)
        return w_max

    if args.attack_train_data or args.attack_state:
        sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "relearn_attack"))
        from attack_lora import inject, attack_params, attack_state, load_attack_state
        names = inject(model)
        quant_rep["attack_lora_modules"] = len(names)
        if args.attack_state:
            load_attack_state(model, torch.load(args.attack_state))
            quant_rep["attack_state"] = args.attack_state
        else:
            import time as _t
            tr = json.load(open(args.attack_train_data))
            g = random.Random(args.seed)
            opt = torch.optim.AdamW(attack_params(model), lr=args.attack_lr, weight_decay=0.0)
            ws = [_ood_weight(str(ex.get("instruction", ""))) for ex in tr]   # 학습 문장도 OOD gate 를 거침
            accum, losses, t0 = 16, [], _t.time()
            for ep in range(args.attack_epochs):
                order = list(range(len(tr))); g.shuffle(order); tot = 0.0
                for k, j in enumerate(order):
                    ex = tr[j]
                    prompt = build_chat_prompt(tokenizer, instruction=str(ex.get("instruction", "")), system=None)
                    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    a_ids = tokenizer(str(ex["answer"]), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
                    ids = torch.tensor([p_ids + a_ids], device=device)
                    lab = torch.tensor([[-100] * len(p_ids) + a_ids], device=device)
                    model.init_oodweight(ood_weight=[1, ws[j]])
                    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lab)
                    (out.loss / accum).backward(); tot += out.loss.item()
                    if (k + 1) % accum == 0 or k + 1 == len(order):
                        torch.nn.utils.clip_grad_norm_(attack_params(model), 1.0)
                        opt.step(); opt.zero_grad(set_to_none=True)
                losses.append(round(tot / len(order), 4))
                print(f"[ATTACK] epoch {ep + 1}/{args.attack_epochs} loss {losses[-1]}", flush=True)
            quant_rep.update(attack_train_data=args.attack_train_data, attack_epochs=args.attack_epochs,
                             attack_lr=args.attack_lr, attack_epoch_loss=losses,
                             attack_train_ood_weight_mean=sum(ws) / len(ws), attack_train_sec=round(_t.time() - t0))
            if args.attack_save:
                os.makedirs(os.path.dirname(args.attack_save), exist_ok=True)
                torch.save(attack_state(model), args.attack_save)
        model.eval()

    max_new_tokens = 128
    save_every = 200

    contains_correct = 0
    exact_correct = 0

    results = []

    domain_total = defaultdict(int)
    domain_contains_correct = defaultdict(int)
    domain_exact_correct = defaultdict(int)

    for start_idx in tqdm(range(0, len(data_a), max_batch_size)):
        end_idx = min(start_idx + max_batch_size, len(data_a))
        batch = data_a[start_idx:end_idx]

        intents = [str(example["intent"]) for example in batch]
        # debug('intent is : ', intents)
        domains = [str(example.get("domain", "unknown")) for example in batch]

        llm_prompts = []
        ood_texts = []

        for example in batch:
            instruction = str(example.get("instruction", ""))
            input_text = str(example.get("input", ""))
            # OOD는 학습 때 text_input plain text를 봤으므로 raw text 유지
            if input_text:
                raw_text = f"{instruction}\n\n{input_text}"
            else:
                raw_text = instruction
            ood_texts.append(raw_text)
            # LLM은 chat template 사용
            llm_prompts.append(
                build_chat_prompt(
                    tokenizer,
                    instruction=raw_text,
                    system=None,
                )
            )

        ##############################################
        ood_input = ood_tokenizer(
            ood_texts, 
            padding='max_length', 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ) # 256
        max_ood = 0

        for i in range((len(ood_weights))):
            mah_score = ood_models[i].get_unsup_Mah_score_s(ood_input, 
                    ood_mean_lists[i], ood_precision_lists[i], ood_fea_lists[i])[:, 1:]
            test_score = ood_clrs[i].score_samples(mah_score)
            w_ood = obtain_weights(test_score, ood_gmm_w_cls[i], ood_x0[i])
            if w_ood > max_ood:
                max_ood = w_ood

        all_w_res.append(max_ood)
        dic_key = str(max_ood)[:5]
        if dic_key in all_w_res_dic:
            all_w_res_dic[dic_key] += 1
        else:
            all_w_res_dic[dic_key] = 1

        print("ood_weight: ", [1, max_ood])
        model.init_oodweight(ood_weight=[1, max_ood])
        ##############################################
        inputs = tokenizer(
            llm_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device) # @ 수정 : llama3.1-instruct

        with torch.no_grad():
            generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        sequences = generation_output.sequences
        gen_only_ids = sequences[:, input_ids.shape[1]:]
        output = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)
        debug("GEN ONLY OUTPUT :", output)

        for r_i, raw_text in enumerate(output):
            d = domains[r_i]
            domain_total[d] += 1

            resp = raw_text.strip()
            intent = resp.splitlines()[0].strip() if resp else "FAILED"
            intent = intent.replace("</s>", "").replace("<|eot_id|>", "").strip()
            if not intent:
                intent = "FAILED"

            pred_text = str(intent).strip().lower()
            gt_text = str(intents[r_i]).strip().lower()

            is_contains_correct = gt_text in pred_text
            is_exact_correct = gt_text == pred_text

            if is_contains_correct:
                contains_correct += 1
                domain_contains_correct[d] += 1

            if is_exact_correct:
                exact_correct += 1
                domain_exact_correct[d] += 1

            debug("intent : ", intent)
            debug("gt : ", intents[r_i])

            if is_exact_correct:
                print("exact correct:", pred_text, gt_text, flush=True)
            elif is_contains_correct:
                print("contains correct:", pred_text, gt_text, flush=True)
            else:
                print("pred/gt:", pred_text, gt_text, flush=True)

            results.append({
                "domain": d,
                "prompt": llm_prompts[r_i],
                "gt": intents[r_i],
                "pred": intent,
                "raw_output": resp,
                "contains_correct": is_contains_correct,
                "exact_correct": is_exact_correct,
                "ood_text": ood_texts[r_i],
                "ood_weight": float(max_ood),
            })

        #########################################
        contains_acc = contains_correct / len(results) * 100
        exact_acc = exact_correct / len(results) * 100

        if end_idx % save_every == 0 or end_idx == len(data_a):
            per_topic = {}
            for t in domain_total:
                tot = int(domain_total[t])
                contains_cor = int(domain_contains_correct[t])
                exact_cor = int(domain_exact_correct[t])

                per_topic[t] = {
                    "total": tot,
                    "contains_correct": contains_cor,
                    "exact_correct": exact_cor,
                    "contains_acc": (contains_cor / tot * 100) if tot > 0 else 0.0,
                    "exact_acc": (exact_cor / tot * 100) if tot > 0 else 0.0,
                }

            per_domain_sorted = dict(
                sorted(per_topic.items(), key=lambda kv: kv[1]["total"], reverse=True)
            )

            print(f"{len(results)}/{len(data_a)}, contains_correct: {contains_correct}, contains_acc: {round(contains_acc, 2)}%, exact_correct: {exact_correct}, exact_acc: {round(exact_acc, 2)}%, saving to {result_file}")
            
            data = {}
            data["contains_acc"] = contains_acc
            data["exact_acc"] = exact_acc
            data["contains_correct"] = contains_correct
            data["exact_correct"] = exact_correct
            data["len"] = len(results)
            data["results"] = results
            data["per_domain"] = per_domain_sorted
            data["_quant_meta"] = quant_rep

            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))

    print(np.mean(all_w_res), min(all_w_res), max(all_w_res))

def debug(*args):
    print(*args, file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()