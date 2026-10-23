#!/usr/bin/env python3
"""모든 latency / memory 측정을 같은 방식으로 감싸는 wrapper.

명령을 실행하는 동안 0.5초마다 기록:
  GPU   : nvidia-smi memory.used (GPU 별, CUDA context 포함) 최대값
  host  : 실행한 프로세스 트리(자식 포함) RSS 합 최대값
  load  : /proc/loadavg 1분 평균 (공유 컨테이너 → 측정 중 다른 부하 확인용), 측정 중 최대 / 평균
  기타 GPU 프로세스: 측정 대상이 아닌 GPU compute 프로세스가 보이면 기록 (다른 GPU idle 조건 확인)
출력: --out JSON (+ 명령 종료 코드). 명령 stdout/stderr 는 그대로 통과.
사용: python latency/monitor.py --out results/v2/mon_x.json -- python latency/measure_infer.py ...
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time


def tree_mem(pid):
    """pid 와 모든 자손: (pid 집합, RSS 합 bytes, PSS 합 bytes).
    RSS 합은 fork 된 프로세스(dataloader worker, DeepSpeed rank)가 공유하는 페이지를 여러 번 센다 → 보고는 PSS
    (공유 페이지를 공유 프로세스 수로 나눠 센 값, /proc/<pid>/smaps_rollup)."""
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            kids.setdefault(ppid, []).append(int(d))
        except (OSError, ValueError, IndexError):
            pass
    pids, rss, pss, stack = set(), 0, 0, [pid]
    while stack:
        p = stack.pop()
        pids.add(p)
        try:
            with open(f"/proc/{p}/statm") as f:
                rss += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            with open(f"/proc/{p}/smaps_rollup") as f:
                for line in f:
                    if line.startswith("Pss:"):
                        pss += int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
        stack += kids.get(p, [])
    return pids, rss, pss


def clean_flag(peak, before, cvd):
    """측정에 안 쓰는 GPU 가 측정 내내 비어 있었는지 (peak 메모리 < 100 MiB). 2 GPU 를 다 쓰면 None (프로세스 수로 판단)."""
    use = {int(x) for x in cvd.split(",")} if cvd else set(peak)
    others = [i for i in peak if i not in use]
    if not others:
        return None
    return all(peak[i] < 100 for i in others)


def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    mem = {}
    for line in out.strip().splitlines():
        i, m = [int(x) for x in line.split(",")]
        mem[i] = m
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    return mem, apps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--light", action="store_true",
                    help="시간 측정용: nvidia-smi + loadavg 만 (host 메모리 /proc 읽기 안 함 — smaps_rollup 읽기가 큰 프로세스를 느리게 함)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    base_mem, base_apps = smi()
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    st = {"gpu_peak_MiB": {}, "host_rss_peak_MB": 0.0, "host_pss_peak_MB": 0.0, "load1": [], "n_samples": 0,
          "max_apps_per_gpu": {}}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            mem, apps = smi()
            for i, m in mem.items():
                st["gpu_peak_MiB"][i] = max(st["gpu_peak_MiB"].get(i, 0), m)
            if not a.light:
                pids, rss, pss = tree_mem(proc.pid)
                st["host_rss_peak_MB"] = max(st["host_rss_peak_MB"], rss / 2**20)
                st["host_pss_peak_MB"] = max(st["host_pss_peak_MB"], pss / 2**20)
            # 컨테이너 안에서는 nvidia-smi 가 호스트 PID 를 보여 줘서 PID 로 우리 프로세스를 가릴 수 없음
            # → GPU 별 compute 프로세스 수 최대값 기록, 겹침 판단 = 측정 안 하는 GPU 의 메모리 / 프로세스 수
            cnt = {}
            for line in apps:
                cnt[line.split(",")[0].strip()] = cnt.get(line.split(",")[0].strip(), 0) + 1
            for u, c in cnt.items():
                st["max_apps_per_gpu"][u] = max(st["max_apps_per_gpu"].get(u, 0), c)
            with open("/proc/loadavg") as f:
                st["load1"].append(float(f.read().split()[0]))
            st["n_samples"] += 1
            stop.wait(a.interval)
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    rc = proc.wait()
    stop.set(); th.join()
    res = {"cmd": cmd, "rc": rc, "wall_sec": time.time() - t0,
           "gpu_peak_MiB": st["gpu_peak_MiB"], "gpu_used_before_MiB": base_mem, "gpu_apps_before": base_apps,
           "host_rss_peak_MB": st["host_rss_peak_MB"], "host_pss_peak_MB": st["host_pss_peak_MB"],
           "max_compute_apps_per_gpu_uuid": st["max_apps_per_gpu"],
           "foreign_gpu_pids": [],  # (호환용, 컨테이너에서는 PID 로 판단 불가 → clean / max_compute_apps 로 판단)
           "clean": clean_flag(st["gpu_peak_MiB"], base_mem, os.environ.get("CUDA_VISIBLE_DEVICES")),
           "load1_max": max(st["load1"]) if st["load1"] else None,
           "load1_mean": sum(st["load1"]) / len(st["load1"]) if st["load1"] else None,
           "n_cpu": os.cpu_count(), "n_samples": st["n_samples"], "light": a.light, "interval": a.interval,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[monitor] rc={rc} wall={res['wall_sec']:.0f}s gpu_peak={res['gpu_peak_MiB']} host_pss_peak={res['host_pss_peak_MB']:.0f}MB (rss {res['host_rss_peak_MB']:.0f}) clean={res['clean']} "
          f"load1 max/mean={res['load1_max']}/{res['load1_mean']:.2f} before={base_mem} apps_before={len(base_apps)}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
