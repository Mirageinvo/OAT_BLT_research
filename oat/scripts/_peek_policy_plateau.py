#!/usr/bin/env python3
"""Print TopK SR trajectory + latest epoch hints for square/lift/coffee."""
import re
import pathlib

RUNS = {
    "square": "output/20260720/215024_train_oatpolicy_square_N200",
    "lift": "output/20260719/144024_train_oatpolicy_lift_N200",
    "coffee": "output/20260720/090816_train_oatpolicy_mw-coffee-pull_st_N50",
}
LOGS = {
    "square": "logs/train_policy_square_paper_s42.log",
    "lift": "logs/train_policy_lift_retrain_s7_n100.log",
    "coffee": "logs/train_oatpolicy_mw-coffee-pull_st_N50_s0.log",
}


def ckpt_hist(run: str):
    p = pathlib.Path(run) / "checkpoints"
    rows = []
    for f in p.glob("ep-*_sr-*.ckpt"):
        m = re.match(r"ep-(\d+)_sr-([0-9.]+)\.ckpt", f.name)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    rows.sort()
    return rows


def last_train_epoch(log: str):
    path = pathlib.Path(log)
    if not path.exists():
        return None
    # read last ~2MB
    data = path.read_bytes()[-2_000_000:].decode("utf-8", errors="ignore")
    eps = [int(x) for x in re.findall(r"Training epoch (\d+)", data)]
    return eps[-1] if eps else None


def main():
    for name, run in RUNS.items():
        print(f"==== {name} ====")
        hist = ckpt_hist(run)
        print("topk:", ", ".join(f"ep{e}={s:.3f}" for e, s in hist) or "(none)")
        cur = last_train_epoch(LOGS[name])
        print(f"last_Training_epoch_in_log ≈ {cur}")
        if not hist:
            continue
        best_e, best_s = max(hist, key=lambda x: (x[1], x[0]))
        last_e, last_s = hist[-1]
        print(f"best_topk=ep{best_e}@{best_s:.3f}  chronologically_last_topk=ep{last_e}@{last_s:.3f}")
        later = [x for x in hist if x[0] > best_e]
        if cur is not None and cur - best_e >= 200 and (not later or all(s <= best_s + 1e-9 for _, s in later)):
            print(f"PLATEAU: best at ep{best_e}; now~ep{cur} (+{cur-best_e}); no better topk since")
        elif later and all(s <= best_s + 1e-9 for _, s in later):
            print(f"SOFT_PLATEAU: best ep{best_e}; later topk saves did not improve SR")
        elif last_s + 1e-9 < best_s:
            print("DECLINE_IN_TOPK_SET: newest kept ckpt worse than best (topk window)")
        else:
            print("CLIMBING_OR_RECENT_BEST")
        print()


if __name__ == "__main__":
    main()
