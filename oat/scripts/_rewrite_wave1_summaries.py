"""One-shot: write matched_s10000 summary.json from existing eval_logs (no re-eval)."""
import json
import pathlib

SUITES = {
    "can": "output/20260706/173343_train_oatpolicy_can_N200/checkpoints/ep-1700_sr-0.940.ckpt",
    "coffee-pull": "output/20260711/134440_train_oatpolicy_mw-coffee-pull_st_N50/checkpoints/ep-1000_sr-0.432.ckpt",
    "stick-pull": "output/20260711/134439_train_oatpolicy_mw-stick-pull_st_N50/checkpoints/ep-0800_sr-0.212.ckpt",
}
N_EXP, BON_N, SEED, N_TEST = 5, 8, 10000, 50


def load(p):
    d = json.load(open(p))
    if "mean_success_rate_mean" in d:
        return float(d["mean_success_rate_mean"]), float(d.get("mean_success_rate_std", 0.0))
    mk = [k for k in d if str(k).endswith("mean_success_rate_mean")]
    if not mk:
        raise KeyError(p)
    return float(d[mk[0]]), float(d.get(mk[0].replace("_mean", "_std"), 0.0))


def main():
    for suite, base in SUITES.items():
        root = pathlib.Path(f"output/eval/matched_s{SEED}/{suite}")
        bpath = root / f"baseline_n{N_EXP}" / "eval_log.json"
        opath = root / f"bon_n{BON_N}_n{N_EXP}" / "eval_log.json"
        assert bpath.exists() and opath.exists(), (suite, bpath.exists(), opath.exists())
        bm, bs = load(bpath)
        om, os_ = load(opath)
        delta = 100.0 * (om - bm)
        summary = {
            "suite": suite,
            "suite_dir": suite,
            "protocol": {
                "test_start_seed": SEED,
                "selection_seed_note": "train TopK used seed 1000; paper report must use this disjoint pool",
                "n_test": N_TEST,
                "num_exp": N_EXP,
                "use_k_tokens": 8,
                "entropy_threshold": 0,
                "temperature": 1.0,
                "topk": 10,
                "bon_free": BON_N,
                "bon_signal": "vote",
                "episodes": f"{SEED}..{SEED + N_TEST - 1}",
            },
            "base_ckpt": base,
            "awr_ckpt": None,
            "artifacts": {
                "root": str(root),
                "baseline": str(root / f"baseline_n{N_EXP}"),
                "bon": str(root / f"bon_n{BON_N}_n{N_EXP}"),
                "awr": str(root / f"awr_n{N_EXP}"),
                "log": f"logs/matched_s{SEED}_{suite}_gpu*.log",
                "summary": str(root / "summary.json"),
                "note": "summary rewritten after mid-run triplet.sh edit desynced bash; eval_logs untouched",
            },
            f"baseline_n{N_EXP}": {"mean": bm, "std": bs, "pct": f"{100 * bm:.1f} ± {100 * bs:.1f}%"},
            f"bon_n{BON_N}_n{N_EXP}": {"mean": om, "std": os_, "pct": f"{100 * om:.1f} ± {100 * os_:.1f}%"},
            "awr_n5": None,
            f"delta_bon_n{BON_N}_n{N_EXP}_pp": delta,
        }
        out = root / "summary.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"{suite}: {summary[f'baseline_n{N_EXP}']['pct']} -> "
              f"{summary[f'bon_n{BON_N}_n{N_EXP}']['pct']}  Δ={delta:+.1f}pp  -> {out}")


if __name__ == "__main__":
    main()
