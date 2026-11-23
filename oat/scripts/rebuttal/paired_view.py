"""Paired view for CS-8 vs Random-8 (AAMAS-27 rebuttal; code from the implementation plan).

usage: paired_view.py VIEW_DIR SEEDS CS_DIR RANDOM_DIR [BASE_DIR]
Checks manifests (same commit/ckpt/scene protocol/config), row counts, K=8/R=16, status, and
identical scene hashes per (task, init_id, repeat_seed). git_commit may differ between
variants: the driver (rebuttal/random8.sh) checks that only rebuttal files changed in between; copies the seed-filtered episodes.csv of
each variant into VIEW_DIR/<variant>/ for the strict aggregator. Any violation -> AssertionError.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

view = Path(sys.argv[1])
seeds = {int(s) for s in sys.argv[2].split(",")}
sources = [Path(s) for s in sys.argv[3:]]
variants = ("cs_prefix", "random", "base")[:len(sources)]
manifests = [json.loads((src / "manifest.json").read_text()) for src in sources]
ref = manifests[0]
same = ("checkpoint_sha256", "git_diff_sha256", "scene_protocol",
        "scene_list_sha256", "cfg_env_runner", "n_test", "test_start_seed",
        "n_parallel_envs", "use_k_tokens", "n_action_steps", "temperature",
        "topk", "entropy_threshold")
expected = {"n_test": 500, "test_start_seed": 10000, "n_parallel_envs": 4,
            "use_k_tokens": 8, "temperature": 1.0, "topk": 10, "entropy_threshold": 0}
assert ref["checkpoint_sha256"] and ref["scene_list_sha256"] and ref["git_commit"]
assert ref["scene_protocol"].startswith("seeded_final_reset_v2:")
rows_by_variant = {}
for v, src, m in zip(variants, sources, manifests):
    print(f"{v}: git_commit={m['git_commit']} src={src}")
scene_reference = None

for variant, src, manifest in zip(variants, sources, manifests):
    assert manifest["variant"] == variant, src
    assert manifest["bon_free"] == (1 if variant == "base" else 8), src
    for key, val in expected.items():
        assert manifest[key] == val, (src, key, manifest[key], val)
    for key in same:
        assert manifest[key] == ref[key], (src, "manifest mismatch", key)
    assert seeds <= set(manifest["gen_seeds"]), (src, "missing generation seeds")
    if variant == "random":
        assert manifest["bon_signal"] == "random" and manifest["selector_seed"] == 0, src
        assert set(manifest["gen_seeds"]) == seeds, src

    with (src / "episodes.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if int(r["repeat_seed"]) in seeds]
    assert len(rows) == 500 * len(seeds), (src, "wrong row count", len(rows))
    keys = set()
    scene_map = {}
    counts = Counter()
    wins = Counter()
    for row in rows:
        i, seed = int(row["init_id"]), int(row["repeat_seed"])
        scene = (row["task"], i)
        assert (*scene, seed) not in keys, (src, "duplicate", scene, seed)
        keys.add((*scene, seed))
        assert row["variant"] == variant and row["status"] == "ok", (src, row)
        assert row["episode_finished"] == "1" and row["success"] in {"0", "1"}, (src, row)
        assert row["k_tokens"] == "8" and row["r_exec"] == "16", (src, row)
        assert row["batch_envs"] == "4" and int(row["seed_label"]) == 10000 + i, (src, row)
        h = row["scene_hash"]
        assert re.fullmatch("[0-9a-fA-F]{16}", h), (src, row)
        assert scene_map.setdefault(scene, h) == h, (src, "scene differs across seeds", scene)
        counts[seed, row["task"]] += 1
        wins[seed] += int(row["success"])
    for seed in seeds:
        assert {i for _, i, s in keys if s == seed} == set(range(500)), (src, seed)
        assert sorted(n for (s, _), n in counts.items() if s == seed) == [50] * 10, (src, seed, counts)
        print(f"{variant} seed={seed}: {wins[seed]}/500 ({wins[seed]/5:.1f}%)")
    sig = {key: scene_map[key[:2]] for key in keys}
    if scene_reference is None:
        scene_reference = sig
    else:
        assert sig == scene_reference, (src, "unmatched keys or scene hashes")
    rows_by_variant[variant] = fields, rows

view.mkdir(parents=True, exist_ok=False)
for variant, (fields, rows) in rows_by_variant.items():
    dest = view / variant
    dest.mkdir()
    with (dest / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
